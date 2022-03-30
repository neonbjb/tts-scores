"""
Contrastive Transformer model trained for associating text and voice clips.
Closely based on the work of lucidrains in his DALLE repo:
https://github.com/lucidrains/DALLE-pytorch
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import nn, einsum
from tqdm import tqdm

from lib.tokenizer import text_to_sequence
from lib.utils import load_audio, to_mel, load_tsv


def exists(val):
    return val is not None


def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth = 1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args


class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim = self.dim, keepdim = True).detach()
        return x / maxes


class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0., mult = 4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}, layer_dropout = 0.):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


class PreShiftToken(nn.Module):
    def __init__(self, fn, image_size, seq_len):
        super().__init__()
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len

    def forward(self, x, **kwargs):
        n = x.shape[1]
        seq_len, image_size = self.seq_len, self.image_size
        img_seq_len = image_size ** 2
        text_len = seq_len - img_seq_len + 1
        padding = seq_len - n + 1

        # get text and image tokens

        x_text, x_img = x[:, :text_len], x[:, text_len:]
        x_img = F.pad(x_img, (0, 0, 0, padding))
        x_img = rearrange(x_img, 'b (h w) d -> b h w d', h = image_size)

        # shift 1 from the left for text tokens

        x_text_shift, x_text_pass = x_text.chunk(2, dim = -1)
        x_text_shift = F.pad(x_text_shift, (0, 0, 1, -1))
        x_text = torch.cat((x_text_shift, x_text_pass), dim = -1)

        # shift from top, left for image tokens

        x_img_shift_top, x_img_shift_left, *x_img_pass = x_img.chunk(4, dim = -1)
        x_img_shift_left = F.pad(x_img_shift_left, (0, 0, 1, -1))
        x_img_shift_top = F.pad(x_img_shift_top, (0, 0, 0, 0, 1, -1))
        x_img = torch.cat((x_img_shift_top, x_img_shift_left, *x_img_pass), dim = -1)

        # merge text and image sequence back together

        x_img = rearrange(x_img, 'b h w d -> b (h w) d')
        x = torch.cat((x_text, x_img[:, :-padding]), dim = 1)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., stable = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.stable = stable
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        softmax = torch.softmax if not self.stable else stable_softmax

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.,
        ff_dropout=0.,
    ):
        super().__init__()
        layers = nn.ModuleList([])

        for ind in range(depth):
            attn = Attention(dim, causal=False, seq_len=seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout)
            ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)
            layers.append(nn.ModuleList([
                LayerScale(dim, ind + 1, PreNorm(dim, attn)),
                LayerScale(dim, ind + 1, PreNorm(dim, ff))
            ]))

        execute_type = SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}
        self.layers = execute_type(layers, args_route=attn_route_map)

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)


class CLVP(nn.Module):
    def __init__(
            self,
            *,
            dim_text=512,
            dim_speech=512,
            dim_latent=512,
            num_text_tokens=256,
            text_enc_depth=6,
            text_seq_len=120,
            text_heads=8,
            num_speech_tokens=8192,
            speech_enc_depth=6,
            speech_heads=8,
            speech_seq_len=250,
            text_mask_percentage=0,
            voice_mask_percentage=0,
            mel_compression=256,
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        self.text_transformer = Transformer( seq_len=text_seq_len, dim=dim_text, depth=text_enc_depth,
                                            heads=text_heads)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        self.speech_enc = nn.Conv1d(80, dim_speech, kernel_size=3, padding=1)
        self.speech_pos_emb = nn.Embedding(num_speech_tokens, dim_speech)
        self.speech_transformer = Transformer(seq_len=speech_seq_len, dim=dim_speech,
                                              depth=speech_enc_depth, heads=speech_heads)
        self.to_speech_latent = nn.Linear(dim_speech, dim_latent, bias=False)

        self.temperature = nn.Parameter(torch.tensor(1.))
        self.text_mask_percentage = text_mask_percentage
        self.voice_mask_percentage = voice_mask_percentage
        self.mel_compression = mel_compression

    def get_text_projections(self, text, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones_like(text.float()).bool()
        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=text.device))
        enc_text = self.text_transformer(text_emb, mask=text_mask)
        text_latents = masked_mean(enc_text, text_mask, dim=1)
        return self.to_text_latent(text_latents).float()

    def get_speech_projection(self, mel, voice_mask=None):
        if voice_mask is None:
            voice_mask = torch.ones_like(mel[:,0,:].float()).bool()
        speech_emb = self.speech_enc(mel).permute(0,2,1)
        speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=mel.device))
        enc_speech = self.speech_transformer(speech_emb, mask=voice_mask)
        speech_latents = masked_mean(enc_speech, voice_mask, dim=1)
        return self.to_speech_latent(speech_latents).float()

    def forward(
            self,
            text,
            mel,
            return_loss=False
    ):
        b, device = text.shape[0], text.device
        if self.training:
            text_mask = torch.rand_like(text.float()) > self.text_mask_percentage
            voice_mask = torch.rand_like(mel[:,0,:].float()) > self.voice_mask_percentage
        else:
            text_mask = torch.ones_like(text.float()).bool()
            voice_mask = torch.ones_like(mel[:,0,:].float()).bool()

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))

        speech_emb = self.speech_enc(mel).permute(0,2,1)
        speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=device))

        enc_text = self.text_transformer(text_emb, mask=text_mask)
        enc_speech = self.speech_transformer(speech_emb, mask=voice_mask)

        text_latents = masked_mean(enc_text, text_mask, dim=1)
        speech_latents = masked_mean(enc_speech, voice_mask, dim=1)

        text_latents = self.to_text_latent(text_latents).float()
        speech_latents = self.to_speech_latent(speech_latents).float()

        text_latents, speech_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (text_latents, speech_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, speech_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, speech_latents) * temp
        labels = torch.arange(b, device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


class CLVPMetric:
    def __init__(self, device='cpu', pretrained_path='.data/clvp.pth'):
        self.device = device
        self.model = CLVP(dim_text=512, dim_latent=512, dim_speech=512, num_text_tokens=148, text_enc_depth=8,
                          text_seq_len=400, text_heads=8, speech_enc_depth=10, speech_heads=8, speech_seq_len=1000).eval().to(device)
        sd = torch.load(pretrained_path, map_location=device)
        self.model.load_state_dict(sd)

    def compute_frechet_distance(self, proj1, proj2):
        # I really REALLY FUCKING HATE that this is going to numpy. I do it because the `pytorch_fid` repo does it and
        # I want to retain parity (and torch.cov doesn't operate the same). Why does "pytorch_fid" operate in numpy land. WHY?
        proj1 = proj1.cpu().numpy()
        proj2 = proj2.cpu().numpy()
        mu1 = np.mean(proj1, axis=0)
        mu2 = np.mean(proj2, axis=0)
        sigma1 = np.cov(proj1, rowvar=False)
        sigma2 = np.cov(proj2, rowvar=False)
        return torch.tensor(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))

    def get_projection_for_files(self, files, verbose=True):
        with torch.no_grad():
            projections = []
            for file in tqdm(files, disable=not verbose):
                # Batching these could make this process faster, but they are being sequentially loaded anyways so whatever.
                audio = load_audio(str(file), 22050).to(self.device)[:1]  # Only take the first channel (if multiple are present)
                mel = to_mel(audio).unsqueeze(0)
                projections.append(self.model.get_speech_projection(mel).cpu())
            return projections

    def compute_fd(self, gen_path, real_path, verbose=True):
        gen_files = pathlib.Path(gen_path).rglob('*.wav')
        gen_projections = torch.cat(self.get_projection_for_files(gen_files, verbose), dim=0)
        real_files = pathlib.Path(real_path).rglob('*.wav')
        real_projections = torch.cat(self.get_projection_for_files(real_files, verbose), dim=0)
        return self.compute_frechet_distance(gen_projections, real_projections)

    def compute_clvp(self, tsv, verbose=True):
        with torch.no_grad():
            paths_and_text = load_tsv(tsv)
            ces = []
            for path, text in tqdm(paths_and_text, disable=not verbose):
                audio = load_audio(str(path), 22050).to(self.device)[:1]  # Only take the first channel (if multiple are present)
                mel = to_mel(audio).unsqueeze(0)
                ces.append(self.model(torch.tensor(text_to_sequence(text), device=self.device).unsqueeze(0), mel, False))
            return torch.stack(ces).mean()


if __name__ == '__main__':
    metric = CLVPMetric(device='cuda')
    print(metric.compute_fd('D:\\tmp\\tortoise-tts-eval\\redo_outlier', 'D:\\tmp\\tortoise-tts-eval\\real'))
