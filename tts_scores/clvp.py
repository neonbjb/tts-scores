import math
import os.path
import pathlib
from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_fid.fid_score import calculate_frechet_distance
from torch import einsum, distributed
from torch.distributed import get_world_size
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from tts_scores.transformers import ContinuousTransformerWrapper, Encoder
from tts_scores.tokenizer import text_to_sequence, VoiceBpeTokenizer
from tts_scores.utils import load_audio, to_mel, load_tsv


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, mask=None):
        if self.do_checkpoint:
            if mask is not None:
                return checkpoint(self._forward, x, mask)
            else:
                return checkpoint(self._forward, x)
        else:
            return self._forward(x, mask)

    def _forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


def exists(val):
    return val is not None


def masked_mean(t, mask):
    t = t.masked_fill(~mask, 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)


class CollapsingTransformer(nn.Module):
    def __init__(self, model_dim, output_dims, heads, dropout, depth, mask_percentage=0, **encoder_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(
            max_seq_len=-1,
            use_pos_emb=False,
            attn_layers=Encoder(
                dim=model_dim,
                depth=depth,
                heads=heads,
                ff_dropout=dropout,
                ff_mult=1,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                **encoder_kwargs,
            ))
        self.pre_combiner = nn.Sequential(nn.Conv1d(model_dim, output_dims, 1),
                                          AttentionBlock(output_dims, num_heads=heads, do_checkpoint=False),
                                          nn.Conv1d(output_dims, output_dims, 1))
        self.mask_percentage = mask_percentage

    def forward(self, x, **transformer_kwargs):
        h = self.transformer(x, **transformer_kwargs)
        h = h.permute(0,2,1)
        h = checkpoint(self.pre_combiner, h).permute(0,2,1)
        if self.training:
            mask = torch.rand_like(h.float()) > self.mask_percentage
        else:
            mask = torch.ones_like(h.float()).bool()
        return masked_mean(h, mask)


class ConvFormatEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(*args, **kwargs)

    def forward(self, x):
        y = self.emb(x)
        return y.permute(0,2,1)


class CLVP(nn.Module):
    """
    Contrastic Language-Voice Pretraining model for generating embedding that can be used to associate text and
    speech clips.
    """

    def __init__(
            self,
            model_dim=512,
            transformer_heads=8,
            dropout=.1,
            num_text_tokens=256,
            text_enc_depth=6,
            text_mask_percentage=0,
            conditioning_enc_depth=4,
            mel_channels=80,
            mel_codes=None,
            speech_enc_depth=6,
            speech_mask_percentage=0,
            latent_multiplier=4,
            is_distributed=False,
    ):
        super().__init__()
        latent_dim = latent_multiplier*model_dim
        self.temperature = nn.Parameter(torch.tensor(1.))

        self.cond_emb = nn.Sequential(nn.Conv1d(mel_channels, model_dim//2, kernel_size=5, stride=2, padding=2),
                                      nn.Conv1d(model_dim//2, model_dim, kernel_size=3, stride=2, padding=1))
        self.conditioning_transformer = CollapsingTransformer(model_dim, model_dim*2, transformer_heads, dropout, conditioning_enc_depth, 0)

        self.text_emb = nn.Embedding(num_text_tokens, model_dim)
        self.text_transformer = CollapsingTransformer(model_dim, latent_dim, transformer_heads, dropout, text_enc_depth, text_mask_percentage, use_rms_scaleshift_norm=True)
        self.to_text_latent = nn.Linear(latent_dim, latent_dim, bias=False)

        self.distributed = is_distributed

        if mel_codes is None:
            self.speech_emb = nn.Conv1d(mel_channels, model_dim, kernel_size=5, padding=2)
        else:
            self.speech_emb = ConvFormatEmbedding(mel_codes, model_dim)
        self.speech_transformer = CollapsingTransformer(model_dim, latent_dim, transformer_heads, dropout, speech_enc_depth, speech_mask_percentage)
        self.to_speech_latent = nn.Linear(latent_dim, latent_dim, bias=False)

    def get_grad_norm_parameter_groups(self):
        return {
            'conditioning': list(self.conditioning_transformer.parameters()),
            'text': list(self.text_transformer.parameters()),
            'speech': list(self.speech_transformer.parameters()),
        }

    def forward(
            self,
            text,
            mel_input,
            mel_cond,
            return_loss=False
    ):
        device = text.device

        text_emb = self.text_emb(text)
        speech_emb = self.speech_emb(mel_input).permute(0,2,1)

        unused_params = []
        cond_emb = self.cond_emb(mel_cond).permute(0,2,1)
        enc_cond = self.conditioning_transformer(cond_emb)
        enc_text = self.text_transformer(text_emb, norm_scale_shift_inp=enc_cond)
        enc_speech = self.speech_transformer(speech_emb)

        text_latents = self.to_text_latent(enc_text)
        speech_latents = self.to_speech_latent(enc_speech)

        text_latents, speech_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (text_latents, speech_latents))
        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, speech_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, speech_latents) * temp
        labels = torch.arange(text_latents.shape[0], device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        # Involve probabilistic or possibly unused parameters in loss so we don't get DDP errors.
        extraneous_addition = 0
        for p in unused_params:
            extraneous_addition = extraneous_addition + p.mean()
        loss = loss + extraneous_addition * 0
        return loss

    def get_speech_projection(self, mel):
        speech_emb = self.speech_emb(mel).permute(0,2,1)
        enc_speech = self.speech_transformer(speech_emb)
        speech_latents = self.to_speech_latent(enc_speech)
        return speech_latents


class CLVPMetric:
    def __init__(self, device='cpu', pretrained_path='.data/clvp.pth'):
        self.device = device
        self.model = CLVP(model_dim=512, transformer_heads=8, dropout=0, num_text_tokens=256, text_enc_depth=8,
                          text_mask_percentage=0, conditioning_enc_depth=4, speech_enc_depth=8,
                          speech_mask_percentage=0, latent_multiplier=2).eval().to(device)
        sd = torch.load(pretrained_path, map_location=device)
        self.model.load_state_dict(sd)
        self.tokenizer = VoiceBpeTokenizer()

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

    def compute_clvp(self, tsv, real_dir, verbose=True):
        with torch.no_grad():
            paths_and_text = load_tsv(tsv)
            ces = []
            for path, text in tqdm(paths_and_text, disable=not verbose):
                audio = load_audio(str(path), 22050).to(self.device)[:1]  # Only take the first channel (if multiple are present)
                mel = to_mel(audio).unsqueeze(0)
                real_path = os.path.join(real_dir, os.path.basename(str(path)))
                cond_audio = load_audio(real_path, 22050).to(self.device)[:1]
                cond_mel = to_mel(cond_audio).unsqueeze(0)
                text_codes = torch.tensor(self.tokenizer.encode(text), device=self.device).unsqueeze(0)
                ces.append(self.model(text_codes, mel, cond_mel, False))
            return torch.stack(ces).mean()
            
    def compute_clvp_directly(self, paths_and_text, real_dir, verbose=True):
        with torch.no_grad():
            ces = []
            for path, text in tqdm(paths_and_text, disable=not verbose):
                audio = load_audio(str(path), 22050).to(self.device)[:1]  # Only take the first channel (if multiple are present)
                mel = to_mel(audio).unsqueeze(0)
                real_path = os.path.join(real_dir, os.path.basename(str(path)))
                cond_audio = load_audio(real_path, 22050).to(self.device)[:1]
                cond_mel = to_mel(cond_audio).unsqueeze(0)
                text_codes = torch.tensor(self.tokenizer.encode(text), device=self.device).unsqueeze(0)
                ces.append(self.model(text_codes, mel, cond_mel, False))
            return torch.stack(ces).mean()


if __name__ == '__main__':
    clvp = CLVP()
    clvp(torch.randint(0,256,(2,120)),
         torch.randn(2,80,100),
         torch.randn(2,80,95),
         return_loss=True)
    nonloss = clvp(torch.randint(0,256,(2,120)),
         torch.randn(2,80,100),
         torch.randn(2,80,95),
         return_loss=False)
    clvp.get_speech_projection(torch.randn(2,80,95))
    clvp = CLVP(mel_codes=8192)
    clvp(torch.randint(0,256,(2,120)),
         torch.randint(0,8192,(2,150)),
         torch.randn(2,80,95),
         return_loss=True)
    print(nonloss.shape)
