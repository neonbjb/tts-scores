import os

import torch
import torchaudio
from torch import nn
import numpy as np
from scipy.io.wavfile import read


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2 ** 31
    elif data.dtype == np.int16:
        norm_fix = 2 ** 15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.
    else:
        raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)


def load_audio(audiopath, sampling_rate):
    if audiopath[-4:] == '.wav':
        audio, lsr = load_wav_to_torch(audiopath)
    else:
        raise RuntimeError

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)


def load_tsv(filename):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = []
        base = os.path.dirname(filename)
        for line in f:
            components = line.strip().split('\t')
            filepaths_and_text.append([os.path.join(base, f'{components[1]}'), components[0]])
    return filepaths_and_text


class TorchMelSpectrogram(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0, mel_fmax=8000,
                 sampling_rate=22050, normalize=False, mel_norm_file='.data/mel_norms.pth'):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=normalize,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, inp):
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel


def to_mel(wav):
    return TorchMelSpectrogram()(wav.unsqueeze(0)).squeeze(0)