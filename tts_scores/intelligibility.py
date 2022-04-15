import os.path

import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC

from tts_scores.tokenizer import text_to_sequence
from tts_scores.utils import load_tsv, load_audio


class IntelligibilityMetric:
    """
    Defines the logic for computing an "intelligibility" score. The intelligibility score measures how well the text
    being spoken can be understood by an ASR model.

    It is computed from the CTC losses for a wav2vec model between a true audio sample and its transcription and a
    fake audio sample and the same transcription.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli").to(device).eval()

    def fetch_ctc_loss(self, sample, text_codes):
        with torch.no_grad():
            norm_s = (sample - sample.mean()) / torch.sqrt(sample.var() + 1e-7)
            norm_s = norm_s.squeeze(1)
            return self.model(input_values=norm_s, labels=text_codes).loss

    def compute_intelligibility(self, tsv_file, real_base_dir=None, verbose=True):
        """
        Computes the intelligibility score and returns it.
        :param tsv_file: A path to a Tab-Separated-Value file that follows this format:
                         {transcription}\t{relative path to TTS audio file}\n
        :param real_base_dir: A folder containing real spoken audio files, with the same basenames as the TTS-generated
                              audio files from the tsv_file above. If None, one-sided intelligibility will be computed,
                              which does not take into account natural intelligibility losses from the ground truth data.
        :param verbose: When true, a TQDM bar showing metric computation status will be shown.
        :return: The mean intelligibility score for the provided data.
        """
        paths_and_text = load_tsv(tsv_file)
        ils = []
        for path, text in tqdm(paths_and_text, disable=not verbose):
            text_codes = torch.tensor(text_to_sequence(text), device=self.device)
            audio = load_audio(str(path), 16000).to(self.device)[:1]
            il = self.fetch_ctc_loss(audio, text_codes)
            if real_base_dir is not None:
                real_path = os.path.join(real_base_dir, os.path.basename(path))
                assert os.path.exists(real_path), real_path
                real_audio = load_audio(str(real_path), 16000).to(self.device)[:1]
                il = il - self.fetch_ctc_loss(real_audio, text_codes)
            if torch.isnan(il) or torch.isinf(il):
                continue
            ils.append(il)
        return torch.stack(ils).mean()
