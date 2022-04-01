# TTS Scores - Better evaluation metrics for text to speech models

TTS quality is a difficult thing to measure. Distance-based metrics are poor measurements because they only measure
similarity to the test set, not the realism of the generated speech. For this reason, most TTS papers rely on Mean
Opinion Scores to report model quality. Computing MOS involves humans in the loop, meaning it is costly and time
consuming. More importantly, it cannot be used while training to evaluate the real-time performance of a model while
training.

The field of image generation has settled on the usage of the Frechet Inception Distance and Inception Score metrics
to measure live performance. They are quite successful. I think we should take a page out of their book. But, we can
modernize this a little:

## Installation

tts-scores is available on pypi:
```shell
pip install tts-scores
```

## Contrastive Language-Voice Pretrained model (CLVP)

To this end, I trained a CLIP-like architecture with a twist: instead of measuring the similarity of text and images,
it measures the similarity of text and voice clips. I call this model CLVP. I believe such a model is an exceptional 
candidate for synthesizing a quality metric for Text->Voice models, much in the way that the Inception model is used for
FID and IS scores.

This repo contains the source code for CLVP and scripts that allow you to use it. I have built two metrics:

### CLVP Score

The CLVP score measures the distance predicted by CVLP between text and an audio clip where that text is spoken. A lower
score is better. It can be obtained by:

```python
from tts_scores.clvp import CLVPMetric
cv_metric = CLVPMetric(device='cuda')
score = cv_metric.compute_clvp('<path_to_your_tsv>', 'D:\\tmp\\tortoise-tts-eval\\real')
```

*Note: the format of the TSV file is described in a later section*

### CLVP Frechet Distance

Similar to FID, this metric compares the distribution of real spoken text with whatever your TTS model generatets.
It is particularly useful if you have a bunch of spoken text that you want to compare against but do not have the 
transcriptions for that text. For example, this is a good fit for measuring the performance of vocoders.

It works by computing the frechet distance of the outputs of the last layer of the CLVP model when fed data from
both distributions. Similar to FID, a lower score is better. It can be obtained by:

```python
from tts_scores.clvp import CLVPMetric
cv_metric = CLVPMetric(device='cuda')
score = cv_metric.compute_fd('<path_to_your_generated_audio>, '<path_to_your_real_audio>')
```

## wav2vec2 Intelligibility Score

One rather obvious way to compute the performance of a TTS system that I have not seen before is to leverage an ASR
system. If the goal is to produce intelligible speech - why not use a speech recognition system to measure that
intelligibility.

The intelligibility score packaged in this repo does exactly that. It takes in a list of generated and real audio files
and their transcriptions, and feeds everything through a pre-trained wav2vec2 model. The raw losses are returned. The
score is the difference between the wav2vec2 losses for the fake/generated samples and the real samples.

While CLVP scores take things like voice quality, voice diversity and prosody into account, the intelligibility score
only considers whether or not the speech your TTS model generates maps coherently to the text you put into it. For some
use cases, this will be the most important score. For others, all of the scores are important.

```python
from tts_scores.intelligibility import IntelligibilityMetric
is_metric = IntelligibilityMetric(device='cuda')
score = is_metric.compute_intelligibility('<path_to_your_tsv>', '<path_to_your_real_audio>')
```

## Scores from common models

A metric is only good if there are benchmarks which can be used as points of comparison. To this end, I computed
all of the scores in this repo on two high-performance TTS models:

1. Tacotron2+waveglow from [NVIDIA's repo](https://github.com/NVIDIA/tacotron2)
2. FastSpeech2+hifigan from [ming024's repo](https://github.com/ming024/FastSpeech2)

See the scores below:

# Citations

Please cite this repo if you use it in your repo:

```
@software{TTS-scores,
  author = {Betker, J ames},
  month = {4},
  title = {{TTS-scores}},
  url = {https://github.com/neonbjb/tts-scores},
  version = {1.0.0},
  year = {2022}
}
```