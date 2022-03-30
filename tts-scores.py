
# baseline:         29.36
# cond_free:        31.63
# typical 256:      29.23
# typical 256 .4:   28.51
import os

from lib.clvp import CLVPMetric
from lib.intelligibility import IntelligibilityMetric

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # Bleh.

    cv_metric = CLVPMetric(device='cuda')
    is_metric = IntelligibilityMetric(device='cuda')
    basedir = 'D:\\tmp\\tortoise-tts-eval\\std_sweep_diffusion'
    for sd in os.listdir(basedir):
        metric_dir = os.path.join(basedir, sd)
        fd = cv_metric.compute_fd(metric_dir, 'D:\\tmp\\tortoise-tts-eval\\real', verbose=False)
        clvp = cv_metric.compute_clvp(os.path.join(metric_dir, 'transcript.tsv'), 'D:\\tmp\\tortoise-tts-eval\\real', verbose=False)
        ism = is_metric.compute_intelligibility(os.path.join(metric_dir, 'transcript.tsv'), 'D:\\tmp\\tortoise-tts-eval\\real', verbose=False)
        print(f"{metric_dir}: FD: {fd}; CLVP: {clvp}; IS: {ism}")