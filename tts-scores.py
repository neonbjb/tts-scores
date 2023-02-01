import os

from tts_scores.clvp import CLVPMetric
from tts_scores.intelligibility import IntelligibilityMetric

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # Bleh.

    cv_metric = CLVPMetric(device='cuda')
    is_metric = IntelligibilityMetric(device='cuda')
    basedir = 'D:\\tmp\\tortoise-tts-eval\\'
    realdir = 'D:\\tmp\\tortoise-tts-eval\\real'
    for sd in os.listdir(basedir):
        metric_dir = os.path.join(basedir, sd)
        if not os.path.exists(os.path.join(metric_dir, 'transcript.tsv')):
            continue
        fd = cv_metric.compute_fd(metric_dir, realdir, verbose=False)
        clvp = cv_metric.compute_clvp(os.path.join(metric_dir, 'transcript.tsv'), realdir, verbose=False)
        ism = is_metric.compute_intelligibility(os.path.join(metric_dir, 'transcript.tsv'), realdir, verbose=False)
        print(f"{metric_dir}: FD: {fd}; CLVP: {clvp}; IS: {ism}")
