import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs'
OUT_ROOT = ROOT / 'checkpoints_age_groups_noise_probe'
STATUS_LOG = LOG_DIR / 'matched_20_29_noise_probe.log'
DONE_FLAG = ROOT / 'results' / 'matched_20_29_noise_probe_done.txt'


def append(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with STATUS_LOG.open('a') as f:
        f.write(msg.rstrip() + '\n')


def launch_caffeinate():
    if shutil.which('caffeinate') is None:
        append('caffeinate not available')
        return None
    proc = subprocess.Popen(['caffeinate', '-dimsu'])
    append(f'caffeinate started pid={proc.pid}')
    return proc


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    append('=== matched 20-29 noise probe started ===')
    caffeinate_proc = launch_caffeinate()
    output_root = OUT_ROOT / 'run_noise_probe_005'
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / 'run_noise_probe_005.log'
    cmd = [
        'python3', '-u', 'train_age_groups_efficient.py',
        '--age_group', '20-29',
        '--cached_only',
        '--data_root', 'data_age_groups_matched',
        '--output_root', str(output_root),
        '--train_logits_path', str(ROOT / 'checkpoints_age_groups_matched/20-29/stage2/train_logits.npz'),
        '--test_logits_path', str(ROOT / 'checkpoints_age_groups_matched/20-29/stage2/test_logits.npz'),
        '--epochs', '10',
        '--scale_values', '0.1,0.3,0.5',
        '--lambda_rt', '1.0',
        '--lambda_choice', '3.0',
        '--lambda_cong', '0.3',
        '--choice_temperature', '0.10',
        '--fixed_noise_ampa', '0.05',
    ]
    append(f"RUN {' '.join(cmd)}")
    with open(log_path, 'w') as log_handle:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
        proc.wait()
    append(f'FINISHED returncode={proc.returncode}')
    if proc.returncode != 0:
        raise RuntimeError('Noise probe failed')
    DONE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    DONE_FLAG.write_text('matched 20-29 noise probe complete\n')
    append(f'Done flag written: {DONE_FLAG}')
    append('=== matched 20-29 noise probe finished ===')
    if caffeinate_proc is not None and caffeinate_proc.poll() is None:
        caffeinate_proc.terminate()


if __name__ == '__main__':
    main()
