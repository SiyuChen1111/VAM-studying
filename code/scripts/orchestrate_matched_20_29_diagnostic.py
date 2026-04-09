import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs'
OUT_ROOT = ROOT / 'checkpoints_age_groups_diagnostic'
STATUS_LOG = LOG_DIR / 'matched_20_29_diagnostic.log'
DONE_FLAG = ROOT / 'results' / 'matched_20_29_diagnostic_done.txt'

RUNS = [
    {
        'name': 'run_A_control',
        'subdir': 'run_A_control',
        'args': ['--epochs', '10', '--scale_values', '0.1,0.3,0.5', '--lambda_rt', '1.0', '--lambda_choice', '2.0', '--lambda_cong', '0.3', '--choice_temperature', '0.05'],
    },
    {
        'name': 'run_B_choice_emphasis',
        'subdir': 'run_B_choice_emphasis',
        'args': ['--epochs', '10', '--scale_values', '0.1,0.3,0.5', '--lambda_rt', '1.0', '--lambda_choice', '3.0', '--lambda_cong', '0.3', '--choice_temperature', '0.10'],
    },
    {
        'name': 'run_C_choice_plus_cong',
        'subdir': 'run_C_choice_plus_cong',
        'args': ['--epochs', '10', '--scale_values', '0.1,0.3,0.5', '--lambda_rt', '1.0', '--lambda_choice', '3.0', '--lambda_cong', '0.5', '--choice_temperature', '0.10'],
    },
]


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
    append('=== matched 20-29 diagnostic started ===')
    caffeinate_proc = launch_caffeinate()

    for run in RUNS:
        output_root = OUT_ROOT / run['subdir']
        output_root.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{run['subdir']}.log"
        cmd = [
            'python3', '-u', 'train_age_groups_efficient.py',
            '--age_group', '20-29',
            '--cached_only',
            '--data_root', 'data_age_groups_matched',
            '--output_root', str(output_root),
            '--train_logits_path', str(ROOT / 'checkpoints_age_groups_matched/20-29/stage2/train_logits.npz'),
            '--test_logits_path', str(ROOT / 'checkpoints_age_groups_matched/20-29/stage2/test_logits.npz'),
        ] + run['args']
        append(f"RUN {' '.join(cmd)}")
        with open(log_path, 'w') as log_handle:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
            proc.wait()
        append(f"FINISHED {run['name']} returncode={proc.returncode}")
        if proc.returncode != 0:
            raise RuntimeError(f"Diagnostic run failed: {run['name']}")

    DONE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    DONE_FLAG.write_text('matched 20-29 diagnostic complete\n')
    append(f'Done flag written: {DONE_FLAG}')
    append('=== matched 20-29 diagnostic finished ===')
    if caffeinate_proc is not None and caffeinate_proc.poll() is None:
        caffeinate_proc.terminate()


if __name__ == '__main__':
    main()
