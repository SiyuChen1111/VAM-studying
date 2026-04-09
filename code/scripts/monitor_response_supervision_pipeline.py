import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs'
STATUS_LOG = LOG_DIR / 'response_supervision_pipeline_monitor.log'
DONE_FLAG = ROOT / 'results' / 'response_supervision_pipeline_done.txt'

RUNS = [
    {
        'name': '80-89',
        'pid': 67377,
        'log': LOG_DIR / 'train_80_89_response_supervision_rerun.log',
    },
    {
        'name': '20-29-matched',
        'pid': 67381,
        'log': LOG_DIR / 'train_20_29_matched_response_supervision_rerun.log',
    },
]


def append(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with STATUS_LOG.open('a') as f:
        f.write(msg.rstrip() + '\n')


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def last_eval_line(log_path: Path) -> str:
    if not log_path.exists():
        return 'log missing'
    lines = log_path.read_text(errors='replace').splitlines()
    for line in reversed(lines):
        if 'Eval epoch ' in line or 'Finished in ' in line or 'Best scale for ' in line:
            return line
    return lines[-1] if lines else 'log empty'


def run_command(cmd: list[str]):
    append(f"RUN {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def refresh_outputs():
    commands = [
        ['python3', 'freeze_response_supervision_current_best.py'],
        ['python3', 'generate_response_supervision_interim_report.py'],
        ['python3', 'generate_response_supervision_multipanel.py'],
        ['python3', 'generate_response_supervision_agegroup_compare.py'],
        ['python3', 'generate_proposal_aligned_behavior_figures.py'],
    ]
    for cmd in commands:
        run_command(cmd)


def main():
    append('=== monitor started ===')
    append(f'Watching PIDs: {[r["pid"] for r in RUNS]}')

    while True:
        alive = []
        for run in RUNS:
            is_alive = pid_alive(run['pid'])
            alive.append(is_alive)
            append(f"STATUS {run['name']} pid={run['pid']} alive={is_alive} tail={last_eval_line(run['log'])}")

        if not any(alive):
            append('All response-supervision runs finished. Refreshing outputs...')
            refresh_outputs()
            DONE_FLAG.parent.mkdir(parents=True, exist_ok=True)
            DONE_FLAG.write_text('response supervision pipeline complete\n')
            append(f'Done flag written: {DONE_FLAG}')
            append('=== monitor finished ===')
            break

        time.sleep(60)


if __name__ == '__main__':
    main()
