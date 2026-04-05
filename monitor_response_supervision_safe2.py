import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs'
STATUS_LOG = LOG_DIR / 'response_supervision_safe2_monitor.log'
DONE_FLAG = ROOT / 'results' / 'response_supervision_safe2_done.txt'

RUNS = [
    {
        'name': '80-89',
        'pid': 77621,
        'meta': ROOT / 'checkpoints_age_groups/80-89/stage2/partial_best/best_checkpoint_meta.json',
        'log': LOG_DIR / 'train_80_89_response_supervision_safe2.log',
    },
    {
        'name': '20-29-matched',
        'pid': 77627,
        'meta': ROOT / 'checkpoints_age_groups_matched/20-29/stage2/partial_best/best_checkpoint_meta.json',
        'log': LOG_DIR / 'train_20_29_matched_response_supervision_safe2.log',
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


def tail_line(log_path: Path) -> str:
    if not log_path.exists():
        return 'log missing'
    lines = log_path.read_text(errors='replace').splitlines()
    for line in reversed(lines):
        if 'Eval epoch ' in line or 'Finished in ' in line or 'Epoch ' in line:
            return line
    return lines[-1] if lines else 'log empty'


def run(cmd: list[str]):
    append(f"RUN {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def refresh_outputs():
    cmds = [
        ['python3', 'freeze_response_supervision_current_best.py'],
        ['python3', 'generate_response_supervision_interim_report.py'],
        ['python3', 'generate_response_supervision_multipanel.py'],
        ['python3', 'generate_response_supervision_agegroup_compare.py'],
        ['python3', 'generate_proposal_aligned_behavior_figures.py'],
    ]
    for cmd in cmds:
        run(cmd)


def main():
    append('=== safe2 monitor started ===')
    while True:
        alive_any = False
        metas_ready = []
        for run_info in RUNS:
            alive = pid_alive(run_info['pid'])
            alive_any = alive_any or alive
            meta_ready = run_info['meta'].exists() and run_info['meta'].stat().st_size > 0
            if meta_ready:
                metas_ready.append(run_info['name'])
            append(f"STATUS {run_info['name']} pid={run_info['pid']} alive={alive} meta_ready={meta_ready} tail={tail_line(run_info['log'])}")

        if len(metas_ready) == len(RUNS):
            append('Both partial-best metadata files detected. Refreshing outputs...')
            refresh_outputs()
            DONE_FLAG.parent.mkdir(parents=True, exist_ok=True)
            DONE_FLAG.write_text('safe2 partial-best outputs generated\n')
            append(f'Done flag written: {DONE_FLAG}')
            append('=== safe2 monitor finished ===')
            break

        if not alive_any and len(metas_ready) < len(RUNS):
            append('All runs stopped before both partial-best files appeared.')
            append('=== safe2 monitor finished with incomplete outputs ===')
            break

        time.sleep(60)


if __name__ == '__main__':
    main()
