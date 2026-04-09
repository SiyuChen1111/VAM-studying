import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs'
STATUS_LOG = LOG_DIR / 'matched_20_29_noise_probe_monitor.log'
RUN_LOG = LOG_DIR / 'run_noise_probe_005.log'
PARTIAL_DIR = ROOT / 'checkpoints_age_groups_noise_probe/run_noise_probe_005/20-29/stage2/partial_best'
DONE_FLAG = ROOT / 'results' / 'matched_20_29_noise_probe_done.txt'
PID = 42572


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


def tail_line(log_path: Path):
    if not log_path.exists():
        return 'log missing'
    lines = log_path.read_text(errors='replace').splitlines()
    for line in reversed(lines):
        if 'Eval epoch ' in line or 'Finished in ' in line or 'Epoch ' in line or 'Best scale for ' in line:
            return line
    return lines[-1] if lines else 'log empty'


def main():
    append('=== noise probe monitor started ===')
    while True:
        alive = pid_alive(PID)
        partial_ready = PARTIAL_DIR.exists() and any(PARTIAL_DIR.iterdir())
        append(f'STATUS pid={PID} alive={alive} partial_ready={partial_ready} tail={tail_line(RUN_LOG)}')

        if partial_ready:
            append(f'Partial best detected in {PARTIAL_DIR}')
            append('=== noise probe monitor finished ===')
            break

        if not alive:
            append('Training process stopped before partial-best was detected')
            append('=== noise probe monitor finished ===')
            break

        time.sleep(60)


if __name__ == '__main__':
    main()
