import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / 'logs'
RESULTS_DIR = ROOT / 'results'
STATUS_LOG = LOG_DIR / 'response_supervision_orchestrator.log'
DONE_FLAG = RESULTS_DIR / 'response_supervision_orchestrator_done.txt'

RUN_CONFIGS = [
    {
        'name': '80-89',
        'command': ['python3', '-u', 'train_age_groups_efficient.py', '--age_group', '80-89', '--cached_only'],
        'log_path': LOG_DIR / 'train_80_89_response_supervision_orchestrated.log',
        'partial_dir': ROOT / 'checkpoints_age_groups/80-89/stage2/partial_best',
    },
    {
        'name': '20-29-matched',
        'command': ['python3', '-u', 'train_age_groups_efficient.py', '--age_group', '20-29', '--cached_only', '--data_root', 'data_age_groups_matched', '--output_root', 'checkpoints_age_groups_matched'],
        'log_path': LOG_DIR / 'train_20_29_matched_response_supervision_orchestrated.log',
        'partial_dir': ROOT / 'checkpoints_age_groups_matched/20-29/stage2/partial_best',
    },
]


def append(msg: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with STATUS_LOG.open('a') as f:
        f.write(msg.rstrip() + '\n')


def launch_caffeinate():
    if shutil.which('caffeinate') is None:
        append('caffeinate not available; cannot actively prevent sleep')
        return None
    proc = subprocess.Popen(['caffeinate', '-dimsu'])
    append(f'caffeinate started pid={proc.pid}')
    return proc


def launch_run(run_cfg):
    log_handle = open(run_cfg['log_path'], 'w')
    proc = subprocess.Popen(run_cfg['command'], cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
    append(f"launched {run_cfg['name']} pid={proc.pid} cmd={' '.join(run_cfg['command'])}")
    return proc, log_handle


def all_partial_files_ready(partial_dir: Path):
    required = [
        partial_dir / 'best_config.partial.json',
        partial_dir / 'best_model_params.partial.npz',
        partial_dir / 'best_test_predictions.partial.npz',
        partial_dir / 'best_checkpoint_meta.json',
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in required)


def tail_line(log_path: Path):
    if not log_path.exists():
        return 'log missing'
    lines = log_path.read_text(errors='replace').splitlines()
    for line in reversed(lines):
        if 'Eval epoch ' in line or 'Finished in ' in line or 'Epoch ' in line or 'Best scale for ' in line:
            return line
    return lines[-1] if lines else 'log empty'


def refresh_outputs():
    cmds = [
        ['python3', 'freeze_response_supervision_current_best.py'],
        ['python3', 'generate_response_supervision_interim_report.py'],
        ['python3', 'generate_response_supervision_multipanel.py'],
        ['python3', 'generate_response_supervision_agegroup_compare.py'],
        ['python3', 'generate_proposal_aligned_behavior_figures.py'],
    ]
    for cmd in cmds:
        append(f"RUN {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    append('=== orchestrator started ===')
    caffeinate_proc = launch_caffeinate()
    processes = []
    handles = []
    for cfg in RUN_CONFIGS:
        proc, handle = launch_run(cfg)
        processes.append((cfg, proc))
        handles.append(handle)

    try:
        no_progress_cycles = {cfg['name']: 0 for cfg, _ in processes}
        last_tail = {cfg['name']: '' for cfg, _ in processes}
        while True:
            all_done = True
            all_ready = True
            for cfg, proc in processes:
                alive = proc.poll() is None
                all_done = all_done and (not alive)
                ready = all_partial_files_ready(cfg['partial_dir'])
                all_ready = all_ready and ready
                tail = tail_line(cfg['log_path'])
                if tail == last_tail[cfg['name']]:
                    no_progress_cycles[cfg['name']] += 1
                else:
                    no_progress_cycles[cfg['name']] = 0
                    last_tail[cfg['name']] = tail
                append(f"STATUS {cfg['name']} pid={proc.pid} alive={alive} partial_ready={ready} stall_cycles={no_progress_cycles[cfg['name']]} tail={tail}")

            if all_ready:
                append('All required partial-best files detected. Refreshing outputs...')
                refresh_outputs()
                DONE_FLAG.parent.mkdir(parents=True, exist_ok=True)
                DONE_FLAG.write_text('response supervision orchestrated run complete\n')
                append(f'Done flag written: {DONE_FLAG}')
                append('=== orchestrator finished ===')
                break

            if all_done and not all_ready:
                append('Training finished before all partial-best files became available.')
                append('=== orchestrator finished incomplete ===')
                break

            time.sleep(60)
    finally:
        for handle in handles:
            handle.close()
        if caffeinate_proc is not None and caffeinate_proc.poll() is None:
            caffeinate_proc.terminate()
            append('caffeinate terminated')


if __name__ == '__main__':
    main()
