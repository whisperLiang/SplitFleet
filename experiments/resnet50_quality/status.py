"""Show the current experiment and its actual measured concurrency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .native import BACKENDS
from .queue import progress, save_json, timestamp


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError):
        return None


def queue_status(pointer):
    """Report full runs from the current training queue."""
    queue = read(pointer['queue']) or {}
    if queue.get('source_identity') != pointer.get('source_identity'):
        return {'updated_utc': timestamp(), 'status': 'direct_queue_identity_invalid',
                'backends': [], 'completed_main_runs': 0, 'objective_proven': False}
    controller_alive = True
    if queue.get('controller_pid') and queue.get('status') in {'running', 'validating_results'}:
        import psutil
        from datetime import datetime
        try:
            controller = psutil.Process(queue['controller_pid'])
            started = datetime.fromisoformat(queue['started_utc']).timestamp()
            controller_alive = (controller.status() != psutil.STATUS_ZOMBIE
                                and abs(controller.create_time() - started) < 300
                                and any(name in controller.cmdline() for name in (
                                    'experiments.resnet50_quality.queue', 'experiments.resnet50_quality.campaign')))
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, ValueError):
            controller_alive = False
    rows = []
    selected = dict.fromkeys(job['backend'] for job in queue.get('jobs', []))
    for backend in selected:
        clients = BACKENDS[backend]
        jobs = [job for job in queue.get('jobs', []) if job['backend'] == backend]
        active = next((job for job in jobs if job['status'] == 'running'), None)
        interrupted = None
        if active and not controller_alive:
            import psutil
            try:
                process = psutil.Process(active['controller_pid'])
                alive = process.status() != psutil.STATUS_ZOMBIE and process.create_time() == active.get('controller_birth_time')
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                alive = False
            if not alive:
                interrupted, active = active, None
        latest = active or next((job for job in reversed(jobs) if job['status'] == 'passed'), None)
        completed = sum(job['status'] == 'passed' for job in jobs)
        row = {'backend': backend, 'clients': clients, 'requested_workers': clients,
               'planned_rounds': queue.get('all_rounds'),
               'source_identity': queue.get('source_identity'), 'phase': 'main',
               'status': 'main_interrupted' if interrupted else 'main_running' if active else 'main_queued' if any(
                   job['status'] == 'pending' for job in jobs) else
                   'main_complete' if completed == len(jobs) else 'main_incomplete',
               'main_completed_runs': completed, 'main_planned_runs': len(jobs),
               'failed_main_runs': sum(job['status'] == 'failed' for job in jobs),
               'requested_suffix_models': clients, 'actual_suffix_model_count': None,
               'actual_worker_count': None, 'peak_running_clients': None,
               'peak_computing_clients': None, 'mean_computing_clients': None,
               'active_main_runs': [], 'worker_plan': queue.get('worker_plan', {}).get(backend)}
        if interrupted:
            row['actual_worker_count'] = 0
            row['last_interrupted_run'] = {key: interrupted[key] for key in ('seed', 'method', 'output')}
        if latest:
            row['worker_limit'] = latest['worker_count']
            if active:
                row['active_main_runs'] = [{key: active[key] for key in ('seed', 'method', 'output')}]
                row['active_main_runs'][0]['progress'] = progress(active)
                import psutil
                try:
                    owner = psutil.Process(active['controller_pid'])
                    if owner.create_time() == active.get('controller_birth_time'):
                        workers = [p for p in owner.children(recursive=True)
                                   if p.name().startswith('ray::ClientAppActor') and p.status() != psutil.STATUS_ZOMBIE]
                        row['actual_worker_count'] = len(workers)
                        row['worker_count_scope'] = 'Observed live Ray ClientAppActor processes, including idle actors'
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            directory = Path(latest['output'])
            rounds = directory / 'round_metrics.jsonl'
            if rounds.exists():
                for line in reversed(rounds.read_text().splitlines()):
                    try:
                        observed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row.update(measurement_phase='main', measurement_method=latest['method'],
                               measurement_is_current_phase=True, measurement_clients=clients,
                               measured_round=observed.get('round_id'),
                               actual_suffix_model_count=observed.get('suffix_model_count'))
                    for key in ('peak_running_clients', 'mean_running_clients', 'peak_computing_clients',
                                'mean_computing_clients', 'resident_suffix_limit'):
                        row[key] = observed.get(key)
                    if not active:
                        row['actual_worker_count'] = observed.get('worker_processes_observed')
                    break
        rows.append(row)
    summary = read(Path(pointer['queue']).parent / 'summary/summary.json') or {}
    objective_proven = (queue.get('status') == 'passed' and summary.get('status') == 'passed'
                        and summary.get('comparisons', {}).get('pytorch', {}).get('both_converged_and_faster') is True)
    return {'updated_utc': timestamp(), 'queue': pointer['queue'],
            'status': queue.get('status', 'not_started') if controller_alive else 'interrupted',
            'controller_alive': controller_alive,
            'execution_mode': 'training',
            'planned_main_runs': queue.get('planned_runs'), 'completed_main_runs': sum(r['main_completed_runs'] for r in rows),
            'rounds_per_main_run': queue.get('all_rounds'),
            'backends': rows, 'objective_proven': objective_proven}


def current_status(root):
    root = Path(root).resolve()
    campaign = read(root / 'campaign.json')
    if campaign:
        from .campaign import campaign_status
        return campaign_status(campaign)
    queue = read(root / 'queue.json')
    if queue:
        return queue_status({'queue': str(root / 'queue.json'), 'source_identity': queue.get('source_identity')})
    return {"updated_utc": timestamp(), "status": "not_started", "completed_main_runs": 0,
            "objective_proven": False, "backends": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/resnet50_torch50'))
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        result = current_status(args.root)
        save_json(args.root / 'execution_status.json', result)
        if not args.watch:
            print(json.dumps(result, indent=2))
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
