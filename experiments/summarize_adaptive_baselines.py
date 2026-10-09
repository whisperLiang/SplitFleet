"""Derive actual-device adaptation endpoints without changing raw receipts.

Only completely validated arms enter the summary. All preplanned arms must have terminal outcomes before cohort-level inference.
Failed arms remain explicit; successful-pair estimates disclose their selection.
Use a fresh --output for every derivation. Synchronization waits use one
coordinator clock and are distinct from server-queue attribution.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
from experiments.common.artifact_retention import verify_retained_or_pruned

ROOT = P = OUTPUT = None


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    state = read(ROOT / 'execution_status.json')
    arms = [entry for entry in state['attempts'] if entry['status'] == 'completed']
    output = OUTPUT or ROOT / 'analysis_checkpoints' / ('completed_' + str(len(arms)))
    output.mkdir(parents=True, exist_ok=False)
    summary, clients, rounds, communication, hashes = [], [], [], [], {}
    for arm in arms:
        folder = Path(arm['path'])
        result = read(folder / 'result.json')
        validation = read(folder / 'validation_report.json')
        assert validation['valid'] and read(folder / 'physical_validation.json')['status'] == 'passed'
        assert read(folder / 'runtime_manifest.json')['source_identity'] == state['source_identity']
        assert not result['fit_failures']
        records = result['physical_placement_experiment']['round_records']
        catalog = {candidate['boundary']: candidate for candidate in result['candidate_catalog']}
        assert len(catalog) == (242 if arm['policy'] == 'CoSplit-UCB-full' else 72)
        row = {'seed': arm['seed'], 'policy': arm['policy'], 'rounds': result['rounds'],
            'primary': arm['seed'] in P['seeds'], 'server_duration_sec': result['server_duration_sec'],
            'controller_arm_wall_sec': arm['finished_unix'] - arm['started_unix'],
            'final_accuracy': result['evaluation_records'][-1]['metrics']['accuracy'],
            'final_macro_f1': result['evaluation_records'][-1]['metrics']['macro_f1'],
            'mean_round_dispatch_sec': statistics.mean(record['coordinator_dispatch_through_aggregation_ms']/1000 for record in records),
            'mean_round_training_sec': statistics.mean(record['coordinator_round_training_ms']/1000 for record in records),
            'first_round_configuration_sec': (records[0]['configured_monotonic_ns']-records[0]['configure_started_monotonic_ns'])/1e9,
            'candidate_count': len(catalog), 'run_dir': str(folder)}
        if row['primary']:
            for phase, condition in [('native', lambda rnd: rnd < P['change_round']), ('capped', lambda rnd: rnd >= P['change_round'])]:
                chosen = [record for record in records if condition(record['round_id'])]
                row[phase+'_mean_dispatch_sec'] = statistics.mean(record['coordinator_dispatch_through_aggregation_ms']/1000 for record in chosen)
        for target in P['quality_thresholds']:
            reached = next((evaluation for evaluation in result['evaluation_records'] if evaluation['metrics']['accuracy'] >= target), None)
            row['ttq_'+str(target)+'_sec'] = reached['elapsed_sec'] if reached else None
            row['ttq_'+str(target)+'_round'] = reached['round_id'] if reached else None
            row['ttq_'+str(target)+'_reached'] = reached is not None
        summary.append(row)
        downloads = {(record['round_id'], record['cid']): record['model_state_download_bytes'] for record in result['state_download_records']}
        previous = {}
        for record in records:
            rnd = record['round_id']
            fits = [fit for fit in result['fit_records'] if fit['round_id'] == rnd]
            assert len(fits) == 3
            starts = [fit['metrics']['coordinator_fit_started_monotonic_ns'] for fit in fits]
            finishes = [fit['metrics']['coordinator_fit_finished_monotonic_ns'] for fit in fits]
            assert all(0 < begin <= end <= record['training_finished_monotonic_ns'] for begin, end in zip(starts, finishes))
            intervals = [(end-begin)/1e9 for begin, end in zip(starts, finishes)]
            waits = [(max(finishes)-end)/1e9 for end in finishes]
            lane = [step for step in result['physical_placement_experiment']['server_lane_step_receipts'] if step['round_id'] == rnd]
            round_row = {'seed': arm['seed'], 'policy': arm['policy'], 'round': rnd, 'primary': row['primary'],
                'configured_cap_mbps': record['cap_mbps'], 'dispatch_aggregation_sec': record['coordinator_dispatch_through_aggregation_ms']/1000,
                'training_including_configuration_sec': record['coordinator_round_training_ms']/1000,
                'client_rpc_max_sec': max(intervals), 'client_rpc_median_sec': statistics.median(intervals),
                'client_rpc_min_sec': min(intervals), 'client_rpc_straggler_ratio': max(intervals)/statistics.median(intervals),
                'client_rpc_duration_spread_sec': max(intervals)-min(intervals),
                'coordinator_sync_wait_max_sec': max(waits), 'coordinator_sync_wait_sum_sec': sum(waits),
                'dispatch_start_spread_sec': (max(starts)-min(starts))/1e9,
                'fleet_server_queue_wait_sec': sum(step['server_queue_wait_ms'] for step in lane)/1000,
                'accuracy': result['evaluation_records'][rnd-1]['metrics']['accuracy'],
                'macro_f1': result['evaluation_records'][rnd-1]['metrics']['macro_f1']}
            rounds.append(round_row)
            for fit, interval, waiting in zip(fits, intervals, waits):
                metrics = fit['metrics']; boundary = metrics['boundary']; client = metrics['logical_client_id']
                assert boundary in catalog
                assert metrics['num_examples'] == result['partition_sizes'][str(metrics['client_index'])]
                components = {}
                for component in ('client_forward', 'client_backward', 'server_service', 'network_roundtrip'):
                    assert int(metrics[component+'_samples']) == int(metrics['num_batches'])
                    components[component+'_total_sec'] = metrics[component+'_mean_ms'] * metrics[component+'_samples']/1000
                clients.append({'seed': arm['seed'], 'policy': arm['policy'], 'round': rnd, 'primary': row['primary'],
                    'client': client, 'device': metrics['device'], 'boundary': boundary,
                    'graph_position_ratio': catalog[boundary]['graph_position_ratio'],
                    'num_examples': fit['num_examples'], 'num_batches': metrics['num_batches'],
                    'actual_fit_rpc_sec': interval, 'coordinator_observed_sync_wait_sec': waiting,
                    'prefix_compute_sec': metrics['prefix_compute_sec'], **components,
                    'model_state_upload_bytes': metrics['model_state_upload_bytes'],
                    'model_state_download_bytes': downloads[(rnd, fit['cid'])],
                    'activation_upload_bytes': metrics['activation_upload_bytes'],
                    'gradient_download_bytes': metrics['gradient_download_bytes'],
                    'cut_switched': client in previous and previous[client] != boundary,
                    'configured_cap_mbps': record['cap_mbps'], 'per_client_server_queue_wait_sec': None,
                    'per_client_server_queue_attribution_known': False})
                previous[client] = boundary
        communication.append({'seed': arm['seed'], 'policy': arm['policy'], **read(folder/'communication.json')})
        for filename, expected in read(folder/'raw_receipts_sha256.json').items():
            verify_retained_or_pruned(folder/filename, expected)
            hashes[str(folder/filename)] = expected
    failures = []
    for arm in state['attempts']:
        if arm['status'] != 'failed' or arm['seed'] not in P['seeds']:
            continue
        folder = Path(arm['path'])
        progress = read(folder/'result.progress.json') if (folder/'result.progress.json').exists() else {}
        lane_path = folder/'result.server_lane.jsonl'
        partial_lane = [json.loads(line) for line in lane_path.read_text().splitlines() if line.strip()] if lane_path.exists() else []
        failure_round = max((step['round_id'] for step in partial_lane), default=None)
        failures.append({'seed': arm['seed'], 'policy': arm['policy'], 'status': 'failed',
            'failure_kind': 'nonfinite_training_loss' if 'Split suffix loss is nonfinite' in (folder/'stdout.log').read_text() else 'other',
            'completed_rounds': len(progress.get('round_records', [])),
            'failure_round': failure_round,
            'recorded_suffix_steps_in_failure_round': sum(step['round_id']==failure_round for step in partial_lane),
            'network_phase_at_failure': None if failure_round is None else ('native' if failure_round<P['change_round'] else 'capped'),
            'elapsed_until_failure_sec': arm['finished_unix']-arm['started_unix'],
            'final_accuracy': None, 'completed_job_duration_sec': None,
            'interpretation': 'Failure is an outcome; partial rounds do not replace complete-job metrics.', 'run_dir': str(folder)})
        for filename, expected in read(folder/'raw_receipts_sha256.json').items():
            verify_retained_or_pruned(folder/filename, expected)
            hashes[str(folder/filename)] = expected
    complete_blocks = []
    pairs = []
    for seed in P['seeds']:
        selected = {row['policy']: row for row in summary if row['seed'] == seed}
        if set(selected) == set(P['policies']):complete_blocks.append(seed)
        for reference in ('CoSplit-UCB-full', 'CoSplit-UCB-layer'):
            for comparator in P['policies']:
                if comparator == reference or reference not in selected or comparator not in selected:
                    continue
                current = selected[reference]; other = selected[comparator]
                pairs.append({'seed': seed, 'reference': reference, 'comparator': comparator,
                    'same_domain': current['candidate_count'] == other['candidate_count'],
                    'server_time_ratio': current['server_duration_sec']/other['server_duration_sec'],
                    'server_time_difference_sec': current['server_duration_sec']-other['server_duration_sec'],
                    'accuracy_difference': current['final_accuracy']-other['final_accuracy']})
    result = {'status': state['status'], 'source_identity': state['source_identity'], 'physical_measurement': True,
        'analysis_source_sha256': digest(Path(__file__)),
        'completed_arms': len(arms), 'primary_completed': sum(row['primary'] for row in summary),
        'primary_planned': 30, 'summaries': summary, 'complete_paired_seeds': complete_blocks,
        'primary_failed': len(failures), 'primary_terminal': sum(row['primary'] for row in summary)+len(failures), 'failures': failures,
        'paired': pairs, 'limits': ['Controller adaptations, not full-system reproductions.',
            'Per-client server queue attribution unknown; actual synchronization waits use one coordinator clock.',
            'Small fixed-work study; separate accuracy/F1 and TTQ, no full convergence claim.',
            'Same actor reused across five workload seeds; these are not five independently pretrained PPO policies.',
            'Client arrival/update order is real and not forced identical between methods.']}
    policy_summaries = []
    for policy in P['policies']:
        selected = [row for row in summary if row['primary'] and row['policy'] == policy]
        if not selected:
            continue
        chosen_rounds = [row for row in rounds if row['primary'] and row['policy'] == policy]
        stats = {'policy': policy, 'paired_n': len(selected), 'candidate_count': selected[0]['candidate_count'],
                 'failed_n': sum(row['policy']==policy for row in failures), 'planned_n':len(P['seeds']),
                 'summary_scope':'completed jobs only; failures reported separately'}
        for key in ('server_duration_sec', 'final_accuracy', 'final_macro_f1', 'mean_round_dispatch_sec',
                    'first_round_configuration_sec', 'native_mean_dispatch_sec', 'capped_mean_dispatch_sec'):
            values = [row[key] for row in selected]
            stats[key+'_mean'] = statistics.mean(values)
            stats[key+'_sd'] = statistics.stdev(values) if len(values) > 1 else None
        for key in ('client_rpc_straggler_ratio', 'coordinator_sync_wait_sum_sec', 'coordinator_sync_wait_max_sec', 'fleet_server_queue_wait_sec'):
            per_seed = [statistics.mean(row[key] for row in chosen_rounds if row['seed'] == seed) for seed in sorted({row['seed'] for row in selected})]
            stats[key+'_mean'] = statistics.mean(per_seed)
            stats[key+'_sd'] = statistics.stdev(per_seed) if len(per_seed) > 1 else None
        for target in P['quality_thresholds']:
            key = 'ttq_'+str(target)
            reached = [row[key+'_sec'] for row in selected if row[key+'_reached']]
            stats[key+'_reached_n'] = len(reached)
            stats[key+'_median_of_reached_sec'] = statistics.median(reached) if reached else None
        selected_communication = [row for row in communication if row['seed'] in P['seeds'] and row['policy'] == policy]
        for key in ('activation_upload_bytes', 'gradient_download_bytes',
                    'model_state_synchronization_bytes', 'total_communication_bytes'):
            values = [row[key] for row in selected_communication]
            stats[key+'_mean'] = statistics.mean(values)
            stats[key+'_sd'] = statistics.stdev(values) if len(values) > 1 else None
        selected_clients = [row for row in clients if row['primary'] and row['policy'] == policy]
        switches = [sum(row['cut_switched'] for row in selected_clients if row['seed'] == seed)
                    for seed in sorted({row['seed'] for row in selected})]
        stats['cut_switches_per_seed_mean'] = statistics.mean(switches)
        stats['cut_switches_per_seed_sd'] = statistics.stdev(switches) if len(switches) > 1 else None
        policy_summaries.append(stats)
    result['policy_summaries'] = policy_summaries
    result['paired_inference'] = []
    if result['primary_terminal'] == P['primary_planned_arms']:
        from experiments.common.statistics import paired_effect, holm_adjust
        plan = read(ROOT/'analysis_plan.json')
        for reference, comparator in plan['comparisons']:
            left = {row['seed']: row for row in summary if row['primary'] and row['policy'] == reference}
            right = {row['seed']: row for row in summary if row['primary'] and row['policy'] == comparator}
            common=sorted(set(left)&set(right))
            assert len(common)>=2
            inference = {'reference': reference, 'comparator': comparator, 'paired_seeds':common,
                         'failed_seeds':sorted(set(P['seeds'])-set(common)),
                         'selection_scope':'conditional on both jobs completing; failures remain separate',
                         'same_domain': left[common[0]]['candidate_count'] == right[common[0]]['candidate_count']}
            for label, key, scale in [('time', 'server_duration_sec', 1), ('accuracy_pp', 'final_accuracy', 100)]:
                inference[label] = paired_effect([left[seed][key]*scale for seed in common],
                    [right[seed][key]*scale for seed in common],
                    bootstrap_samples=plan['bootstrap_samples'], seed=plan['bootstrap_seed']).to_dict()
            result['paired_inference'].append(inference)
        for label in ('time', 'accuracy_pp'):
            adjusted = holm_adjust(row[label]['sign_flip_p_value'] for row in result['paired_inference'])
            for row, value in zip(result['paired_inference'], adjusted):
                row[label]['holm_adjusted_p_value'] = value
    pretraining = next((row for row in summary if not row['primary']), None)
    if pretraining is not None:
        result['ppo_pretraining'] = {'server_duration_sec':pretraining['server_duration_sec'],
            'controller_arm_wall_sec':pretraining['controller_arm_wall_sec'],
            'rounds':pretraining['rounds'], 'pretraining_receipt': read(ROOT/'ppo_pretraining_receipt.json')}
        deployment = next((row for row in policy_summaries if row['policy']=='FedAdapt-PPO'), None)
        if deployment is not None:
            result['ppo_pretraining']['cost_amortized_over_five_primary_jobs_sec'] = pretraining['server_duration_sec']/5 + deployment['server_duration_sec_mean']
    result['analysis_plan'] = read(ROOT/'analysis_plan.json')
    if (ROOT/'analysis_plan_after_failure.json').exists():
        result['analysis_plan_after_failure'] = read(ROOT/'analysis_plan_after_failure.json')
    write_csv(output/'policy_summary.csv', policy_summaries)
    write_csv(output/'failed_arms.csv', failures)
    report = ['# Actual-device adaptive placement comparison', '',
        'Source identity: '+state['source_identity']+'. Complete / failed / planned primary arms: '+str(result['primary_completed'])+' / '+str(result['primary_failed'])+' /30.', '',
        'Controller adaptations share the physical SFL execution path. Deployment time includes actual calibration, model synchronization, training, aggregation and evaluation. Offline PPO preparation is separate.', '',
        '| Policy | Complete / failed / planned | Job seconds, mean ± SD | Accuracy %, mean ± SD | Native / capped dispatch seconds | Synchronization wait sum per round, seconds | TTQ .50 / .60 reached |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'NA' if value is None else f'{value:.2f}'
    for row in policy_summaries:
        report.append('| '+row['policy']+' | '+str(row['paired_n'])+' / '+str(row['failed_n'])+' / '+str(row['planned_n'])+' | '+fmt(row['server_duration_sec_mean'])+' ± '+fmt(row['server_duration_sec_sd'])+' | '+fmt(100*row['final_accuracy_mean'])+' ± '+fmt(None if row['final_accuracy_sd'] is None else 100*row['final_accuracy_sd'])+' | '+fmt(row['native_mean_dispatch_sec_mean'])+' / '+fmt(row['capped_mean_dispatch_sec_mean'])+' | '+fmt(row['coordinator_sync_wait_sum_sec_mean'])+' | '+str(row['ttq_0.5_reached_n'])+' / '+str(row['ttq_0.6_reached_n'])+' |')
    report += ['', 'Failed arms are retained in failed_arms.csv and never retried/replaced. Summaries and pairwise estimates are conditional on completed jobs, so they do not estimate failure-inclusive overall advantage. All per-seed and per-round endpoints remain in the CSV files. No missing result is replaced by zero. Five seeds cannot attain a two-sided exact sign-flip p below .0625; conditional bootstrap intervals and Holm-adjusted values remain descriptive.', '',
        'The 72-cut CoSplit/Independent contrast holds the candidate domain constant. Full-versus-layer CoSplit uses the same production policy and domain-specific admitted calibration anchors. Published controls change the placement rule within a shared SFL pipeline, rather than reproducing complete published systems.', '',
        'Per-client server queue attribution is unknown because wire client IDs are absent. Coordinator synchronization waits use one clock. Unreached TTQ endpoints in completed jobs are censored; failed jobs are excluded from target counts and timing rather than classified as unreached. The actor is shared across five workload seeds; these are not five independently trained RL policies.']
    report += ['', '| Policy | Complete / planned | TTQ .50 reached / planned; median seconds among reached | TTQ .60 reached / planned; median seconds among reached |',
               '|---|---:|---:|---:|']
    for row in policy_summaries:
        report.append('| '+row['policy']+' | '+str(row['paired_n'])+' /5 | '+
            str(row['ttq_0.5_reached_n'])+' /5; '+fmt(row['ttq_0.5_median_of_reached_sec'])+' | '+
            str(row['ttq_0.6_reached_n'])+' /5; '+fmt(row['ttq_0.6_median_of_reached_sec'])+' |')
    report += ['', 'TTQ medians condition on completion and reaching the target, and do not estimate an unconditional equal-quality ranking. Failed and censored times are not imputed.']
    if pretraining is not None:
        report += ['', 'PPO preparation completed '+str(pretraining['rounds'])+' actual rounds (one reference + fifty learning transitions), taking '+fmt(pretraining['server_duration_sec'])+' server seconds. See ppo_pretraining_receipt.json for the final actor and source hashes.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'communication.json').write_text(json.dumps(communication, indent=2, allow_nan=False)+'\n')
    (output/'raw_receipts_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    for name, values in [('physical_summary.csv', summary), ('client_trajectories.csv', clients), ('round_trajectories.csv', rounds), ('paired_comparisons.csv', pairs)]:
        write_csv(output/name, values)
    print(json.dumps({'completed_arms':len(arms), 'primary_completed':result['primary_completed'], 'complete_seeds':result['complete_paired_seeds'], 'output':str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ROOT, OUTPUT = args.study_root.resolve(), args.output
    P = read(ROOT / 'protocol.json')
    main()
