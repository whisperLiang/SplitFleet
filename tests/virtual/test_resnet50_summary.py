"""Statistical reporting guards; these fixtures are not measured training."""

import json
import csv

import numpy as np
import pytest
from scipy.stats import t

from experiments.resnet50_quality.efficiency import (
    TRAINING_WALL_SCOPE,
    measured_training_time,
    paired_efficiency_statistics,
    validate_timing_pair,
)
from experiments.resnet50_quality.protocol import PROTOCOL_VERSION, SEEDS, hash_file, learning_rate
from experiments.resnet50_quality.summarize import completed_run, paired_statistics, summarize_queue


@pytest.fixture
def complete_budget_fixture(tmp_path):
    """Small reporting receipts only; these are never native training evidence."""
    rounds, clients = 100, 50
    protocol = tmp_path / 'protocol'
    protocol.mkdir()
    frozen = {'protocol_version': PROTOCOL_VERSION, 'backend': 'pytorch', 'clients': clients,
              'backends': {'pytorch': clients}, 'worker_count_policy': 'fixed_hardware', 'logical_suffix_models': clients,
              'rounds': rounds, 'batch_size': 16, 'seeds': list(SEEDS), 'local_epochs': 1, 'all_clients_participate': True,
              'convergence': {'target_val_top1': .7, 'sustained_rounds': 5},
              'main_examples_per_run': rounds * 45_000,
              'optimizer': {'learning_rates': [float(learning_rate(r)) for r in range(1, rounds + 1)]}}
    (protocol / 'protocol.json').write_text(json.dumps(frozen))
    run = tmp_path / 'run'
    run.mkdir()
    steps_per_client = (900 + 15) // 16
    documents = {
        'config.json': {'rounds': rounds, 'workload': {'protocol': str(protocol)},
                        'resources': {'max_concurrent_clients': clients, 'worker_count_policy': 'one_per_client'}},
        'run_manifest.json': {'status': 'passed', 'stage': 'main', 'backend': 'pytorch',
                              'method': 'fedavg', 'seed': 233, 'clients': clients,
                              'protocol_sha256': hash_file(protocol / 'protocol.json')},
        'quality_validation.json': {'status': 'passed', 'examples_seen': rounds * 45_000,
                                    'training_tasks': clients * rounds, 'worker_count': clients,
                                    'optimization_steps': steps_per_client * clients * rounds},
        'test_metrics.json': {checkpoint: {'samples': 10_000, 'loss': 2., 'top1': .2}
                              for checkpoint in ('final', 'best_validation')},
        'result.json': {'status': 'passed', 'wall_sec': 120., 'wall_scope': TRAINING_WALL_SCOPE},
    }
    for name, document in documents.items():
        (run / name).write_text(json.dumps(document))
    with (run / 'round_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['round', 'examples_seen', 'successful_clients', 'optimization_steps',
                                                           'train_online_ce', 'val_top1', 'val_ce', 'elapsed_wall_sec'])
        writer.writeheader()
        for round_id in range(1, rounds + 1):
            writer.writerow({'round': round_id, 'examples_seen': round_id * 45_000,
                             'successful_clients': clients, 'optimization_steps': clients * steps_per_client,
                             'train_online_ce': 2. - round_id / 100, 'val_top1': .2 + round_id / 150,
                             'val_ce': 2. - round_id / 100, 'elapsed_wall_sec': float(round_id)})
    with (run / 'client_metrics.jsonl').open('w') as stream:
        for round_id in range(1, rounds + 1):
            for client in range(clients):
                row = {'status': 'success', 'partition_id': client, 'round_id': round_id,
                       'num_examples': 900, 'metrics': {'num_batches': steps_per_client, 'data_batch_hash': 'a' * 64}}
                stream.write(json.dumps(row) + '\n')
    return run, rounds


def test_reporting_accepts_complete_fifty_client_hundred_round_budget(complete_budget_fixture):
    path, rounds = complete_budget_fixture
    result = completed_run(path, backend='pytorch', method='fedavg', seed=233)
    assert len(result['rounds']) == rounds and len(result['identities']) == rounds * 50


def test_partial_run_does_not_become_complete_by_changing_config(complete_budget_fixture):
    path, rounds = complete_budget_fixture
    config = json.loads((path / 'config.json').read_text())
    config['rounds'] = rounds - 1
    (path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='frozen main budget'):
        completed_run(path, backend='pytorch', method='fedavg', seed=233)


def test_paired_interval_uses_differences_not_independent_method_variance():
    fedavg = np.array([.70, .79])
    splitfleet = fedavg + np.array([.001, .002])
    result = paired_statistics(fedavg, splitfleet)
    delta = (splitfleet - fedavg) * 100
    half_width = t.ppf(.975, 1) * delta.std(ddof=1) / np.sqrt(2)
    np.testing.assert_allclose(result["paired_95pct_t_interval_pp"], [delta.mean() - half_width, delta.mean() + half_width])
    assert result["degrees_of_freedom"] == 1 and result["interval_within_margin"]


def test_zero_mean_difference_with_wide_interval_is_not_equivalence():
    result = paired_statistics([.7] * 2, [.67, .73])
    assert abs(result["mean_delta_pp"]) < 1e-12
    assert not result["interval_within_margin"]


def test_incomplete_repeats_cannot_produce_primary_statistics():
    with pytest.raises(ValueError, match="all two"):
        paired_statistics([.7], [.71])


def test_pending_queue_reports_missing_results_without_accuracy_claims(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"jobs": [{"backend": "pytorch", "status": "pending"}]}))
    result = summarize_queue(path, tmp_path / "summary")
    assert result["status"] == "incomplete" and not result["main_results_available"]
    assert result["comparisons"] == {} and [row['backend'] for row in result['backends']] == ['pytorch']
    assert all("mean_delta_pp" not in row for row in result["backends"])
    assert all("geometric_mean_speedup" not in row for row in result["backends"])


def test_complete_cohort_reports_convergence_and_measured_wall_time(tmp_path, monkeypatch):
    from experiments.resnet50_quality import summarize

    selected = ('pytorch',)
    path = tmp_path / 'queue.json'
    jobs = [{'backend': backend, 'seed': seed, 'method': method, 'status': 'passed', 'output': 'fixture'}
            for backend in selected for seed in SEEDS for method in ('fedavg', 'splitfleet')]
    path.write_text(json.dumps({'jobs': jobs}))

    def fixture_run(_path, *, backend, method, seed):
        config = {key: {} for key in ('resources', 'profiles', 'client_profiles', 'server_device',
                  'networks', 'network_by_round', 'batch_size', 'local_epochs', 'optimizer',
                  'learning_rate', 'virtual_clients', 'transport')}
        config['rounds'] = 100
        from experiments.resnet50_quality.convergence import convergence_metrics
        duration = 200. if method == 'fedavg' else 100.
        curve = [{'round': r, 'train_online_ce': 2. - r / 100, 'val_top1': .2 + r / 150,
                  'val_ce': 2. - r / 100, 'elapsed_wall_sec': duration * r / 101} for r in range(1, 101)]
        convergence = convergence_metrics(curve, {'target_val_top1': .7, 'sustained_rounds': 5}, duration)
        return {'config': config, 'manifest': {'initial_hash': str(seed), 'protocol_sha256': 'fixture',
                'source_identity': 'fixture', 'device': 'cuda:1', 'CUDA_VISIBLE_DEVICES': '0,1',
                'native_environment': {'framework_backend': backend}}, 'identities': {'fixture': seed},
                'test': {'final': {'top1': .2}}, 'timing': {'wall_seconds': duration}, 'convergence': convergence, 'rounds': curve}

    monkeypatch.setattr(summarize, 'completed_run', fixture_run)
    result = summarize_queue(path, tmp_path / 'summary')
    assert result['status'] == 'passed' and result['completed_backend_comparisons'] == 1
    assert set(result['comparisons']) == set(selected)
    assert [row['backend'] for row in result['backends']] == list(selected)
    assert result['comparisons']['pytorch']['both_converged_and_faster']
    assert (tmp_path / 'summary/convergence_comparison.png').is_file()
    assert (tmp_path / 'summary/convergence_comparison.svg').is_file()
    assert (tmp_path / 'summary/time_to_target.csv').is_file()
    with (tmp_path / 'summary/backend_summary.csv').open() as stream:
        assert {row['backend'] for row in csv.DictReader(stream)} == set(selected)


def test_timing_interval_pairs_seeds_even_when_run_durations_vary_greatly():
    baseline = np.array([10., 100000.])
    result = paired_efficiency_statistics(baseline, baseline / 2)
    np.testing.assert_allclose(result["paired_95pct_t_interval_speedup"], [2., 2.])
    assert result["speedup_interval_above_one"]
    assert result["geometric_time_reduction_pct"] == pytest.approx(50.)


def test_faster_point_estimate_with_wide_interval_cannot_claim_speed_advantage():
    result = paired_efficiency_statistics([100.] * 2, [50., 150.])
    assert result["geometric_mean_speedup"] > 1
    assert not result["speedup_interval_above_one"]


def test_slower_runtime_is_reported_with_negative_time_reduction():
    result = paired_efficiency_statistics([100.] * 2, [200.] * 2)
    assert result["geometric_mean_speedup"] == pytest.approx(.5)
    assert result["geometric_time_reduction_pct"] == pytest.approx(-100.)
    assert not result["speedup_interval_above_one"]


@pytest.mark.parametrize("durations", [[100.], [100., 0.], [100., np.nan]])
def test_missing_or_invalid_time_observations_cannot_enter_statistics(durations):
    with pytest.raises(ValueError):
        paired_efficiency_statistics([100.] * 2, durations)


@pytest.mark.parametrize("receipt", [{"status": "passed"}, {"status": "failed", "wall_sec": 10},
                                     {"status": "passed", "wall_sec": True}])
def test_wall_time_is_not_inferred_from_partial_or_missing_measurements(tmp_path, receipt):
    (tmp_path / "result.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="measured wall_sec"):
        measured_training_time(tmp_path)


def test_different_worker_resource_budgets_cannot_form_timing_pair():
    manifest = {"source_identity": "fixture", "device": "cpu", "CUDA_VISIBLE_DEVICES": "",
                "native_environment": {"framework_backend": "torch"}}
    config = {key: {} for key in ("resources", "profiles", "client_profiles", "server_device",
              "networks", "network_by_round", "batch_size", "local_epochs", "rounds", "optimizer",
              "learning_rate", "virtual_clients", "transport")}
    left = {"manifest": manifest, "config": {**config, "resources": {"max_concurrent_clients": 2}}}
    right = {"manifest": manifest, "config": {**config, "resources": {"max_concurrent_clients": 4}}}
    with pytest.raises(ValueError, match="differs in resources"):
        validate_timing_pair(left, right)
