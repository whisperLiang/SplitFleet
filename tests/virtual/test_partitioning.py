from dataclasses import replace

import pytest
import torch

from experiments.virtual_sfl.workloads import fixture_dataset
from splitfleet.virtual.config import PartitionConfig
from splitfleet.virtual.partitions import ClientBatches, partition_dataset


@pytest.mark.parametrize("method", ["iid", "dirichlet"])
def test_partitions_reproduce_without_overlap_or_loss(method):
    config = PartitionConfig(method=method, min_samples=2)
    labels = [i % 3 for i in range(64)]
    first = partition_dataset(64, 16, config=config, seed=8, labels=labels)
    assert first == partition_dataset(64, 16, config=config, seed=8, labels=labels)
    assert sorted(i for values in first.values() for i in values) == list(range(64))
    assert min(map(len, first.values())) >= 2


def test_unbalanced_iid_and_insufficient_data_are_explicit():
    config = PartitionConfig(sample_counts=[1, 3, 8])
    parts = partition_dataset(12, 3, config=config, seed=3)
    assert list(map(len, parts.values())) == [1, 3, 8]
    with pytest.raises(ValueError, match="too small"):
        partition_dataset(3, 4, config=PartitionConfig(), seed=3)
    with pytest.raises(ValueError, match="cover"):
        partition_dataset(
            12, 3, config=replace(config, sample_counts=[1, 3, 7]), seed=3
        )
    with pytest.raises(ValueError, match="unsupported"):
        partition_dataset(
            12,
            3,
            config=PartitionConfig(method="dirichlet"),
            seed=3,
            labels=[0] * 12,
            family="object_detection",
        )


def test_sampler_accesses_only_assigned_indices_and_ignores_worker_order():
    dataset = fixture_dataset(20, 4)
    first = ClientBatches(
        dataset, [2, 7, 9, 14], batch_size=2, seed=4, partition_id=1, round_id=2
    )
    second = ClientBatches(
        dataset, [1, 3, 8], batch_size=2, seed=4, partition_id=0, round_id=2
    )
    values = torch.cat([raw[0] for raw in first])
    list(second)
    assert torch.equal(values, torch.cat([raw[0] for raw in first]))
    assert {tuple(row.tolist()) for row in values} == {
        tuple(dataset[i][0].tolist()) for i in [2, 7, 9, 14]
    }
