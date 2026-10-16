import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from splitfleet.autosplit.state_ownership import validate_owned
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.common import BatchData, ControlCode, ServerModelFitIns
from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY
from splitfleet.server.stage_runtime.manager import StageRuntimeManager
from splitfleet.server.stage_runtime.storage import RoundStateStore
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.tasks import ImageClassificationTask
from splitfleet.virtual.suffix_service import (
    LeasedSuffixServicer,
    SuffixService,
    suffix_connection,
)


class BatchNormClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 8)
        self.bn = nn.BatchNorm1d(8)
        self.feature = nn.ReLU()
        self.tail = nn.Linear(8, 2)

    def forward(self, x):
        return self.tail(self.feature(self.bn(self.head(x))))


def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)


def test_private_suffix_slot_resets_all_state_and_optimizer_and_invalidates_cut(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(37)
    model = BatchNormClassifier()
    task = ImageClassificationTask()
    # Dense training BN rejects B1, so this fixture uses its native B2 capture.
    batches = [(torch.randn(2, 4), torch.tensor([0, 1])) for _ in range(6)]
    sample = task.prepare_batch(batches[0])
    strategy = AutoSplitStrategy(model=model, sample_inputs=sample, task=task,
                                 boundary="after:head", aggregation_policy="splitfed",
                                 owned_state_exchange=True, optimizer_fn=optimizer)
    store = RoundStateStore(tmp_path)
    manager = StageRuntimeManager(init_server_model_fn=strategy.init_server_model_fn,
        autosplit_session=strategy.autosplit_session, replica_capacity=1,
        state_store=store, reuse_completed_replicas=True)
    strategy.bind_stage_runtime_manager(manager)
    manager.set_strategy(strategy)
    sids = ["1", "2", "3", "4"]
    strategy._cid_to_sid_mapping = dict(zip(sids, sids, strict=True))
    strategy._round_active_clients = sids
    configs = {}
    for index, sid in enumerate(sids):
        placement = strategy.get_or_create_placement_plan("after:head" if index < 2 else "after:feature")
        config = strategy._autosplit_config(1, placement=placement)
        config.update({CLIENT_ID_CONFIG_KEY: sid, "virtual_partition_id": index})
        configs[sid] = config
    parameters = strategy.backend_adapter.export_ndarrays(model)
    manager.initialize_server_models([ServerModelFitIns(parameters=parameters, config=configs[sid], sid=sid)
                                     for sid in sids])
    servicer = LeasedSuffixServicer(manager, capacity=1, concurrency=1, state_store=store, seed=37)
    servicer.prepare_round(sids, 1)
    service = SuffixService(manager, servicer=servicer).start()
    request = BatchData(data={}, control_code=ControlCode.OK)
    identities, handles, expected_updates = [], [], {}
    try:
        for index, sid in enumerate(sids[:3]):
            current = batches[index * 2:index * 2 + 2]
            reference = copy.deepcopy(model)
            native_optimizer = optimizer(reference)
            for x, y in current:
                native_optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(reference(x), y).backward()
                native_optimizer.step()
            client = AutoSplitSplitLearningClient(model=copy.deepcopy(model), sample_inputs=sample,
                train_data=current, task=task, max_cached_runtimes=1, optimizer_fn=optimizer)
            with suffix_connection(service.address, sid, timeout=10,
                    metadata=(("authorization", f"Bearer {servicer.tokens[sid]}"),)) as proxy:
                reply = proxy.virtual_acquire(request, _streams_=False)
                receipt = json.loads(reply.data["metadata"])
                suffix = manager.get_server_model(sid)
                identities.append(id(suffix))
                handles.append(suffix.runtime_handle)
                assert receipt["suffix_replica_reused"] == bool(index)
                if index == 0:
                    with pytest.raises(RuntimeError, match="capacity exceeded"):
                        manager.acquire_replica("2")
                    assert manager.get_server_model(sid) is suffix
                assert not suffix.optimizer.state
                assert not suffix._processed_steps
                for value, initial in zip(suffix.get_parameters(), parameters, strict=True):
                    np.testing.assert_array_equal(value, initial)
                client.server_model_proxy = proxy
                prefix, count, _ = client.fit(parameters, configs[sid])
                assert count == 4
                proxy.virtual_commit(request, _streams_=False)
            result = next(row for row in manager.collect_server_fit_results() if row.sid == sid)
            ownership = json.loads(configs[sid]["autosplit_state_ownership"])
            merged = {**validate_owned(prefix, ownership, "prefix", parameters),
                      **validate_owned(result.parameters, ownership, "suffix", parameters)}
            assert len(merged) == len(parameters)
            for state_index, expected in enumerate(reference.state_dict().values()):
                np.testing.assert_allclose(merged[state_index], expected.detach().numpy(), atol=1e-5, rtol=1e-4)
            assert result.config["suffix_optimization_steps"] == 2
            expected_updates[sid] = [value.copy() for value in result.parameters]
            assert len(manager._delegate.idle) == 1 and not manager._delegate.live
            idle = manager._delegate.idle[0]
            assert idle.optimizer is None and not idle.sid and not idle._processed_steps
            assert len(idle._runtime_cache) == 1
        assert len(set(identities)) == 1
        assert handles[0] is handles[1] and handles[1] is not handles[2]
        assert manager._delegate.peak_live == 1
        assert manager._delegate.total_created == 1 and manager._delegate.total_reused == 2
        for result in manager.collect_server_fit_results():
            for value, expected in zip(result.parameters, expected_updates[result.sid], strict=True):
                np.testing.assert_array_equal(value, expected)
        with suffix_connection(service.address, "4", timeout=10,
                metadata=(("authorization", f"Bearer {servicer.tokens['4']}"),)) as proxy:
            proxy.virtual_acquire(request, _streams_=False)
            proxy.virtual_abort(request, _streams_=False)
        assert not manager._delegate.idle and not manager._clone_runtime_cache
        assert "4" not in manager._delegate.completed
        with pytest.raises(RuntimeError, match="already attempted"):
            manager.acquire_replica("4")
    finally:
        service.stop()
        manager.end_round()
        store.clear()
    assert not manager._delegate.idle and not manager._clone_runtime_cache
