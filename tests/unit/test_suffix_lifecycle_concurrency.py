"""Preparing or exporting one suffix cannot block unrelated live models."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from splitfleet.common import ServerModelFitIns, ServerModelFitRes
from splitfleet.server.stage_runtime.leased_manager import LeasedServerModelManager
from splitfleet.server.stage_runtime.storage import RoundStateStore


@pytest.mark.parametrize('operation', ['prepare', 'finish'])
def test_lifecycle_does_not_hold_registry_lock_or_release_capacity_early(tmp_path, operation):
    entered, release = Event(), Event()

    class Model:
        def to_server_model(self):
            return self

        def configure_fit(self, config):
            self.sid = config.sid
            if self.sid == 'slow' and operation == 'prepare':
                entered.set()
                assert release.wait(5)

        def get_fit_result(self):
            entered.set()
            assert release.wait(5)
            return ServerModelFitRes(parameters=[np.ones(2, dtype=np.float32)], config={})

    manager = LeasedServerModelManager(Model, capacity=2, state_store=RoundStateStore(tmp_path),
                                      release_model=lambda model: None)
    manager.initialize_server_models([ServerModelFitIns(parameters=[], config={}, sid=sid)
                                     for sid in ['live', 'slow', 'next']])
    manager.acquire('live')
    if operation == 'finish':
        manager.acquire('slow')
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(manager.acquire, 'slow') if operation == 'prepare' else pool.submit(
            manager.finish, 'slow', success=True)
        assert entered.wait(5)
        try:
            assert pool.submit(manager.get_server_model, 'live').result(timeout=1).sid == 'live'
            with pytest.raises(RuntimeError, match='capacity exceeded'):
                manager.acquire('next')
            with pytest.raises(RuntimeError, match='lifecycle'):
                manager.end_round()
        finally:
            release.set()
        pending.result(timeout=5)
    assert not manager.preparing and not manager.finishing
    manager.end_round()
