"""Control RPC concurrency preserves logical identities and atomic admission."""

import json
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from splitfleet.virtual.server_adapter import VirtualStrategy


@pytest.mark.parametrize('duplicate', [False, True])
def test_identity_lookup_uses_worker_bound_and_only_commits_a_bijection(tmp_path, duplicate):
    barrier, lock = Barrier(2), Lock()
    active, peak, calls = 0, 0, 0

    def properties(pid, ins, **kwargs):
        nonlocal active, peak, calls
        assert ins.config == {'virtual_identity_only': True}
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return SimpleNamespace(properties={'virtual_partition_id': 0 if duplicate and pid == 3 else pid})

    clients = {str(i): SimpleNamespace(get_properties=lambda ins, _pid=i, **kw:
                                      properties(_pid, ins, **kw)) for i in range(4)}
    strategy = VirtualStrategy.__new__(VirtualStrategy)
    strategy.client_properties, strategy.identity_lookup_sec = {}, 0.0
    strategy.virtual_config = SimpleNamespace(virtual_clients=SimpleNamespace(total=4),
        resources=SimpleNamespace(max_concurrent_clients=2, effective_worker_capacity=2),
        profile=lambda _pid: SimpleNamespace(device='cpu'))
    strategy.evidence = SimpleNamespace(directory=tmp_path,
        write=lambda name, value: (tmp_path / name).write_text(json.dumps(value)))
    manager = SimpleNamespace(all=lambda: clients)
    if duplicate:
        with pytest.raises(RuntimeError, match='not bijective'):
            strategy._identify_clients(manager)
        assert strategy.client_properties == {} and not (tmp_path / 'client_manifest.json').exists()
    else:
        assert strategy._identify_clients(manager) is clients
        assert {p['virtual_partition_id'] for p in strategy.client_properties.values()} == set(range(4))
        assert strategy._identify_clients(manager) is clients
        manifest = json.loads((tmp_path / 'client_manifest.json').read_text())
        assert [row['virtual_partition_id'] for row in manifest] == list(range(4))
    assert peak == 2 and calls == 4 and active == 0
