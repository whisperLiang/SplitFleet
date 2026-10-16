import numpy as np

from splitfleet.autosplit.state_ownership import aggregate_owned
from splitfleet.server.stage_runtime.storage import RoundStateStore


def test_spilled_owned_aggregation_matches_weighted_state_and_integer_buffers(tmp_path):
    store = RoundStateStore(tmp_path)
    initial = [np.array([1.0, 2.0], dtype=np.float32), np.array(1, dtype=np.int64)]
    manifests = [{"owners": ["prefix", "suffix"], "names": ["weight", "counter"]}] * 2
    updates = [
        (
            manifests[0],
            store.put([np.array([2.0, 3.0], dtype=np.float32)]),
            store.put([np.array(3, dtype=np.int64)]),
            1,
        ),
        (
            manifests[1],
            store.put([np.array([5.0, 6.0], dtype=np.float32)]),
            store.put([np.array(7, dtype=np.int64)]),
            3,
        ),
    ]
    result = aggregate_owned(initial, iter(updates))
    np.testing.assert_array_equal(result[0], [4.25, 5.25])
    assert result[1] == 6
    assert result[1].dtype == np.int64
    store.clear()
    assert not list(tmp_path.iterdir())
