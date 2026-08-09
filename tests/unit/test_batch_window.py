import pytest

from splitfleet.autosplit.batch_window import (
    BatchWindowError,
    batch_window_contains,
    describe_batch_window,
    normalize_batch_window,
    require_batch_in_window,
)


def test_normalize_accepts_json_lists_tuples_and_none() -> None:
    assert normalize_batch_window(None) is None
    assert normalize_batch_window("null") is None
    assert normalize_batch_window("") is None
    assert normalize_batch_window("[1, 64]") == (1, 64)
    assert normalize_batch_window([2, 8]) == (2, 8)
    assert normalize_batch_window((2, 8)) == (2, 8)


@pytest.mark.parametrize("value", ["[1]", [0, 8], [8, 2], "not-json", 4])
def test_normalize_rejects_malformed_windows(value) -> None:
    with pytest.raises(BatchWindowError):
        normalize_batch_window(value)


def test_unbounded_window_accepts_any_positive_batch() -> None:
    assert batch_window_contains(None, 1)
    assert batch_window_contains(None, 4096)
    assert describe_batch_window(None) == "unbounded"


def test_window_bounds_are_inclusive() -> None:
    assert batch_window_contains((2, 8), 2)
    assert batch_window_contains((2, 8), 8)
    assert not batch_window_contains((2, 8), 1)
    assert not batch_window_contains((2, 8), 9)
    assert not batch_window_contains((2, 8), 0)


def test_require_batch_names_the_window_and_the_fix() -> None:
    require_batch_in_window(4, (2, 8), stage="Split suffix")

    with pytest.raises(BatchWindowError) as short_batch:
        require_batch_in_window(1, (2, 8), stage="Split prefix", trace_batch_mode="batch_gt1")
    message = str(short_batch.value)
    assert "batch size 1" in message
    assert "[2, 8]" in message
    assert "batch_gt1" in message
    assert "drop_last=True" in message

    with pytest.raises(BatchWindowError) as long_batch:
        require_batch_in_window(64, (2, 8), stage="Split prefix")
    assert "lower the client batch size" in str(long_batch.value)
