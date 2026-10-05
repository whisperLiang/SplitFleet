"""Metrics with explicit missing values and frozen adaptation definitions."""

import math


def oracle_gap(observed_ms, oracle_ms):
    if observed_ms is None or oracle_ms is None:
        return None
    if not all(math.isfinite(value) for value in (observed_ms, oracle_ms)) or observed_ms < 0 or oracle_ms <= 0:
        raise ValueError("Observed cost must be nonnegative; oracle cost must be positive and finite")
    return (observed_ms - oracle_ms) / oracle_ms


def adaptation_rounds(rows, *, change_round, tolerance, sustained_rounds):
    """First consecutive measured rounds within a predefined oracle gap.

    Returns None if not reached. Missing rounds/costs break the sequence. This
    metric measures recovery relative to an oracle, not convergence of training.
    """
    if change_round < 1 or tolerance < 0 or not math.isfinite(tolerance) or sustained_rounds < 1:
        raise ValueError("Declare a valid change round, tolerance and positive sustained window")
    relevant = sorted((row for row in rows if row["round_id"] >= change_round), key=lambda row: row["round_id"])
    if len({row["round_id"] for row in relevant}) != len(relevant):
        raise ValueError("Adaptation rows must have unique round IDs")
    first, count, previous = None, 0, None
    for row in relevant:
        gap = oracle_gap(row.get("observed_ms"), row.get("oracle_ms"))
        acceptable = gap is not None and gap <= tolerance
        if not acceptable or (previous is not None and row["round_id"] != previous + 1):
            first, count = None, 0
        if acceptable:
            first = row["round_id"] if first is None else first
            count += 1
            if count == sustained_rounds:
                return first - change_round
        previous = row["round_id"]
    return None
