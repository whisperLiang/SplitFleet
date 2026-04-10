"""Frontier-based split enumeration for concrete execution DAGs."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from .replay_plan import ExecutionPlan, FrontierSplit
from .replay_utils import canonicalize_batch_agnostic_shape, hash_graph_signature

UNBOUNDED_ALL_FRONTIER_MAX_CANDIDATES = 18


def enumerate_frontier_splits(
    plan: ExecutionPlan,
    *,
    mode: str = "minimal",
    max_frontier_size: Optional[int] = None,
    max_splits: Optional[int] = None,
    include_single_node: bool = True,
    require_device_compatible: bool = True,
) -> List[FrontierSplit]:
    """Enumerate valid frontier splits of the concrete execution DAG.

    Args:
        plan: Compiled execution plan.
        mode: ``"minimal"`` or ``"all"``.
        max_frontier_size: Optional frontier-size cap.
        max_splits: Optional cap on the number of splits returned.
        include_single_node: Whether a frontier of size 1 is allowed.
        require_device_compatible: Present for API symmetry; concrete plans are
            device-agnostic, so all enumerated splits are device-compatible.

    Returns:
        List of valid :class:`FrontierSplit` objects.
    """

    if mode not in {"minimal", "all"}:
        raise ValueError("mode must be either 'minimal' or 'all'.")

    parents = {node.idx: set(node.parents) for node in plan.nodes}
    children: Dict[int, Set[int]] = {node.idx: set() for node in plan.nodes}
    for node in plan.nodes:
        for parent_idx in node.parents:
            children[parent_idx].add(node.idx)

    input_set = set(plan.input_node_indices)
    output_set = set(plan.output_node_indices)
    reachable_from_inputs = _forward_reachable(input_set, children)
    reaches_outputs = _backward_reachable(output_set, parents)
    candidate_nodes = [
        idx
        for idx in range(len(plan.nodes))
        if idx in reachable_from_inputs
        and idx in reaches_outputs
        and idx not in input_set
        and idx not in output_set
    ]
    candidate_set = set(candidate_nodes) | output_set | input_set
    descendant_cache = _compute_descendant_cache(candidate_set, children)

    if mode == "minimal":
        minimal_frontier_specs = _enumerate_minimal_frontiers(
            plan,
            candidate_set,
            parents,
            children,
            input_set,
            output_set,
            descendant_cache,
            max_frontier_size=max_frontier_size,
            include_single_node=include_single_node,
        )
        splits = [
            _build_frontier_split(
                plan,
                boundary,
                candidate_set,
                parents,
                children,
                input_set,
                output_set,
                cutoff=cutoff,
                validation_mode=validation_mode,
            )
            for boundary, cutoff, validation_mode in minimal_frontier_specs
        ]
    else:
        all_frontier_specs = _enumerate_all_frontiers(
            plan,
            candidate_nodes,
            candidate_set,
            parents,
            children,
            input_set,
            output_set,
            descendant_cache,
            max_frontier_size=max_frontier_size,
            include_single_node=include_single_node,
            max_splits=max_splits,
        )
        splits = [
            _build_frontier_split(
                plan,
                boundary,
                candidate_set,
                parents,
                children,
                input_set,
                output_set,
                validation_mode=validation_mode,
            )
            for boundary, validation_mode in all_frontier_specs
        ]

    if max_splits is not None:
        splits = splits[:max_splits]

    return splits


def _enumerate_minimal_frontiers(
    plan: ExecutionPlan,
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    children: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
    descendant_cache: Dict[int, Set[int]],
    *,
    max_frontier_size: Optional[int],
    include_single_node: bool,
) -> List[Tuple[Tuple[int, ...], int, str]]:
    frontier_specs: Dict[Tuple[int, ...], Tuple[int, str]] = {}
    candidate_nodes = candidate_set - input_set - output_set

    for cutoff in range(len(plan.nodes) - 1):
        prefix = set(range(cutoff + 1)) - input_set - output_set
        boundary = tuple(
            sorted(
                idx for idx in _boundary_from_prefix(prefix, children) if idx in candidate_nodes
            )
        )
        if not boundary:
            continue
        if not include_single_node and len(boundary) == 1:
            continue
        if max_frontier_size is not None and len(boundary) > max_frontier_size:
            continue
        validation_mode = _frontier_validation_mode(
            boundary,
            candidate_set,
            parents,
            children,
            input_set,
            output_set,
            descendant_cache,
        )
        if validation_mode is not None:
            frontier_specs.setdefault(boundary, (cutoff, validation_mode))

    unique_frontiers = sorted(
        frontier_specs,
        key=lambda items: (
            0 if frontier_specs[items][1] == "classic" else 1,
            len(items),
            items,
        ),
    )
    minimal_frontiers: List[Tuple[Tuple[int, ...], int, str]] = []
    for frontier in unique_frontiers:
        cutoff, validation_mode = frontier_specs[frontier]
        if any(
            existing_mode == validation_mode and set(other).issubset(frontier)
            for other, _, existing_mode in minimal_frontiers
        ):
            continue
        minimal_frontiers.append((frontier, cutoff, validation_mode))
    return minimal_frontiers


def _enumerate_all_frontiers(
    plan: ExecutionPlan,
    candidate_nodes: Sequence[int],
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    children: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
    descendant_cache: Dict[int, Set[int]],
    *,
    max_frontier_size: Optional[int],
    include_single_node: bool,
    max_splits: Optional[int],
) -> List[Tuple[Tuple[int, ...], str]]:
    frontier_size_cap = _resolve_all_frontier_size_cap(
        candidate_nodes,
        max_frontier_size=max_frontier_size,
        max_splits=max_splits,
    )
    frontier_specs: List[Tuple[Tuple[int, ...], str]] = []
    start_size = 1 if include_single_node else 2
    ancestor_cache = _compute_ancestor_cache(set(candidate_nodes), parents)
    incompatibility_map = {
        idx: descendant_cache.get(idx, set()) | ancestor_cache.get(idx, set())
        for idx in candidate_nodes
    }

    def dfs(
        start_index: int,
        current: List[int],
        blocked: Set[int],
    ) -> bool:
        if len(current) >= start_size:
            validation_mode = _frontier_validation_mode(
                current,
                candidate_set,
                parents,
                children,
                input_set,
                output_set,
                descendant_cache,
            )
            if validation_mode is not None:
                frontier_specs.append((tuple(current), validation_mode))
                if max_splits is not None and len(frontier_specs) >= max_splits:
                    return True

        if len(current) >= frontier_size_cap:
            return False

        for position in range(start_index, len(candidate_nodes)):
            candidate = candidate_nodes[position]
            if candidate in blocked:
                continue
            current.append(candidate)
            next_blocked = blocked | incompatibility_map[candidate] | {candidate}
            should_stop = dfs(position + 1, current, next_blocked)
            current.pop()
            if should_stop:
                return True
        return False

    dfs(0, [], set())
    return sorted(
        frontier_specs,
        key=lambda item: (0 if item[1] == "classic" else 1, len(item[0]), item[0]),
    )


def _resolve_all_frontier_size_cap(
    candidate_nodes: Sequence[int],
    *,
    max_frontier_size: Optional[int],
    max_splits: Optional[int],
) -> int:
    if max_frontier_size is not None:
        return max_frontier_size

    if len(candidate_nodes) <= UNBOUNDED_ALL_FRONTIER_MAX_CANDIDATES:
        return len(candidate_nodes)

    if max_splits is not None:
        return len(candidate_nodes)

    raise ValueError(
        "enumerate_frontier_splits(mode='all') would explore an unsafe number of "
        f"frontiers for {len(candidate_nodes)} candidate nodes. "
        "Set max_frontier_size and/or max_splits to bound the search."
    )


def _boundary_from_prefix(prefix: Set[int], children: Dict[int, Set[int]]) -> Set[int]:
    return {idx for idx in prefix if any(child not in prefix for child in children[idx])}


def _is_valid_frontier(
    boundary: Sequence[int],
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    children: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
    descendant_cache: Dict[int, Set[int]],
) -> bool:
    return _frontier_validation_mode(
        boundary,
        candidate_set,
        parents,
        children,
        input_set,
        output_set,
        descendant_cache,
    ) is not None


def _frontier_validation_mode(
    boundary: Sequence[int],
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    children: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
    descendant_cache: Dict[int, Set[int]],
) -> Optional[str]:
    boundary_set = set(boundary)
    if not boundary_set:
        return None
    if boundary_set & input_set:
        return None
    if boundary_set & output_set:
        return None

    if _contains_path_among_boundary(boundary, descendant_cache):
        return None

    if _is_valid_frontier_classic(
        boundary_set,
        candidate_set,
        parents,
        children,
        input_set,
        output_set,
    ):
        return "classic"

    if _is_valid_frontier_with_passthrough_inputs(
        boundary_set,
        candidate_set,
        parents,
        children,
        input_set,
        output_set,
    ):
        return "passthrough"

    return None


def _is_valid_frontier_classic(
    boundary_set: Set[int],
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    children: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
) -> bool:
    reachable_without_boundary = _forward_reachable(
        {idx for idx in input_set if idx not in boundary_set},
        children,
        blocked=boundary_set,
        candidate_set=candidate_set,
    )
    if reachable_without_boundary & output_set:
        return False

    prefix_core = reachable_without_boundary & candidate_set
    suffix_nodes = candidate_set - prefix_core - boundary_set
    suffix_nodes -= {
        idx
        for idx in suffix_nodes
        if idx not in output_set and not children[idx] and not parents[idx]
    }
    if not suffix_nodes:
        return False

    for node_idx in suffix_nodes:
        hidden_prefix_parents = (parents[node_idx] & prefix_core) - boundary_set
        if hidden_prefix_parents:
            return False

    for boundary_idx in boundary_set:
        if not any(child in suffix_nodes for child in children[boundary_idx]):
            return False

    return True


def _is_valid_frontier_with_passthrough_inputs(
    boundary_set: Set[int],
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    children: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
) -> bool:

    suffix_nodes = _compute_candidate_suffix_closure(
        boundary_set,
        candidate_set,
        parents,
        input_set,
        output_set,
    )
    if not output_set.issubset(suffix_nodes):
        return False

    if not suffix_nodes:
        return False

    passthrough_inputs = {
        parent_idx
        for node_idx in suffix_nodes
        for parent_idx in parents[node_idx]
        if parent_idx in input_set
    }
    boundary_ancestors = _backward_reachable(boundary_set, parents)
    if passthrough_inputs & boundary_ancestors:
        return False

    for node_idx in suffix_nodes:
        hidden_prefix_parents = (
            (parents[node_idx] & candidate_set) - input_set - boundary_set - suffix_nodes
        )
        if hidden_prefix_parents:
            return False

    for boundary_idx in boundary_set:
        if not any(child in suffix_nodes for child in children[boundary_idx]):
            return False
        if (parents[boundary_idx] - input_set) & suffix_nodes:
            return False

    return True


def _contains_path_among_boundary(
    boundary: Sequence[int],
    descendant_cache: Dict[int, Set[int]],
) -> bool:
    boundary_set = set(boundary)
    for idx in boundary:
        if descendant_cache[idx] & boundary_set:
            return True
    return False


def _build_frontier_split(
    plan: ExecutionPlan,
    boundary: Sequence[int],
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    children: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
    cutoff: Optional[int] = None,
    validation_mode: str = "classic",
) -> FrontierSplit:
    boundary_set = set(boundary)
    if cutoff is not None:
        prefix_nodes = list(range(cutoff + 1))
    else:
        prefix_nodes = _expand_partition_with_parents(boundary_set, parents)
    candidate_prefix_nodes = (set(prefix_nodes) & candidate_set) - boundary_set - input_set
    suffix_nodes = _compute_suffix_execution_nodes(
        boundary_set,
        parents,
        input_set=input_set,
        output_set=output_set,
        blocked=candidate_prefix_nodes,
    )
    passthrough_input_indices = sorted(
        {
            parent_idx
            for node_idx in suffix_nodes
            for parent_idx in parents[node_idx]
            if parent_idx in input_set
        }
    )

    boundary_labels = [plan.nodes[idx].label for idx in boundary]
    boundary_schema = {
        label: {
            "idx": idx,
            "shape": canonicalize_batch_agnostic_shape(plan.nodes[idx].meta.get("tensor_shape")),
            "dtype": plan.nodes[idx].meta.get("tensor_dtype"),
        }
        for label, idx in zip(boundary_labels, boundary)
    }
    split_id = hash_graph_signature(("frontier", tuple(sorted(boundary))))

    return FrontierSplit(
        split_id=split_id,
        boundary_labels=boundary_labels,
        boundary_indices=list(boundary),
        prefix_node_indices=prefix_nodes,
        suffix_node_indices=suffix_nodes,
        boundary_schema=boundary_schema,
        meta={
            "mode": "frontier",
            "validation_mode": validation_mode,
            "execution_cutoff": cutoff,
            "passthrough_input_indices": passthrough_input_indices,
            "passthrough_input_labels": [
                plan.nodes[idx].label for idx in passthrough_input_indices
            ],
        },
    )


def _expand_partition_with_parents(
    seed_nodes: Set[int],
    parents: Dict[int, Set[int]],
    *,
    blocked: Optional[Set[int]] = None,
) -> List[int]:
    expanded = set(seed_nodes)
    frontier = list(seed_nodes)
    blocked = blocked or set()

    while frontier:
        current = frontier.pop()
        for parent in parents[current]:
            if parent in blocked or parent in expanded:
                continue
            expanded.add(parent)
            frontier.append(parent)

    return sorted(expanded)


def _compute_suffix_closure(
    boundary_set: Set[int],
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
) -> Set[int]:
    return _compute_candidate_suffix_closure(
        boundary_set,
        candidate_set,
        parents,
        input_set,
        output_set,
    )


def _compute_candidate_suffix_closure(
    boundary_set: Set[int],
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
    input_set: Set[int],
    output_set: Set[int],
) -> Set[int]:
    suffix_nodes = set()
    frontier = list(output_set)

    while frontier:
        current = frontier.pop()
        if current in boundary_set or current in input_set or current in suffix_nodes:
            continue
        if current in candidate_set:
            suffix_nodes.add(current)
        for parent_idx in parents[current]:
            if parent_idx not in boundary_set and parent_idx not in input_set:
                frontier.append(parent_idx)

    return suffix_nodes


def _compute_suffix_execution_nodes(
    boundary_set: Set[int],
    parents: Dict[int, Set[int]],
    *,
    input_set: Set[int],
    output_set: Set[int],
    blocked: Set[int],
) -> List[int]:
    suffix_nodes = set()
    frontier = list(output_set)

    while frontier:
        current = frontier.pop()
        if current in boundary_set or current in input_set or current in suffix_nodes:
            continue
        if current in blocked:
            raise ValueError(
                "Invalid frontier split: suffix execution depends on prefix-only nodes."
            )
        suffix_nodes.add(current)
        for parent_idx in parents[current]:
            if parent_idx not in boundary_set and parent_idx not in input_set:
                frontier.append(parent_idx)

    return sorted(suffix_nodes)


def _forward_reachable(
    starts: Set[int],
    children: Dict[int, Set[int]],
    *,
    blocked: Optional[Set[int]] = None,
    candidate_set: Optional[Set[int]] = None,
) -> Set[int]:
    blocked = blocked or set()
    visited = set()
    frontier = [idx for idx in starts if idx not in blocked]
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        if candidate_set is not None and current not in candidate_set:
            continue
        visited.add(current)
        for child in children[current]:
            if child not in blocked and child not in visited:
                frontier.append(child)
    return visited


def _backward_reachable(starts: Set[int], parents: Dict[int, Set[int]]) -> Set[int]:
    visited = set()
    frontier = list(starts)
    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)
        for parent in parents[current]:
            if parent not in visited:
                frontier.append(parent)
    return visited


def _compute_descendant_cache(
    candidate_set: Set[int],
    children: Dict[int, Set[int]],
) -> Dict[int, Set[int]]:
    cache: Dict[int, Set[int]] = {}
    for idx in candidate_set:
        descendants = _forward_reachable({idx}, children, candidate_set=candidate_set)
        descendants.discard(idx)
        cache[idx] = descendants
    return cache


def _compute_ancestor_cache(
    candidate_set: Set[int],
    parents: Dict[int, Set[int]],
) -> Dict[int, Set[int]]:
    cache: Dict[int, Set[int]] = {}
    for idx in candidate_set:
        ancestors = _backward_reachable({idx}, parents) & candidate_set
        ancestors.discard(idx)
        cache[idx] = ancestors
    return cache
