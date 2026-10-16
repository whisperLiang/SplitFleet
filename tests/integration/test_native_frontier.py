import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from splitfleet.runtime.native_frontier import minimal_frontier_node_ids


def graph(rows, inputs=("input",), outputs=()):
    nodes = [SimpleNamespace(canonical_id=name, parents=parents, is_output=name in outputs,
                is_input=name in inputs, is_buffer=False, is_buffer_only_source=False, target=None)
             for name, parents in rows]
    by_id = {node.canonical_id: node for node in nodes}
    return SimpleNamespace(nodes=nodes, input_node_ids=inputs, output_node_ids=outputs,
                           order_by_id={name: index for index, (name, _) in enumerate(rows)},
                           node_for_label=by_id.get)


def test_prefix_statistics_outputs_keep_values_without_their_unused_large_parents():
    g = graph([("input",()),("activation",("input",)),("mean",("activation",)),
               ("feature",("activation",)),("logits",("feature",))], outputs=("mean","logits"))
    assert minimal_frontier_node_ids(g, prefix_node_ids={"input","activation","mean","feature"},
                                    suffix_node_ids={"logits"}) == ("mean","feature")


def test_genuine_suffix_use_of_same_parent_is_retained():
    g = graph([("input",()),("activation",("input",)),("mean",("activation",)),
               ("logits",("activation","mean"))], outputs=("mean","logits"))
    assert minimal_frontier_node_ids(g,prefix_node_ids={"input","activation","mean"},
                                    suffix_node_ids={"logits"}) == ("activation","mean")


def test_suffix_input_and_terminal_fallback_remain_available():
    g = graph([("input",()),("feature",("input",))])
    assert minimal_frontier_node_ids(g,prefix_node_ids=set(),suffix_node_ids={"input","feature"}) == ("input",)
    assert minimal_frontier_node_ids(g,prefix_node_ids={"input","feature"},suffix_node_ids=set()) == ("feature",)


@pytest.mark.parametrize("prefix,suffix", [({"input"},{"input","feature"}),({"input"},set()),({"invented"},{"input","feature"})])
def test_ambiguous_or_incomplete_partitions_are_rejected(prefix,suffix):
    g=graph([("input",()),("feature",("input",))])
    with pytest.raises(Exception,match="partition"):
        minimal_frontier_node_ids(g,prefix_node_ids=prefix,suffix_node_ids=suffix)


def test_unresolved_or_hidden_native_operands_are_rejected():
    g=graph([("input",()),("feature",("missing",))])
    with pytest.raises(Exception,match="Unresolved"):
        minimal_frontier_node_ids(g,prefix_node_ids={"input"},suffix_node_ids={"feature"})
    g=graph([("input",()),("feature",())])
    g.nodes[-1].target=SimpleNamespace(inputs=[SimpleNamespace(producer_label_raw="input",source_label_raw=None)])
    with pytest.raises(Exception,match="absent from graph"):
        minimal_frontier_node_ids(g,prefix_node_ids={"input"},suffix_node_ids={"feature"})


def test_private_native_function_changes_only_redundant_prefix_output_parents():
    from torchlens.split import planner

    from splitfleet.runtime.native_frontier import _private_native_frontier
    native = planner._frontier_node_ids
    private = _private_native_frontier(native)
    g = graph([("input",()),("activation",("input",)),("mean",("activation",)),
               ("feature",("activation",)),("logits",("feature",))], outputs=("mean","logits"))
    partition = {"prefix_node_ids": {"input","activation","mean","feature"}, "suffix_node_ids": {"logits"}}
    assert native(g, **partition) == ("activation","mean","feature")
    assert private(g, **partition) == minimal_frontier_node_ids(g, **partition) == ("mean","feature")
    assert planner._frontier_node_ids is native
    assert private.__globals__ is not native.__globals__


def test_native_bindings_leave_installed_sdk_functions_unchanged():
    from torchlens.split import api, pipeline, planner

    from splitfleet.runtime.native_frontier import _native_bindings
    before = (api.prepare,pipeline.normalize_to_split_ir,pipeline._probe_batch_replay,
              pipeline._execute_batch_probe,planner.plan_split,planner._frontier_node_ids)
    private_plan, private_prepare = _native_bindings()
    assert before == (api.prepare,pipeline.normalize_to_split_ir,pipeline._probe_batch_replay,
                      pipeline._execute_batch_probe,planner.plan_split,planner._frontier_node_ids)
    assert private_plan.__code__ is planner.plan_split.__code__
    assert private_prepare.__code__ is api.prepare.__code__


def test_wrong_native_version_fails_closed(monkeypatch):
    import splitfleet.runtime.native_frontier as frontier
    monkeypatch.setattr(frontier,"version",lambda _: "2.34.2")
    with pytest.raises(RuntimeError,match="requires torchlens"):
        frontier._native_bindings()


def test_wrong_native_source_fails_closed(monkeypatch):
    import splitfleet.runtime.native_frontier as frontier
    original = frontier.inspect.getsource
    monkeypatch.setattr(frontier.inspect,"getsource",lambda f: original(f)+"\n# changed\n")
    with pytest.raises(RuntimeError,match="source/signature"):
        frontier._native_bindings()


def test_wrong_native_signature_fails_closed(monkeypatch):
    import splitfleet.runtime.native_frontier as frontier
    monkeypatch.setattr(frontier.inspect,"signature",lambda _: "(changed)")
    with pytest.raises(RuntimeError,match="source/signature"):
        frontier._native_bindings()


def test_real_native_tensorflow_ce_sgd_state_and_changed_abi_in_isolated_cpu(tmp_path):
    if importlib.util.find_spec("tensorflow") is None or 6 not in os.sched_getaffinity(0):
        pytest.skip("The dedicated native control requires TensorFlow and CPU6")
    root = Path(__file__).resolve().parents[2]
    helper = "tests/integration/tensorflow_native_frontier_control.py"
    output = tmp_path / "native_control"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1",TF_NUM_INTRAOP_THREADS="1",TF_NUM_INTEROP_THREADS="1",
                       NVIDIA_TF32_OVERRIDE="0")
    prepare = subprocess.run([sys.executable,helper,"--prepare","--output",str(output)],
                             cwd=root,env=environment,timeout=120,capture_output=True,text=True,check=False)
    assert prepare.returncode == 0,prepare.stdout+prepare.stderr
    completed = subprocess.run([sys.executable,str(output/"runtime_snapshot"/helper),"--output",str(output),"--small-only"],
                               cwd=root,env=environment,timeout=120,capture_output=True,text=True,check=False)
    assert completed.returncode == 0,completed.stdout+completed.stderr
    summary = json.loads((output/"summary.json").read_text())
    assert summary["status"] == "passed" and summary["native_sdk_unmodified"]
    assert summary["cuda_jobs_launched"] == 0
    assert [row["batch_size"] for row in summary["small_model_steps"]] == [2,1]
    assert all(all(check["passed"] for check in row["checks"]) for row in summary["small_model_steps"])
    proof=json.loads((output/"small_frontier_proof.json").read_text())
    assert proof["native_partition_unchanged"] and proof["native_non_boundary_ir_unchanged"]
    assert proof["original_contract"]["boundary_schema_hash"] != proof["new_contract"]["boundary_schema_hash"]
    assert proof["removed_field_ids"]
