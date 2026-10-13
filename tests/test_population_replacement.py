"""A code-defect retirement preserves all outcomes and every original seed."""
from pathlib import Path
import hashlib
import json
import pytest
from experiments.analysis.population_replacement import validate_replacement


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def retired(tmp_path):
    root,new=tmp_path/"old",tmp_path/"new"
    source=root/"runtime_snapshot/experiments/baselines/async_fl.py";source.parent.mkdir(parents=True);source.write_text("old_buffer_delta = True\n")
    hashes={"experiments/baselines/async_fl.py":hashlib.sha256(source.read_bytes()).hexdigest()}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    plan={"source_identity":identity,"source_root":str(root/"runtime_snapshot"),
        "method_order":["fedavg","fedasync","fedbuff","fedcompass","splitfed_fixed","splitfleet"],
        "stages":[{"name":n,"tasks":[n],"seeds":list(range(10501,10506)),"time_budget_sec":600 if n=="deep" else 300}
                  for n in ("resnet","bert","rfdetr","deep")]}
    bound={};bound[str(root/"frozen_plan.json")]=write(root/"frozen_plan.json",plan)
    new_plan={**plan,"source_identity":"corrected","source_root":str(new/"runtime_snapshot")}
    bound[str(new/"frozen_plan.json")]=write(new/"frozen_plan.json",new_plan)
    bound[str(root/"runtime_manifest.json")]=write(root/"runtime_manifest.json",{"source_identity":identity,"source_hashes":hashes})
    jobs=[]
    for seed in range(10501,10506):
        run_id=f"resnet_s{seed}";jobs.append({"run_id":run_id,"status":"completed"})
        p=root/"physical"/run_id/"attempts.json";bound[str(p)]=write(p,[{"status":"completed"}]*8)
    run_id="bert_s10501";jobs.append({"run_id":run_id,"status":"failed"})
    p=root/"physical"/run_id/"attempts.json";bound[str(p)]=write(p,[{"status":"completed"}]*2+[{"status":"running"}])
    bound[str(root/"ledger.json")]=write(root/"ledger.json",{"jobs":jobs,"finished_unix":3})
    proof=root/"evidence/reproduction.json";bound[str(proof)]=write(proof,{"schema":"splitfleet.fedbuff-buffer-regression.v1",
        "test_fixture_only":True,"included_in_primary":False,"final_state_all_finite":True,
        "evaluation_predictions_all_finite":False,"committed_running_variances":[.1,-.8],"source_sha256":hashes["experiments/baselines/async_fl.py"]})
    receipt={"schema":"splitfleet.method-correction-population-replacement.v1","created_unix":5,
        "original_root":str(root),"corrected_root":str(new),"original_source_identity":identity,"corrected_source_identity":"corrected",
        "expected_primary_jobs":160,"retired_successful_method_attempts":42,"retired_complete_paired_endpoints":40,
        "retired_unpaired_successes":2,"retained_controlled_interrupted_method_preparations":1,
        "quality_or_timing_selection_gate":False,"hyperparameters_or_seeds_changed":False,"original_statuses_modified":False,
        "bound_artifacts_sha256":bound}
    write(root/"replacement_population.json",receipt)
    return root,new,receipt


def test_whole_population_retirement_keeps_controls_and_interruption_counts(retired):
    root,new,receipt=retired
    corrected,result,artifacts=validate_replacement(root)
    assert corrected==new and result["retired_successful_method_attempts"]==42 and artifacts


@pytest.mark.parametrize("change",["seed","budget","buffer_size"])
def test_replacement_cannot_change_frozen_settings_even_rehashed(retired,change):
    root,new,receipt=retired;path=new/"frozen_plan.json";plan=json.loads(path.read_text())
    if change=="seed":plan["stages"][0]["seeds"][-1]=99999
    elif change=="budget":plan["stages"][0]["time_budget_sec"]=400
    else:plan["buffer_size"]=3
    receipt["bound_artifacts_sha256"][str(path)]=write(path,plan);write(root/"replacement_population.json",receipt)
    with pytest.raises(ValueError,match="settings"):
        validate_replacement(root)


def test_retirement_cannot_drop_unpaired_successes(retired):
    root,_,receipt=retired;receipt["retired_successful_method_attempts"]=40;receipt["retired_unpaired_successes"]=0
    write(root/"replacement_population.json",receipt)
    with pytest.raises(ValueError,match="attempted outcomes"):
        validate_replacement(root)


def test_low_accuracy_without_a_reproduced_defect_cannot_authorize_retirement(retired):
    root,_,receipt=retired;path=root/"evidence/reproduction.json";proof=json.loads(path.read_text());proof["evaluation_predictions_all_finite"]=True
    receipt["bound_artifacts_sha256"][str(path)]=write(path,proof);write(root/"replacement_population.json",receipt)
    with pytest.raises(ValueError,match="reproduced implementation defect"):
        validate_replacement(root)
