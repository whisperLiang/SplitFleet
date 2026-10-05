"""An independent remote coordinator receives the physical run inputs."""

from types import SimpleNamespace
import json
import pytest

from experiments import orchestrate_physical_multitask as orchestrator


@pytest.mark.parametrize("model_name", ["rfdetr_nano", "resnet50_pretrained", "bert_base", "deeplabv3_resnet50"])
@pytest.mark.parametrize("layout", ["legacy", "first_gpu", "all_gpu"])
def test_remote_server_outside_worker_hosts_receives_code_and_bundle(tmp_path, monkeypatch, model_name, layout):
    hosts = [
        {"id": f"edge-{index}", "ssh": f"edge-{index}", "workdir": "/work"}
        for index in range(3)
    ]
    if layout == "first_gpu":
        hosts[0]["workers"] = ["gpu"]
    if layout == "all_gpu":
        for host in hosts:
            host["workers"] = ["gpu"]
    server_ssh = "coordinator"
    commands = []
    remote_calls = []
    training_calls = []
    preparations = []

    def fake_ssh(host, script, **kwargs):
        remote_calls.append((host["ssh"], script))
        if "hashlib.sha256" in script:
            return (tmp_path / "remote-server" / "source_hashes.json").read_text()
        return "GPU 0: test\nNTPSynchronized=yes\ncommit" if "nvidia-smi" in script else ""

    def fake_run(args, **kwargs):
        commands.append(list(args))
        return SimpleNamespace(returncode=0)

    def fake_prepare_bundle(*, task, output, **kwargs):
        preparations.append(kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"server")
        for index in range(6):
            output.with_name(f"{task}.client_{index}.pt").write_bytes(b"client")
        return {"task": task}

    monkeypatch.setattr(orchestrator, "_ssh", fake_ssh)
    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    monkeypatch.setattr(orchestrator, "prepare_bundle", fake_prepare_bundle)

    def fake_train(config, **kwargs):
        training_calls.append(kwargs)
        return {"task": kwargs["task"], "method": kwargs["method"], "final_metrics": {}}

    monkeypatch.setattr(orchestrator, "_run_one", fake_train)

    orchestrator.run_matrix(
        {"hosts": hosts, "server": {"ssh": server_ssh}},
        run_id="remote-server", output_root=tmp_path,
        tasks=["image_classification"], methods=list(orchestrator.METHODS), data_root="unused",
        seed=1, train_samples=1, test_samples=1, batch_size=1,
        rounds=1, timeout=1,
        model_name=model_name,
    )

    remote_root = "/tmp/splitfleet_physical_multitask_remote-server"
    assert any(command[0] == "rsync" and command[-1] == f"{server_ssh}:{remote_root}/"
               for command in commands)
    assert any(command[0] == "scp" and command[-1] ==
               f"{server_ssh}:{remote_root}/bundles/image_classification.pt"
               for command in commands)
    assert (server_ssh, f"rm -rf {remote_root}") in remote_calls
    fixed = [call for call in training_calls if call["method"] == "splitfed_fixed"]
    assert [call["fixed_boundary"] for call in fixed] == ["25%", "50%", "75%"]
    assert len({call["run_dir"] for call in training_calls}) == 6
    assert len({call["bundle_path"] for call in training_calls}) == 1
    assert all(call["bundle"] is training_calls[0]["bundle"] for call in training_calls)
    assert all(call["local_source_root"] == tmp_path / "remote-server" / "source_snapshot"
               for call in training_calls)
    assert (tmp_path / "remote-server" / "source_snapshot" / "splitfleet" / "__init__.py").exists()
    worker_ids = preparations[0]["worker_ids"]
    assert len(worker_ids) == {"legacy": 6, "first_gpu": 5, "all_gpu": 3}[layout]
    assert worker_ids[0] == ("edge-0-cpu" if layout == "legacy" else "edge-0-gpu")
    assert any(command[-1] == f"edge-1:{remote_root}/bundles/image_classification.client_{2 if layout == 'legacy' else 1}.pt"
               for command in commands)
    assert all("device_profiles" not in call and "online_initialization" not in call for call in training_calls)
    assert {row["status"] for row in json.loads((tmp_path / "remote-server" / "attempts.json").read_text())} == {"completed"}
