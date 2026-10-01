"""An independent remote coordinator receives the physical run inputs."""

from types import SimpleNamespace

from experiments import orchestrate_physical_multitask as orchestrator


def test_remote_server_outside_worker_hosts_receives_code_and_bundle(tmp_path, monkeypatch):
    hosts = [
        {"id": f"edge-{index}", "ssh": f"edge-{index}", "workdir": "/work"}
        for index in range(3)
    ]
    server_ssh = "coordinator"
    commands = []
    remote_calls = []
    training_calls = []

    def fake_ssh(host, script, **kwargs):
        remote_calls.append((host["ssh"], script))
        return "GPU 0: test\nNTPSynchronized=yes\ncommit" if "nvidia-smi" in script else ""

    def fake_run(args, **kwargs):
        commands.append(list(args))
        return SimpleNamespace(returncode=0)

    def fake_prepare_bundle(*, task, output, **kwargs):
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
