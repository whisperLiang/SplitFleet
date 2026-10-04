import shlex
import sys
import sysconfig
from pathlib import Path

from experiments.run_edge_standard_study import environment_setup_code


def test_environment_inherits_base_with_valid_pth_import(tmp_path, monkeypatch):
    target = tmp_path / "environment with 'quoted' path"
    inherited = tmp_path / "original interpreter packages"
    site_packages = target / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    site_packages.mkdir(parents=True)
    calls = []
    monkeypatch.setattr("subprocess.run", lambda arguments, **kwargs: calls.append((arguments, kwargs)))
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(inherited)})
    script = environment_setup_code(str(target))
    exec(compile(script, "remote-setup", "exec"), {})
    content = (site_packages / "splitfleet_base.pth").read_text()
    assert content == f"import site; site.addsitedir({str(inherited)!r})\n"
    compile(content, "splitfleet_base.pth", "exec")
    assert len(content.splitlines()) == 1
    assert calls[0][0][-1] == str(target)
    assert calls[1][0][0] == str(target / "bin/python")
    assert calls[1][0][-1] == "transformers==5.8.1"
    assert all(kwargs["check"] for _, kwargs in calls)


def test_remote_setup_survives_shell_transport():
    script = environment_setup_code("/tmp/edge model's environment")
    arguments = ["/usr/bin/python3", "-c", script]
    assert shlex.split(shlex.join(arguments)) == arguments
    compile(script, "remote-setup", "exec")
