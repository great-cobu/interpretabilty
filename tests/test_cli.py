import json
import subprocess
import sys

import pytest

from models.MNIST.cli import main


def test_module_and_console_help():
    console = subprocess.run(["mnist-cnn", "--help"], capture_output=True, text=True)
    module = subprocess.run([sys.executable, "-m", "mnist_cnn", "--help"], capture_output=True, text=True)
    assert console.returncode == module.returncode == 0
    assert console.stdout == module.stdout


@pytest.mark.parametrize("command", ["evaluate"])
def test_unimplemented_commands_fail(command):
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2


def test_check_config_and_invalid_input(tmp_path, capsys):
    output = tmp_path / "run"
    assert main(["check-config", "--batch-size", "3", "--output-dir", str(output)]) == 0
    assert "batch_size: 3" in capsys.readouterr().out
    assert main(["check-config", "--output-dir", str(output)]) == 2
    assert "not empty" in capsys.readouterr().err
    assert main(["check-config", "--batch-size", "-1", "--output-dir", str(tmp_path / "invalid")]) == 2
    assert not (tmp_path / "invalid").exists()


def test_inspection_exports(tiny_mnist, tmp_path):
    output = tmp_path / "inspect"
    assert main(["inspect-data", "--no-download", "--batch-size", "4", "--output-dir", str(output)]) == 0
    report = json.loads((output / "data_summary.json").read_text())
    assert report["splits"]["train"]["count"] == 18
    assert (output / "split_indices.json").is_file()
