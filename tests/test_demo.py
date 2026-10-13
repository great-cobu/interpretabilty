import json

import pytest
import torch

from models.MNIST.demo import train_demo


def test_selected_epochs_create_fresh_runs_and_restore_threads(tiny_mnist, tmp_path):
    previous_threads = torch.get_num_threads()
    first = train_demo(tmp_path, epochs=2, threads=2)
    second = train_demo(tmp_path, epochs=1, threads=2)
    assert first != second
    first_summary = json.loads((first / "training_summary.json").read_text())
    second_summary = json.loads((second / "training_summary.json").read_text())
    assert first_summary["completed_epochs"] == 2
    assert second_summary["completed_epochs"] == 1
    assert first_summary["smoke_run"]
    assert first_summary["source_checkpoint"] is None
    assert (first / "checkpoints/best.ckpt").is_file()
    assert torch.get_num_threads() == previous_threads


def test_failed_training_restores_threads(tiny_mnist, tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    def fail(*args):
        raise RuntimeError("training failed")
    monkeypatch.setattr("mnist_cnn.demo.train", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        train_demo(tmp_path, epochs=2, threads=2)
    assert torch.get_num_threads() == previous_threads


@pytest.mark.parametrize("epochs", [0, 101, 2.5, True])
def test_invalid_epochs_create_no_artifacts(tmp_path, epochs):
    with pytest.raises(ValueError, match="epoch count"):
        train_demo(tmp_path, epochs=epochs, threads=2)
    assert not (tmp_path / "runs").exists()
