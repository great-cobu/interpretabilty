from dataclasses import replace

import pytest
import yaml

from models.MNIST.config import Config, ConfigError, load_config
from models.MNIST.runs import create_run


def test_precedence_and_sanitized_run(tmp_path, monkeypatch):
    path = tmp_path / "input.yaml"
    path.write_text('data:\n  batch_size: 16\ntracking:\n  uri: https://yaml.example\n')
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://env.example")
    assert load_config(path).tracking.uri == "https://env.example"
    config = load_config(path, {"data.batch_size": 7, "tracking.uri": "https://user:password@cli.example/path?token=abc", "tracking.tags": {"api_key": "hidden", "purpose": "test"}})
    assert config.data.batch_size == 7
    config = replace(config, output_dir=str(tmp_path / "run"))
    resolved, output = create_run(config, "test")
    saved = (output / "config.yaml").read_text()
    assert "password" not in saved and "hidden" not in saved and "abc" not in saved
    parsed = yaml.safe_load(saved)
    assert parsed["tracking"]["uri"] == "https://cli.example/path"
    assert parsed["output_dir"] == str(output.resolve())
    assert resolved.data.batch_size == 7
    with pytest.raises(ValueError, match="not empty"):
        create_run(config, "test")


@pytest.mark.parametrize("document", [
    'data:\n  batch_size: 0', 'data:\n  batch_size: true',
    'data:\n  std: 0', 'data:\n  split_seed: -1', 'model:\n  dropout: 1',
    'optimizer:\n  learning_rate: .nan', 'optimizer:\n  weight_decay: -1',
    'trainer:\n  limit_train_batches: 0', 'trainer:\n  limit_val_batches: true',
    'trainer:\n  limit_train_batches: 0.5',
    'trainer:\n  devices: 2', 'trainer:\n  precision: 16-mixed',
    'data:\n  unknown: 1', 'unknown: 1', 'tracking:\n  tags: []',
    'data: []', '- a', 'data:\n  batch_size: 1\n  batch_size: 2',
    'output_dir: false', 'tracking:\n  enabled: true',
])
def test_reject_invalid_config(tmp_path, monkeypatch, document):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    path = tmp_path / "invalid.yaml"
    path.write_text(document)
    with pytest.raises(ConfigError):
        load_config(path)


def test_config_error_does_not_echo_secret_yaml(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text('tracking: [SECRET_VALUE')
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "SECRET_VALUE" not in str(exc.value)


def test_defaults_equal_yaml(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert load_config("configs/default.yaml") == Config()


def test_missing_config_explains_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="Configuration file not found.*current working directory"):
        load_config("configs/default.yaml")


def test_invalid_yaml_has_specific_error(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("data: [")
    with pytest.raises(ConfigError, match="Invalid configuration YAML"):
        load_config(path)


def test_config_requires_utf8(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigError, match="readable UTF-8 YAML"):
        load_config(path)


def test_empty_output_accepted_and_auto_paths_unique(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, first = create_run(Config(), "check-config")
    _, second = create_run(Config(), "check-config")
    assert first != second
    empty = tmp_path / "empty"
    empty.mkdir()
    create_run(replace(Config(), output_dir=str(empty)), "check-config")
