from dataclasses import replace
import json

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from models.MNIST.config import DataConfig
from models.MNIST.data import MNISTDataModule, resolve_accelerator


def test_splits_repeatable_independent_of_global_rng_and_setup(tiny_mnist):
    first = MNISTDataModule()
    first.prepare_data()
    assert first.train_dataset is None
    first.setup("fit")
    assert first.test_dataset is None
    indices = first.split_indices
    first.setup("validate")
    assert first.split_indices is indices
    torch.manual_seed(876)
    second = MNISTDataModule()
    second.setup()
    assert second.split_indices == first.split_indices
    train, val = set(indices["train"]), set(indices["validation"])
    assert not train & val
    assert train | val == set(range(23))
    different = MNISTDataModule(DataConfig(split_seed=7))
    different.setup("fit")
    assert different.split_indices != indices


def test_batches_transforms_order_and_partial_batches(tiny_mnist):
    module = MNISTDataModule(DataConfig(batch_size=4))
    module.setup()
    loader = module.train_dataloader()
    assert isinstance(loader.sampler, RandomSampler)
    assert not loader.pin_memory and not loader.persistent_workers
    batches = list(loader)
    assert sum(len(y) for x, y in batches) == 18
    assert batches[-1][0].shape == (2, 1, 28, 28)
    test_loader = module.test_dataloader()
    assert isinstance(test_loader.sampler, SequentialSampler)
    x, y = next(iter(test_loader))
    assert x.dtype == torch.float32 and y.dtype == torch.int64
    assert y.tolist() == [0, 1, 2, 3]
    assert torch.allclose(x[0], torch.full_like(x[0], -0.1307 / 0.3081))
    assert len(list(module.val_dataloader())[-1][1]) == 1
    assert torch.cat([y for x, y in module.predict_dataloader()]).tolist() == [i % 10 for i in range(11)]


def test_seeded_training_order(tiny_mnist):
    modules = [MNISTDataModule(DataConfig(batch_size=4)) for _ in range(2)]
    for module in modules:
        module.setup("fit")
    order = [torch.cat([x[:, 0, 0, 0] for x, y in m.train_dataloader()]) for m in modules]
    assert torch.equal(*order)


def test_saved_indices_roundtrip_and_reject_corruption(tiny_mnist, tmp_path):
    original = MNISTDataModule()
    original.setup("fit")
    path = tmp_path / "indices.json"
    original.save_split_indices(path)
    restored = MNISTDataModule(DataConfig(split_indices=str(path)))
    restored.setup("validate")
    assert restored.split_indices == original.split_indices
    corrupted = json.loads(path.read_text())
    corrupted["train"][0] = corrupted["validation"][0]
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="partition"):
        MNISTDataModule(DataConfig(split_indices=str(path))).setup("fit")


def test_missing_cache_does_not_download(tmp_path):
    module = MNISTDataModule(DataConfig(directory=str(tmp_path), download=False))
    with pytest.raises(RuntimeError, match="prepare-data"):
        module.prepare_data()


def test_stage_guards_and_test_only_setup(tiny_mnist):
    module = MNISTDataModule()
    with pytest.raises(RuntimeError, match="setup"):
        module.train_dataloader()
    module.setup("test")
    assert module.train_dataset is None and len(module.test_dataset) == 11
    with pytest.raises(ValueError, match="Unsupported"):
        module.setup("typo")


def test_accelerator_validation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_accelerator("auto") == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_accelerator("cuda")


@pytest.mark.integration
def test_real_cached_mnist(tmp_path):
    module = MNISTDataModule(DataConfig(download=False))
    module.prepare_data()
    module.setup()
    assert [len(d) for d in (module.train_dataset, module.val_dataset, module.test_dataset)] == [55000, 5000, 10000]
    module.save_split_indices(tmp_path / "indices.json")
    restored = MNISTDataModule(replace(module.config, split_indices=str(tmp_path / "indices.json")))
    restored.setup("fit")
    assert restored.split_indices == module.split_indices
    train, val = set(module.split_indices["train"]), set(module.split_indices["validation"])
    assert not train & val and train | val == set(range(60000))
    for loader in (module.train_dataloader(), module.val_dataloader(), module.test_dataloader()):
        count = 0
        for x, y in loader:
            assert x.dtype == torch.float32 and y.dtype == torch.int64
            assert x.shape[1:] == (1, 28, 28)
            assert torch.isfinite(x).all() and 0 <= y.min() <= y.max() <= 9
            count += len(y)
        assert count == len(loader.dataset)
