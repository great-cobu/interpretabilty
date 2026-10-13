"""Data loading, preprocessing, and PyTorch Lightning DataModule for MNIST."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms

# Standard MNIST mean and standard deviation for normalization
MNIST_MEAN: float = 0.1307
MNIST_STD: float = 0.3081


def preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    normalize: bool = True,
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
) -> torch.Tensor:
    """Preprocess a single MNIST image into a normalized PyTorch tensor.

    Args:
        image: PIL Image, NumPy array, or Torch Tensor of shape (28, 28) or (1, 28, 28).
        normalize: Whether to apply standardization (mean and standard deviation).
        mean: Normalization mean (default: 0.1307).
        std: Normalization std (default: 0.3081).

    Returns:
        Tensor of shape (1, 28, 28) with dtype torch.float32.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
    elif isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] != 1:
            tensor = tensor[:1]
        if normalize and tensor.min() >= 0.0:
            tensor = (tensor - mean) / std
        return tensor
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if normalize:
        arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).unsqueeze(0)  # Shape: (1, 28, 28)
    return tensor


def create_mnist_transform(
    normalize: bool = True,
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
) -> Callable[[Any], Dict[str, torch.Tensor]]:
    """Create a transform function or Compose pipeline for MNIST images.

    Args:
        normalize: Whether to apply standardization.
        mean: Normalization mean.
        std: Normalization standard deviation.

    Returns:
        Callable that transforms raw images/batches into tensors.
    """
    def transform_fn(batch_or_img: Any) -> Any:
        if isinstance(batch_or_img, dict):
            # Batch dict format: {"image": [...], "label": [...]} or {"pixel_values": [...]}
            if "image" in batch_or_img:
                tensors = [preprocess_image(img, normalize=normalize, mean=mean, std=std) for img in batch_or_img["image"]]
                res = {"pixel_values": torch.stack(tensors)}
                if "label" in batch_or_img:
                    res["labels"] = torch.tensor(batch_or_img["label"], dtype=torch.long)
                return res
            if "pixel_values" in batch_or_img:
                return batch_or_img

        return preprocess_image(batch_or_img, normalize=normalize, mean=mean, std=std)

    return transform_fn


class MNISTDataset(Dataset):
    """Custom PyTorch Dataset for MNIST samples with dict or tuple output support."""

    def __init__(
        self,
        pixel_values: Union[torch.Tensor, List[Any]],
        labels: Optional[Union[torch.Tensor, List[int]]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(pixel_values, list):
            self.pixel_values = [
                preprocess_image(x) if not isinstance(x, torch.Tensor) else x
                for x in pixel_values
            ]
            self.pixel_values = torch.stack(self.pixel_values) if self.pixel_values else torch.empty(0)
        else:
            self.pixel_values = pixel_values

        if labels is not None:
            if isinstance(labels, list):
                self.labels = torch.tensor(labels, dtype=torch.long)
            else:
                self.labels = labels.long()
        else:
            self.labels = None

        self.transform = transform

    def __len__(self) -> int:
        return len(self.pixel_values)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.pixel_values[idx]
        if self.transform is not None:
            x = self.transform(x)

        item: Dict[str, torch.Tensor] = {"pixel_values": x}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class TorchvisionMNISTWrapper(Dataset):
    """Wraps torchvision MNIST to output consistent {'pixel_values', 'labels'} dicts."""

    def __init__(self, tv_dataset: tv_datasets.MNIST) -> None:
        self.tv_dataset = tv_dataset

    def __len__(self) -> int:
        return len(self.tv_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, label = self.tv_dataset[idx]
        if not isinstance(img, torch.Tensor):
            img = preprocess_image(img, normalize=True)
        return {"pixel_values": img, "labels": torch.tensor(label, dtype=torch.long)}


class MNISTDataCollator:
    """Collates a list of samples into stacked pixel_values and labels batch."""

    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}

        first = batch[0]
        if isinstance(first, dict):
            collated: Dict[str, torch.Tensor] = {}
            if "pixel_values" in first:
                collated["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
            elif "image" in first:
                collated["pixel_values"] = torch.stack([preprocess_image(b["image"]) for b in batch])

            if "labels" in first:
                collated["labels"] = torch.stack([
                    b["labels"] if isinstance(b["labels"], torch.Tensor) else torch.tensor(b["labels"], dtype=torch.long)
                    for b in batch
                ])
            elif "label" in first:
                collated["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)

            return collated

        if isinstance(first, (tuple, list)):
            xs = [b[0] for b in batch]
            ys = [b[1] for b in batch]
            return {
                "pixel_values": torch.stack(xs),
                "labels": torch.tensor(ys, dtype=torch.long) if not isinstance(ys[0], torch.Tensor) else torch.stack(ys),
            }

        raise TypeError(f"Unsupported item type in batch: {type(first)}")


def load_mnist_datasets(
    data_dir: str = "./data",
    train_val_split: float = 0.1,
    seed: int = 42,
    normalize: bool = True,
    download: bool = True,
) -> Dict[str, Dataset]:
    """Load MNIST datasets using torchvision without Hugging Face.

    Args:
        data_dir: Directory to cache and store downloaded dataset.
        train_val_split: Fraction of train dataset reserved for validation (0.0 to disable).
        seed: Random seed for train/val split.
        normalize: Whether to apply standardization.
        download: Whether to download if not present locally.

    Returns:
        Dictionary containing 'train', 'validation' (if split > 0), and 'test' datasets.
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    transform = transforms.Compose(transform_list)

    raw_train = tv_datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )
    raw_test = tv_datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=download,
    )

    if train_val_split > 0.0:
        val_size = int(len(raw_train) * train_val_split)
        train_size = len(raw_train) - val_size
        train_subset, val_subset = random_split(
            raw_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        return {
            "train": TorchvisionMNISTWrapper(train_subset),
            "validation": TorchvisionMNISTWrapper(val_subset),
            "test": TorchvisionMNISTWrapper(raw_test),
        }

    return {
        "train": TorchvisionMNISTWrapper(raw_train),
        "test": TorchvisionMNISTWrapper(raw_test),
    }


class MNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MNIST digit dataset."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        eval_batch_size: int = 128,
        train_val_split: float = 0.1,
        seed: int = 42,
        num_workers: int = 0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.train_val_split = train_val_split
        self.seed = seed
        self.num_workers = num_workers
        self.normalize = normalize
        self.collator = MNISTDataCollator()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download dataset if not already present."""
        tv_datasets.MNIST(root=self.data_dir, train=True, download=True)
        tv_datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Partition datasets for training, validation, and testing."""
        datasets = load_mnist_datasets(
            data_dir=self.data_dir,
            train_val_split=self.train_val_split,
            seed=self.seed,
            normalize=self.normalize,
            download=False,
        )
        self.train_dataset = datasets["train"]
        self.val_dataset = datasets.get("validation")
        self.test_dataset = datasets["test"]

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            self.setup("fit")
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )
