"""Unit tests for MNIST CNN training, testing, data loading, and PyTorch Lightning integration."""

from dataclasses import dataclass
import shutil
import tempfile
from typing import Any
import unittest
import numpy as np
from PIL import Image
import torch

from models.MNIST import (
    MNISTCNN,
    MNISTCNNConfig,
    MNISTDataCollator,
    MNISTDataset,
    MNISTTrainingConfig,
    compute_metrics,
    create_mnist_transform,
    predict,
    preprocess_image,
    test_mnist,
    train_mnist,
)


@dataclass
class MockEvalPrediction:
    """Mock container mimicking evaluation predictions for metrics."""

    predictions: Any
    label_ids: Any


class TestMNISTModel(unittest.TestCase):
    """Test CNN model architecture and PyTorch Lightning integration."""

    def setUp(self) -> None:
        self.config = MNISTCNNConfig(
            in_channels=1,
            num_classes=10,
            conv_channels=[16, 32],
            fc_dim=64,
            dropout_rate=0.2,
        )
        self.model = MNISTCNN(self.config)

    def test_forward_without_labels(self) -> None:
        pixel_values = torch.randn(2, 1, 28, 28)
        outputs = self.model(pixel_values=pixel_values)
        self.assertIsNone(outputs.loss)
        self.assertEqual(outputs.logits.shape, (2, 10))

    def test_forward_with_labels(self) -> None:
        pixel_values = torch.randn(4, 1, 28, 28)
        labels = torch.tensor([0, 3, 7, 9], dtype=torch.long)
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(outputs.loss.item() > 0)
        self.assertEqual(outputs.logits.shape, (4, 10))

    def test_forward_hidden_states(self) -> None:
        pixel_values = torch.randn(2, 1, 28, 28)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        self.assertIsNotNone(outputs.hidden_states)
        self.assertTrue(len(outputs.hidden_states) > 0)

    def test_get_feature_maps(self) -> None:
        pixel_values = torch.randn(2, 1, 28, 28)
        feature_maps = self.model.get_feature_maps(pixel_values)
        self.assertIn("conv1", feature_maps)
        self.assertIn("conv2", feature_maps)
        self.assertIn("conv3", feature_maps)
        self.assertIn("conv4", feature_maps)
        self.assertEqual(feature_maps["conv1"].shape[1], 16)
        self.assertEqual(feature_maps["conv4"].shape[1], 32)

    def test_lightning_training_step(self) -> None:
        pixel_values = torch.randn(4, 1, 28, 28)
        labels = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        batch = {"pixel_values": pixel_values, "labels": labels}
        loss = self.model.training_step(batch, 0)
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_save_and_load_pretrained(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            self.model.save_pretrained(temp_dir)
            loaded_model = MNISTCNN.from_pretrained(temp_dir)
            pixel_values = torch.randn(2, 1, 28, 28)

            self.model.eval()
            loaded_model.eval()

            with torch.no_grad():
                orig_out = self.model(pixel_values=pixel_values).logits
                loaded_out = loaded_model(pixel_values=pixel_values).logits

            torch.testing.assert_close(orig_out, loaded_out)
        finally:
            shutil.rmtree(temp_dir)


class TestMNISTData(unittest.TestCase):
    """Test data preprocessing and transformation utilities."""

    def test_preprocess_pil_image(self) -> None:
        img = Image.new("L", (28, 28), color=128)
        tensor = preprocess_image(img, normalize=True)
        self.assertEqual(tensor.shape, (1, 28, 28))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_preprocess_numpy(self) -> None:
        arr = np.zeros((28, 28), dtype=np.uint8)
        tensor = preprocess_image(arr, normalize=False)
        self.assertEqual(tensor.shape, (1, 28, 28))
        self.assertEqual(tensor.sum().item(), 0.0)

    def test_create_mnist_transform(self) -> None:
        transform = create_mnist_transform(normalize=True)
        batch = {
            "image": [Image.new("L", (28, 28), color=0), Image.new("L", (28, 28), color=255)],
            "label": [3, 8],
        }
        res = transform(batch)
        self.assertEqual(res["pixel_values"].shape, (2, 1, 28, 28))
        self.assertEqual(res["labels"].shape, (2,))

    def test_mnist_dataset_and_collator(self) -> None:
        pixels = [torch.randn(1, 28, 28) for _ in range(4)]
        labels = [0, 1, 2, 3]
        dataset = MNISTDataset(pixels, labels)
        self.assertEqual(len(dataset), 4)

        sample = dataset[0]
        self.assertEqual(sample["pixel_values"].shape, (1, 28, 28))
        self.assertEqual(sample["labels"].item(), 0)

        collator = MNISTDataCollator()
        batch = collator([dataset[i] for i in range(4)])
        self.assertEqual(batch["pixel_values"].shape, (4, 1, 28, 28))
        self.assertEqual(batch["labels"].shape, (4,))


class TestComputeMetrics(unittest.TestCase):
    """Test evaluation metric calculations."""

    def test_metrics_with_object(self) -> None:
        # Perfect predictions
        logits = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
        labels = np.array([0, 1, 0])
        eval_pred = MockEvalPrediction(predictions=logits, label_ids=labels)
        metrics = compute_metrics(eval_pred)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_metrics_direct(self) -> None:
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        metrics = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 1.0)


class TestEndToEndPipeline(unittest.TestCase):
    """Test training, testing, and prediction pipeline end-to-end with PyTorch Lightning."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        # Create small synthetic dataset
        train_pixels = torch.randn(16, 1, 28, 28)
        train_labels = torch.tensor([i % 10 for i in range(16)], dtype=torch.long)

        val_pixels = torch.randn(8, 1, 28, 28)
        val_labels = torch.tensor([i % 10 for i in range(8)], dtype=torch.long)

        test_pixels = torch.randn(8, 1, 28, 28)
        test_labels = torch.tensor([i % 10 for i in range(8)], dtype=torch.long)

        self.mock_datasets = {
            "train": MNISTDataset(train_pixels, train_labels),
            "validation": MNISTDataset(val_pixels, val_labels),
            "test": MNISTDataset(test_pixels, test_labels),
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_train_test_predict(self) -> None:
        config = MNISTCNNConfig(
            in_channels=1,
            num_classes=10,
            conv_channels=[8, 16],
            fc_dim=32,
        )
        model = MNISTCNN(config)

        # 1. Train with PyTorch Lightning
        trained_model, trainer, metrics = train_mnist(
            model=model,
            datasets=self.mock_datasets,
            output_dir=self.temp_dir,
            epochs=1,
            batch_size=8,
            eval_batch_size=8,
            accelerator="cpu",
            report_to="none",
        )
        self.assertIsNotNone(trained_model)
        self.assertIsNotNone(trainer)

        # 2. Test
        test_results = test_mnist(
            model=trained_model,
            test_dataset=self.mock_datasets["test"],
            batch_size=8,
            device="cpu",
            verbose=False,
        )
        self.assertIn("test_loss", test_results)
        self.assertIn("test_accuracy", test_results)
        self.assertIn("classification_report", test_results)
        self.assertIn("confusion_matrix", test_results)

        # Test loading from directory in test_mnist
        test_from_disk = test_mnist(
            model=self.temp_dir,
            test_dataset=self.mock_datasets["test"],
            batch_size=8,
            device="cpu",
            verbose=False,
        )
        self.assertEqual(test_results["test_accuracy"], test_from_disk["test_accuracy"])

        # 3. Predict
        test_img = Image.new("L", (28, 28), color=50)
        pred_res = predict(trained_model, test_img, device="cpu")
        self.assertEqual(len(pred_res["predictions"]), 1)
        self.assertIn(pred_res["predictions"][0], list(range(10)))
        self.assertEqual(pred_res["probabilities"].shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
