import os
import unittest
import numpy as np
import shutil

from datamodule import SoccerActionDataModule, SoccerActionDataset


class TestSoccerActionDataset(unittest.TestCase):
    def setUp(self):
        # Create a small dataset for testing
        self.data_dir = "./test_data"
        os.makedirs(self.data_dir, exist_ok=True)
        for i in range(5):
            features = np.random.rand(10, 784)
            labels = np.random.randint(0, 3, 10)
            np.save(os.path.join(self.data_dir, f"features_{i}.npy"), features)
            np.save(os.path.join(self.data_dir, f"labels_{i}.npy"), labels)
        self.data = [
            (
                os.path.join(self.data_dir, f"features_{i}.npy"),
                os.path.join(self.data_dir, f"labels_{i}.npy"),
                10,
            )
            for i in range(5)
        ]

    def test_len(self):
        dataset = SoccerActionDataset(self.data, window_size=5, step_size=2)
        self.assertEqual(
            len(dataset), 15
        )  # With a window size of 5 and step size of 2, there should be 3 windows per sequence

    def test_getitem(self):
        dataset = SoccerActionDataset(self.data, window_size=5, step_size=2)
        features, labels = dataset[0]
        self.assertEqual(
            features.shape, (5, 784)
        )  # The features should have shape (window_size, 784)
        self.assertEqual(
            labels.shape, (5,)
        )  # The labels should have shape (window_size,)

    def tearDown(self):
        # Clean up the test data
        shutil.rmtree(self.data_dir)


class TestSoccerActionDataModule(unittest.TestCase):
    def setUp(self):
        # Create a small dataset for testing
        self.data_dir = "./test_data"
        os.makedirs(self.data_dir, exist_ok=True)
        for i in range(5):
            for set_name in ["train", "valid", "test"]:
                set_dir = os.path.join(self.data_dir, set_name)
                os.makedirs(set_dir, exist_ok=True)
                features = np.random.rand(10, 784)
                labels = np.random.randint(0, 3, 10)
                np.save(os.path.join(set_dir, f"features_{i}.npy"), features)
                np.save(os.path.join(set_dir, f"labels_{i}.npy"), labels)

    def test_dataloaders(self):
        datamodule = SoccerActionDataModule(
            self.data_dir, window_size=5, step_size=2, batch_size=2
        )
        datamodule.prepare_data()
        datamodule.setup()

        # Test the train dataloader
        train_dataloader = datamodule.train_dataloader()
        valid_dataloader = datamodule.val_dataloader()
        test_dataloader = datamodule.test_dataloader()
        for dataloader in [train_dataloader, valid_dataloader, test_dataloader]:
            for batch in dataloader:
                features, labels = batch
                self.assertEqual(
                    features.shape, (2, 5, 784)
                )  # The features should have shape (batch_size, window_size, 784)
                self.assertEqual(
                    labels.shape, (2, 5)
                )  # The labels should have shape (batch_size, window_size)
                break  # Test one batch for brevity

    def tearDown(self):
        # Clean up the test data
        shutil.rmtree(self.data_dir)


if __name__ == "__main__":
    unittest.main()
