import os

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class SoccerActionDataset(Dataset):
    """
    A PyTorch Dataset for the Soccer Action Spotting challenge.

    The dataset loads sequences of features and labels from .npy files and slices them into windows for training or evaluation.

    Attributes:
        data (list of tuple): A list of tuples, where each tuple contains the paths to the features and labels files and the sequence length.
        window_size (int): The size of the windows to slice the sequences into.
        step_size (int): The number of frames to shift the window at each step.
        index_map (list of tuple): A list of tuples, where each tuple maps an index to a sequence and a window start.
    """

    def __init__(self, data, window_size, step_size=1, transform=None):
        self.window_size = window_size
        self.step_size = step_size
        self.data = data
        self.transform = transform
        self.index_map = self.create_index_map()

    def create_index_map(self):
        index_map = []
        for sequence_index, (_, _, sequence_length) in enumerate(self.data):
            for window_start in range(0, sequence_length - self.window_size + 1, self.step_size):
                index_map.append((sequence_index, window_start))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sequence_index, window_start = self.index_map[idx]
        features_path, labels_path, _ = self.data[sequence_index]

        # Load the entire sequence first
        features_sequence = np.load(features_path)
        labels_sequence = np.load(labels_path)

        # Then slice the sequence according to the window start and size
        features = features_sequence[window_start : window_start + self.window_size]
        labels = labels_sequence[window_start : window_start + self.window_size]

        # Convert to PyTorch tensors
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).float()

        if self.transform:
            features, labels = self.transform(features, labels)

        return features, labels


class SoccerActionDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the Soccer Action Spotting challenge.

    The DataModule prepares and sets up the data for training, validation, and testing.

    Attributes:
        data_dir (str): The directory containing the data.
        window_size (int): The size of the windows to slice the sequences into.
        step_size (int): The number of frames to shift the window at each step.
        batch_size (int): The size of the batches to load the data in.
    """

    def __init__(self, data_dir, window_size, step_size=1, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size

    def prepare_data(self):
        # Index the data instead of loading it
        self.data = {}
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.data_dir, split)
            feature_files = sorted([f for f in os.listdir(split_dir) if f.endswith("features.npy")])
            label_files = sorted([f for f in os.listdir(split_dir) if f.endswith("labels.npy")])
            self.data[split] = []
            for f, l in zip(feature_files, label_files):
                labels_path = os.path.join(split_dir, l)
                sequence_length = np.load(labels_path).shape[0]
                self.data[split].append((os.path.join(split_dir, f), labels_path, sequence_length))

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SoccerActionDataset(self.data["train"], self.window_size, self.step_size)
            self.valid_dataset = SoccerActionDataset(self.data["val"], self.window_size, self.step_size)

        if stage == "test" or stage is None:
            self.test_dataset = SoccerActionDataset(self.data["test"], self.window_size, self.step_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)


if __name__ == "__main__":
    data_dir = "data"
    batch_size = 8
    window_size = 16
    step_size = 1
    dm = SoccerActionDataModule(data_dir, window_size, step_size=step_size, batch_size=batch_size)
    dm.prepare_data()
    dm.setup()

    print(len(dm.train_dataloader()))
    for i, batch in enumerate(dm.train_dataloader()):
        print(i, batch[0].shape, batch[1].shape)
