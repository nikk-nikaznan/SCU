from typing import Union

import lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class SCU_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_data: Union[list, torch.Tensor],
        input_label: Union[list, torch.Tensor],
        batch_size: int,
        seed_n: int,
    ):
        """
        Lightning DataModule for SCU model training and testing.

        Args:
            input_data (Union[list, torch.Tensor]): Input data for the model.
            input_label (Union[list, torch.Tensor]): Corresponding labels for the input data.
            batch_size (int): Batch size for DataLoader.
            seed_n (int): Seed for random number generator.
        """
        super().__init__()
        self.input_data = input_data
        self.input_label = input_label
        self.batch_size = batch_size
        self.seed_n = seed_n

    def setup(self, stage: str = None) -> None:
        """
        Setup datasets for training and testing.

        Args:
            stage (str, optional): Stage of setup (train, test).
            Defaults to None.
        """
        generator = torch.Generator().manual_seed(self.seed_n)
        # Convert your input data and labels to PyTorch tensors
        input_data_tensor = torch.tensor(self.input_data, dtype=torch.float32)
        input_label_tensor = torch.tensor(self.input_label, dtype=torch.float32)

        # Combine data and labels into a TensorDataset
        dataset = TensorDataset(input_data_tensor, input_label_tensor)

        # Define the sizes of train and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # Use random_split to split the dataset into train and test sets
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training set.

        Returns:
            DataLoader: DataLoader for the training set.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the test set.

        Returns:
            DataLoader: DataLoader for the test set.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
