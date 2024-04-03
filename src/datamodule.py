import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning as pl


class SCU_DataModule(pl.LightningDataModule):
    def __init__(self, input_data, input_label, batch_size, seed_n):
        super().__init__()
        self.input_data = input_data
        self.input_label = input_label
        self.batch_size = batch_size
        self.seed_n = seed_n

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(self.seed_n)
        # Convert your input data and labels to PyTorch tensors
        input_data_tensor = torch.tensor(self.input_data, dtype=torch.float64)
        input_label_tensor = torch.tensor(self.input_label, dtype=torch.float64)

        # Combine data and labels into a TensorDataset
        dataset = TensorDataset(input_data_tensor, input_label_tensor)

        # Define the sizes of train and test sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # Use random_split to split the dataset into train and test sets
        self.train_dataset, self.test_dataset = random_split(
            dataset, [train_size, test_size], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
