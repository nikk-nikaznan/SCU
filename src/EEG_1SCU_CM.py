import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy


import numpy as np
import yaml
import random
import argparse

from sklearn.model_selection import train_test_split

from model import SCU, weights_init
from utils import get_accuracy, CCM, load_data, load_label


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Setting seeds for reproducibility
seed_n = 74
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)


class SCU_DataModule(pl.LightningDataModule):
    def __init__(self, input_data, input_label, batch_size=32):
        super().__init__()
        self.input_data = input_data
        self.input_label = input_label
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.input_label, test_size=0.2, stratify=self.input_label
        )

    def train_dataloader(self):
        train_dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.y_train))
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        test_dataset = TensorDataset(torch.Tensor(self.X_test), torch.Tensor(self.y_test))
        return DataLoader(test_dataset, batch_size=self.batch_size)


class SCU_Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SCU(self.config)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='MULTICLASS')
        self.val_acc = Accuracy(task='MULTICLASS')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y.long())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.val_acc(torch.argmax(y_hat, dim=1), y.long())
        return y_hat, y

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    # def data_prep(self) -> None:
    #     # split the data training and testing
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    #         self.input_data, self.input_label, test_size=0.2, stratify=self.input_label
    #     )

    #     self._load_model()
    #     self._build_training_objects()
    #     # training
    #     self._train_SCU()
    #     # testing
    #     test_cumulative_accuracy, ntest, cnf_labels, cnf_predictions = self._test_SCU()
    #     print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy / ntest * 100)))
    #     # compute and plot confusion matrix
    #     CCM(cnf_labels, cnf_predictions)

def main(args):
    config_file = args.config_file
    config = yaml.safe_load(open(config_file))

    input_data = load_data()
    input_label = load_label()

    datamodule = SCU_DataModule(input_data, input_label)
    model = SCU_Model(config)

    trainer = pl.Trainer(max_epochs=config["num_epochs"], gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/scu.yaml",
        help="location of YAML config to control training",
    )
    args = parser.parse_args()

    args = parser.parse_args()
    main(args)