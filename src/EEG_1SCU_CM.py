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
import logging
logging.basicConfig(level=logging.INFO)

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
    def __init__(self, input_data, input_label, batch_size):
        super().__init__()
        self.input_data = input_data
        self.input_label = input_label
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.input_label, test_size=0.1, stratify=self.input_label
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
        self.train_acc = Accuracy(num_classes=config['num_class'], task='multiclass')
        self.val_acc = Accuracy(num_classes=config['num_class'], task='multiclass')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y.long())
        print(loss)
        self.log('train_loss', loss)
        Acc_train = self.train_acc(torch.argmax(y_hat, dim=1), y.long())
        print(Acc_train)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.val_acc(torch.argmax(y_hat, dim=1), y.long())
        return y_hat, y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

def main(args):
    config_file = args.config_file
    config = yaml.safe_load(open(config_file))

    input_data = load_data()
    input_label = load_label()

    datamodule = SCU_DataModule(input_data, input_label, config["batch_size"])
    model = SCU_Model(config)

    trainer = pl.Trainer(max_epochs=config["num_epochs"], logger=True)

    # Check if CUDA is available and specify the number of GPUs if it is
    # if torch.cuda.is_available():
    #     trainer.gpus = 1
    # else:
    trainer.gpus = None

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