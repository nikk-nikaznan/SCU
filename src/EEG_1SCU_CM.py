from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl
from torchmetrics import Accuracy

import yaml
import argparse

from model import SCU
from utils import get_accuracy, CCM, load_data, load_label
from datamodule import SCU_DataModule
import logging

logging.basicConfig(level=logging.INFO)


class SCU_Model(pl.LightningModule):
    """
    LightningModule for training and evaluating the SCU (Subject Classification Unit) model.

    This class defines the structure of the SCU model, including its architecture, loss function,
    training and testing steps, and optimization process.

    Attributes:
        config (Dict[str, Any]): A dictionary containing configuration parameters for the SCU model.
        model (SCU): An instance of the SCU model architecture.
        criterion (nn.Module): The loss function used for training the model.
        train_loss: Loss metric for training data.
        test_step_outputs (List[Dict[str, torch.Tensor]]): A list to store test step outputs,
            including test loss and accuracy.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Performs forward pass through the model.
        training_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            Defines the training step logic.
        test_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
            Defines the test step logic.
        on_test_epoch_end() -> None:
            Calculates and logs test loss and accuracy at the end of each testing epoch.
        configure_optimizers() -> torch.optim.Optimizer:
            Configures the optimizer used for training the model.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the SCU_Model with the provided configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters for the SCU model.
        """
        super().__init__()
        self.config = config
        self.model = SCU(self.config).double()
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(num_classes=config["num_class"], task="multiclass")
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the SCU model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output tensor from the SCU model.
        """
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Defines the training step logic.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y.long())
        Acc_train = self.train_acc(torch.argmax(y_hat, dim=1), y.long())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """
        Defines the test step logic.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the batch.
        """
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.criterion(y_hat, y.long())
        test_acc = get_accuracy(y.cpu(), torch.argmax(y_hat, dim=1).cpu())
        self.test_step_outputs.append({"test_loss": test_loss, "test_acc": test_acc})

    def on_test_epoch_end(self) -> None:
        """
        Calculates and logs test loss and accuracy at the end of each testing epoch.
        """
        test_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs]).mean()
        self.log("test_loss", test_loss, on_epoch=True)

        mean_acc = torch.tensor([x["test_acc"] for x in self.test_step_outputs]).mean()
        self.log("test_acc", mean_acc, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configures the optimizer used for training the model.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer


def main(args):
    config_file = args.config_file
    config = yaml.safe_load(open(config_file))

    input_data = load_data()
    input_label = load_label()

    datamodule = SCU_DataModule(
        input_data, input_label, config["batch_size"], args.seed_n
    )
    model = SCU_Model(config)

    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        logger=True,
        log_every_n_steps=1,
        accelerator="cpu",
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/scu.yaml",
        help="location of YAML config to control training",
    )
    parser.add_argument(
        "--seed_n",
        type=int,
        default=74,
        help="seeds for reproducibility",
    )
    args = parser.parse_args()
    main(args)
