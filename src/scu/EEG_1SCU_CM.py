"""Code to run the training and test of SCU model."""

import argparse
from typing import Any, Dict, Tuple

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torchmetrics import Accuracy

from scu.datamodule import SCUDataModule
from scu.model import SCU
from scu.utils import compute_confusion_matrix, load_data, load_label


class SCUmodel(pl.LightningModule):
    """
    LightningModule for training and evaluating the SCU (Subject Classification Unit) model.

    This class defines the structure of the SCU model, including its architecture, loss function,
    training and testing steps, and optimization process.

    Attributes:
        config (Dict[str, Any]): A dictionary containing configuration parameters for the SCU model.
        model (SCU): An instance of the SCU model architecture.
        criterion (nn.Module): The loss function used for training the model.
        test_acc (Accuracy): Accuracy metric for testing data.
        test_step_outputs (List[Dict[str, torch.Tensor]]): A list to store test step outputs,
            including test loss and accuracy.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Performs forward pass through the model.
        training_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            Defines the training step logic.
        test_step(batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
            Defines the test step logic.
        on_test_epoch_end() -> None:
            Calculates and logs test loss and accuracy at the end of each testing epoch.
        configure_optimizers() -> torch.optim.Optimizer:
            Configures the optimizer used for training the model.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SCU_Model with the provided configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters for the SCU model.
        """
        super().__init__()
        self.config = config
        self.model = SCU(self.config).float()
        self.criterion = nn.CrossEntropyLoss()
        self.test_acc = Accuracy(num_classes=config["num_class"], task="multiclass")
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the SCU model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output tensor from the SCU model.
        """
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Define the training step logic.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss tensor.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y.long())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Define the test step logic.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing test loss, test accuracy, and other outputs.
        """
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.criterion(y_hat, y.long())
        test_acc = self.test_acc(torch.argmax(y_hat, dim=1), y.long())
        self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", test_acc, prog_bar=True, on_epoch=True)
        self.test_step_outputs.append({"test_loss": test_loss, "test_acc": test_acc, "y": y, "y_hat": y_hat})
        return {"test_loss": test_loss, "test_acc": test_acc, "y": y, "y_hat": y_hat}

    def on_test_epoch_end(self) -> None:
        """Calculate and logs test loss and accuracy at the end of each testing epoch."""
        if self.test_step_outputs:
            # Extract true labels and predicted labels for CCM function
            cnf_labels = np.concatenate([x["y"].cpu().numpy() for x in self.test_step_outputs])
            cnf_raw_scores = np.concatenate([x["y_hat"].cpu().numpy() for x in self.test_step_outputs])

            # Apply softmax to raw scores to obtain probabilities
            cnf_probs = torch.softmax(torch.tensor(cnf_raw_scores), dim=1)

            # Get predicted labels by selecting the class with the highest probability
            cnf_predictions = np.argmax(cnf_probs, axis=1)

            # Call your CCM function to plot the confusion matrix
            compute_confusion_matrix(cnf_labels, cnf_predictions)

        # Clear the test step outputs after each epoch
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure the optimizer used for training the model.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer


def main(args):
    """
    Execute the main training and testing loop for the SCU model.

    This function loads the configuration, data, and labels, initializes the data module and model,
    and then uses a PyTorch Lightning Trainer to fit and test the model.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration and runtime parameters.

            args.config_file (str): Path to the configuration YAML file.
            args.seed_n (int): Seed number for reproducibility.
            args.accelerator (str): Type of accelerator to use (e.g., 'cpu', 'gpu').

    """
    config_file = args.config_file
    config = yaml.safe_load(open(config_file))

    input_data = load_data()
    input_label = load_label()

    datamodule = SCUDataModule(input_data, input_label, config["batch_size"], args.seed_n)
    model = SCUmodel(config)

    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        logger=True,
        log_every_n_steps=1,
        accelerator=args.accelerator,
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
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Accelerator to train lightning. By default CPU, (other option : GPU)",
    )
    args = parser.parse_args()
    main(args)
