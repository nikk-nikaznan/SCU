import torch
import torch.nn as nn

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


class SCU_Class:
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        print(self.config_file)
        self.input_data = load_data()
        self.input_label = load_label()
        self.load_config_yaml()

    def load_config_yaml(self) -> None:
        """Load a YAML file describing the training setup"""

        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def _load_model(self) -> None:
        """Load the EEG subject classification model"""

        # Build the subject classification model and initalise weights
        self.scu = SCU(self.config).to(device)
        self.scu.apply(weights_init)

    def _build_training_objects(self) -> None:
        """Create the training objects"""

        # Loss and Optimizer
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer= torch.optim.Adam(
            self.scu.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )

    def _train_SCU(self):
        # convert NumPy Array to Torch Tensor
        train_input = torch.from_numpy(self.X_train)
        train_label = torch.from_numpy(self.y_train)

        # create the data loader for the training set
        trainset = torch.utils.data.TensorDataset(train_input, train_label)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.config["batch_size"], shuffle=True, num_workers=4
        )

        self.scu.train()

        # loop through the required number of epochs
        for epoch in range(self.config["num_epochs"]):
            # loop through batches
            cumulative_accuracy = 0
            for i, data in enumerate(trainloader, 0):
                # format the data from the dataloader
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                outputs = self.scu(inputs)

                loss = self.ce_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # calculate the accuracy over the training batch
                _, predicted = torch.max(outputs, 1)

                cumulative_accuracy += get_accuracy(labels, predicted)

    def _test_SCU(self):
        # convert NumPy Array to Torch Tensor
        test_input = torch.from_numpy(self.X_test)
        test_label = torch.from_numpy(self.y_test)

        # create the data loader for the test set
        testset = torch.utils.data.TensorDataset(test_input, test_label)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=4
        )

        self.scu.eval()
        test_cumulative_accuracy = 0
        label_list = []
        prediction_list = []
        for i, data in enumerate(testloader, 0):
            # format the data from the dataloader
            test_inputs, test_labels = data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_inputs = test_inputs.float()

            test_outputs = self.scu(test_inputs)
            _, test_predicted = torch.max(test_outputs, 1)

            test_acc = get_accuracy(test_labels, test_predicted)
            test_cumulative_accuracy += test_acc

            label_list.append(test_labels.cpu().data.numpy())
            prediction_list.append(test_predicted.cpu().data.numpy())

        label_list = np.array(label_list)
        cnf_labels = np.concatenate(label_list)
        prediction_list = np.array(prediction_list)
        cnf_predictions = np.concatenate(prediction_list)

        return test_cumulative_accuracy, len(testloader), cnf_labels, cnf_predictions

    def data_prep(self) -> None:
        # split the data training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.input_label, test_size=0.2, stratify=self.input_label
        )

        self._load_model()
        self._build_training_objects()
        # training
        self._train_SCU()
        # testing
        test_cumulative_accuracy, ntest, cnf_labels, cnf_predictions = self._test_SCU()
        print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy / ntest * 100)))
        # compute and plot confusion matrix
        CCM(cnf_labels, cnf_predictions)


if __name__ == "__main__":
    # config loading
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/scu.yaml",
        help="location of YAML config to control training",
    )
    args = parser.parse_args()

    trainer = SCU_Class(config_file=args.config_file)
    trainer.data_prep()
