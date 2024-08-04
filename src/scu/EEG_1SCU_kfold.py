"""Code to run the training and test of SCU model on kfold."""

import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

from scu.model import SCU
from scu.utils import get_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--dropout_level", type=float, default=0.5, help="dropout level")
parser.add_argument("--w_decay", type=float, default=0.001, help="weight decay")
parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--seed_n", type=int, default=74, help="seed number")
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

random.seed(opt.seed_n)
np.random.seed(opt.seed_n)
torch.manual_seed(opt.seed_n)
torch.cuda.manual_seed(opt.seed_n)

num_classes = 4


def train_scu(x_train: np.ndarray, y_train: np.ndarray) -> SCU:
    """
    Train the SCU model on the provided training data.

    Args:
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.

    Returns:
        SCU: The trained SCU model.
    """
    # Convert NumPy array to Torch tensor
    train_input = torch.from_numpy(x_train)
    train_label = torch.from_numpy(y_train)

    # Create the data loader for the training set
    trainset = torch.utils.data.TensorDataset(train_input, train_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    cnn = SCU(opt, num_classes).to(device)
    cnn.train()
    # Loss and Optimizer
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=opt.lr, weight_decay=opt.w_decay)

    # Loop through the required number of epochs
    for epoch in range(opt.n_epochs):
        # Loop through the batches
        cumulative_accuracy = 0
        for i, data in enumerate(trainloader, 0):
            # Format the data from the dataloader
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(inputs)

            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate the accuracy over the training batch
            _, predicted = torch.max(outputs, 1)
            cumulative_accuracy += get_accuracy(labels, predicted)

    return cnn


def test_scu(cnn: SCU, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, int]:
    """
    Test the SCU model on the provided test data.

    Args:
        cnn (SCU): The trained SCU model.
        x_test (np.ndarray): Test data.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[float, int]: Cumulative test accuracy and number of test samples.
    """
    # Convert NumPy array to Torch tensor
    test_input = torch.from_numpy(x_test)
    test_label = torch.from_numpy(y_test)

    # Create the data loader for the test set
    testset = torch.utils.data.TensorDataset(test_input, test_label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    cnn.eval()
    test_cumulative_accuracy = 0
    for i, data in enumerate(testloader, 0):
        # Format the data from the dataloader
        test_inputs, test_labels = data
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_inputs = test_inputs.float()

        test_outputs = cnn(test_inputs)
        _, test_predicted = torch.max(test_outputs, 1)

        test_acc = get_accuracy(test_labels, test_predicted)
        test_cumulative_accuracy += test_acc

    return test_cumulative_accuracy, len(testloader)


if __name__ == "__main__":
    """
    Main function to load data, perform k-fold validation, train and test the SCU model.
    """
    # Data loading
    EEGdata = np.load("SampleData_S01_4class.npy")
    EEGlabel = np.load("SampleData_S01_4class_labels.npy")
    EEGdata = EEGdata.swapaxes(1, 2)

    # K-fold validation with random shuffle
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    sss.get_n_splits(EEGdata, EEGlabel)

    test_mean_acc = []
    for train_idx, test_idx in sss.split(EEGdata, EEGlabel):
        X_train, y_train = EEGdata[train_idx], EEGlabel[train_idx]
        X_test, y_test = EEGdata[test_idx], EEGlabel[test_idx]

        # Training
        cnn = train_scu(X_train, y_train)

        # Testing
        test_cumulative_accuracy, ntest = test_scu(cnn, X_test, y_test)
        print("Test Accuracy: %2.1f" % (test_cumulative_accuracy / ntest * 100))
        test_mean_acc.append(test_cumulative_accuracy / ntest)

    test_mean_acc = np.asarray(test_mean_acc)
    print("Mean Test Accuracy: %f" % test_mean_acc.mean())
    print("Standard Deviation: %f" % np.std(test_mean_acc, dtype=np.float64))
