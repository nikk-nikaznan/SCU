import torch
import torch.nn as nn

import numpy as np
import yaml
import random

from sklearn.model_selection import train_test_split

from model import SCU
from utils import get_accuracy, CCM


device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Setting seeds for reproducibility
seed_n = 74
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)

def train_SCU(X_train, y_train, config):

    # convert NumPy Array to Torch Tensor
    train_input = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train)

    # create the data loader for the training set
    trainset = torch.utils.data.TensorDataset(train_input, train_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    
    cnn = SCU(config).to(device)
    cnn.train()
    # Loss and Optimizer
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=config['learning_rate'], weight_decay=config["wdecay"])

    # loop through the required number of epochs
    for epoch in range(config["num_epochs"]):

        # loop through batches
        cumulative_accuracy = 0
        for i, data in enumerate(trainloader, 0):
            # format the data from the dataloader
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(inputs)

            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate the accuracy over the training batch
            _, predicted = torch.max(outputs, 1)
            
            cumulative_accuracy += get_accuracy(labels, predicted)
    
    return cnn

def test_SCU(cnn, X_test, y_test):

     # convert NumPy Array to Torch Tensor
    test_input = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test)

    # create the data loader for the test set
    testset = torch.utils.data.TensorDataset(test_input, test_label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    cnn.eval()
    test_cumulative_accuracy = 0
    label_list = []
    prediction_list = []
    for i, data in enumerate(testloader, 0):
        # format the data from the dataloader
        test_inputs, test_labels = data
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_inputs = test_inputs.float()    

        test_outputs = cnn(test_inputs)
        _, test_predicted = torch.max(test_outputs, 1)    
        
        test_acc = get_accuracy(test_labels,test_predicted)
        test_cumulative_accuracy += test_acc

        label_list.append(test_labels.cpu().data.numpy())
        prediction_list.append(test_predicted.cpu().data.numpy())

    label_list = np.array(label_list)
    cnf_labels = np.concatenate(label_list)
    prediction_list = np.array(prediction_list)
    cnf_predictions = np.concatenate(prediction_list)

    return test_cumulative_accuracy, len(testloader), cnf_labels, cnf_predictions

if __name__ == "__main__":

    # config loading
    with open("config/scu.yaml", "r") as f:
        config = yaml.safe_load(f)

    # data loading
    EEGdata = np.load('data/SampleData_S01_4class.npy')
    EEGlabel = np.load('data/SampleData_S01_4class_labels.npy')
    EEGdata = EEGdata.swapaxes(1, 2)
    
    # split the data  training and testing
    X_train, X_test, y_train, y_test = train_test_split(EEGdata, EEGlabel, test_size=0.2, stratify=EEGlabel)

    # training
    cnn = train_SCU(X_train, y_train, config)

    # testing
    test_cumulative_accuracy, ntest, cnf_labels, cnf_predictions = test_SCU(cnn, X_test, y_test)

    print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy/ntest*100)))

    # compute and plot confusion matrix
    CCM(cnf_labels, cnf_predictions)
