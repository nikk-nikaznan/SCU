import torch
import torch.nn as nn

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.00001, help='adam: learning rate')
parser.add_argument('--dropout_level', type=float, default=0.5, help='dropout level')
parser.add_argument('--w_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--seed_n', type=int, default=74, help='seed number')
opt = parser.parse_args()

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(opt.seed_n)
torch.cuda.manual_seed(opt.seed_n)

num_classes = 4

# data loading

EEGdata = np.load('{Path.home()}/Data/Data_S01_4class.npy')
EEGlabel = np.load('{Path.home()}/Data/Data_S01_4class_labels.npy')
EEGdata = EEGdata.swapaxes(1, 2)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(opt.dropout_level))
        
        self.dense_layers = nn.Sequential(
            nn.Linear(5984, 600),
            nn.ReLU(),
            nn.Dropout(opt.dropout_level),
            nn.Linear(600, 60),
            nn.ReLU(),
            nn.Dropout(opt.dropout_level),
            nn.Linear(60, num_classes))

    def forward(self, x):

        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)     
        return out

def get_accuracy(actual, predicted):
    # actual: cuda longtensor variable
    # predicted: cuda longtensor variable
    assert(actual.size(0) == predicted.size(0))
    return float(actual.eq(predicted).sum()) / actual.size(0)

# kfold validation with random shuffle
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
sss.get_n_splits(EEGdata, EEGlabel)

test_meanAcc = []
for train_idx, test_idx in sss.split(EEGdata, EEGlabel):

    cnn = CNN().to(device)
    cnn.train()

    # Loss and Optimizer
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=opt.lr, weight_decay=opt.w_decay)

    X_train = EEGdata[train_idx]
    y_train = EEGlabel[train_idx]
    X_test = EEGdata[test_idx]
    y_test = EEGlabel[test_idx]

    # convert NumPy Array to Torch Tensor
    train_input = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train)
    test_input = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test)

    # create the data loader for the training set
    trainset = torch.utils.data.TensorDataset(train_input, train_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    # create the data loader for the test set
    testset = torch.utils.data.TensorDataset(test_input, test_label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # loop through the required number of epochs
    for epoch in range(opt.n_epochs):

        # loop through the batches yo!!!
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
        # print("Training Loss:", loss.item())
        # print("Training Accuracy: %2.1f" % ((cumulative_accuracy/len(trainloader)*100)))

    cnn.eval()
    test_cumulative_accuracy = 0
    for i, data in enumerate(testloader, 0):
        # format the data from the dataloader
        test_inputs, test_labels = data
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_inputs = test_inputs.float()    

        test_outputs = cnn(test_inputs)
        _, test_predicted = torch.max(test_outputs, 1)    
        test_acc = get_accuracy(test_labels,test_predicted)
        test_cumulative_accuracy += test_acc

    test_meanAcc.append(test_cumulative_accuracy/len(testloader))
    print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy/len(testloader)*100)))

test_meanAcc = np.asarray(test_meanAcc)
print("Mean Test Accuracy: %f" % test_meanAcc.mean())
print ("Standard Deviation: %f" % np.std(test_meanAcc, dtype=np.float64))
