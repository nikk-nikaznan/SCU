import torch
import torch.nn as nn

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import argparse

from SCU import SCU
from utils import get_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.00001, help='adam: learning rate')
parser.add_argument('--dropout_level', type=float, default=0.5, help='dropout level')
parser.add_argument('--w_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=10, help='batch size')
parser.add_argument('--seed_n', type=int, default=10, help='seed number')
opt = parser.parse_args()

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

random.seed(opt.seed_n)
np.random.seed(opt.seed_n)
torch.manual_seed(opt.seed_n)
torch.cuda.manual_seed(opt.seed_n)

num_classes = 4

def plot_error_matrix(cm, classes, cmap=plt.cm.Blues):
   """ Plot the error matrix for the neural network models """

   from sklearn.metrics import confusion_matrix
   import itertools

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   #plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   print(cm)

   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.tight_layout()

def train_SCU(X_train, y_train):

    # convert NumPy Array to Torch Tensor
    train_input = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train)

    # create the data loader for the training set
    trainset = torch.utils.data.TensorDataset(train_input, train_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    
    cnn = SCU(opt, num_classes).to(device)
    cnn.train()
    # Loss and Optimizer
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=opt.lr, weight_decay=opt.w_decay)

    # loop through the required number of epochs
    for epoch in range(opt.n_epochs):

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
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

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
    
def CCM(cnf_labels, cnf_predictions):

    class_names = ["10", "12", "15", "30"] 
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(cnf_labels, cnf_predictions)
    np.set_printoptions(precision=2)

    # Normalise
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix = cnf_matrix.round(4)
    matplotlib.rcParams.update({'font.size': 16})

    # Plot normalized confusion matrix
    plt.figure()
    plot_error_matrix(cnf_matrix, classes=class_names)
    plt.tight_layout()
    filename = "S01_SCU.pdf"
    plt.savefig(filename, format='PDF', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # data loading
    EEGdata = np.load('SampleData_S01_4class.npy')
    EEGlabel = np.load('SampleData_S01_4class_labels.npy')
    EEGdata = EEGdata.swapaxes(1, 2)
    
    # split the data  training and testing
    X_train, X_test, y_train, y_test = train_test_split(EEGdata, EEGlabel, test_size=0.2)

    # training
    cnn = train_SCU(X_train, y_train)

    # testing
    test_cumulative_accuracy, ntest, cnf_labels, cnf_predictions = test_SCU(cnn, X_test, y_test)

    print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy/ntest*100)))

    # compute and plot confusion matrix
    CCM(cnf_labels, cnf_predictions)
