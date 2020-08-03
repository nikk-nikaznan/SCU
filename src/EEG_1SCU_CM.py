import torch
import torch.nn as nn

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path
import argparse


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

cnn = CNN().to(device)
cnn.train()
# Loss and Optimizer
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=opt.lr, weight_decay=opt.w_decay)

correct = 0
total = 0

# kfold validation with random shuffle
X_train, X_test, y_train, y_test = train_test_split(EEGdata, EEGlabel, test_size=0.2)

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

cnn.eval()
test_cumulative_accuracy = 0
label_list=[]
prediction_list=[]
for i, data in enumerate(testloader, 0):
    # format the data from the dataloader
    test_inputs, test_labels = data
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    test_inputs = test_inputs.float()    

    test_outputs = cnn(test_inputs)
    _, test_predicted = torch.max(test_outputs, 1)    
    
    test_acc = get_accuracy(test_labels,test_predicted)
    test_cumulative_accuracy += test_acc
    
    test_labels = test_labels.cpu()
    test_predicted = test_predicted.cpu()
    label_list.append(test_labels.data.numpy())
    prediction_list.append(test_predicted.data.numpy())

print("Test Accuracy: %2.1f" % ((test_cumulative_accuracy/len(testloader)*100)))

label_list = np.array(label_list)
cnf_labels = np.concatenate(label_list)
prediction_list = np.array(prediction_list)
cnf_predictions = np.concatenate(prediction_list)

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
