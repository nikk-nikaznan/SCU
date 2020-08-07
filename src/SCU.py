import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, opt, num_classes):
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