import torch.nn as nn
import torch.nn.functional as F

"""
class LNN(nn.Module):
    def __init__(self, in_features=40, output=1):
        #Initialize a learning network for testing algorithm
        #   in_features: number of features of input.
        # super(LNN, self).__init__()
        # self.fc1 = nn.Linear(in_features, 256)
        self.fc1 = nn.Linear(in_features, output)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, output)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # return self.fc4(x)
        return F.relu(self.fc1(x))

"""


class LNN(nn.Module):
    def __init__(self, in_features, output):
        super(LNN, self).__init__()
        self.linear = nn.Linear(in_features, output, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out