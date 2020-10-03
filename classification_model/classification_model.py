import torch.nn as nn
import torch.nn.functional as F

num_of_classes = 100

class LNN(nn.Module):
    # model#2
    def __init__(self, in_features=29, output=1):
        #   Initialize a learning network for testing algorithm
        #   in_features: number of features of input.
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.generator = nn.Linear(128, num_of_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.generator(x)

