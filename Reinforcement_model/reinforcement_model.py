import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    # model#1
    def __init__(self, input_size, num_actions):
        #   Initialize a learning network for testing algorithm
        #   in_features: number of features of input.
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
