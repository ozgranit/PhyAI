import torch.nn as nn
import torch.nn.functional as F

# model constants are defined here, because multiple files import them
NUM_ACTIONS = 39*39 # choose every pair you want
INPUT_SIZE = 39*39   # 20 leafs + 19 internal nodes - rooted tree
NUM_NODES = 39   # 20 leafs + 19 internal nodes - rooted tree


class DQN(nn.Module):
    # model#1
    def __init__(self, input_size, num_actions):
        #   Initialize a learning network for testing algorithm
        #   in_features: number of features of input.
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
