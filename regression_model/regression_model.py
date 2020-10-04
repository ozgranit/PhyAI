import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class LNN(nn.Module):
    # best model - model#1
    def __init__(self, in_features=29, output=1):
        #   Initialize a learning network for testing algorithm
        #   in_features: number of features of input.
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


"""

class LNN(nn.Module):
    # model#2
    def __init__(self, in_features=29, output=1):
        #   Initialize a learning network for testing algorithm
        #   in_features: number of features of input.
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


"""

class LNN(nn.Module):
    # model#3
    def __init__(self, in_features=29, output=1):
        #   Initialize a learning network for testing algorithm
        #   in_features: number of features of input.
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return self.fc5(x)



class LNN(nn.Module):
    # model#4
    def __init__(self, in_features=29, output=1):
        #   Initialize a learning network for testing algorithm
        #   in_features: number of features of input.
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class LNN(nn.Module):
	# simplest model
    def __init__(self, in_features, output):
        super(LNN, self).__init__()
        self.linear = nn.Linear(in_features, output, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out


class LNN(nn.Module):
	def __init__(self, in_features=29, output=1):
		super(LNN, self).__init__()
		self.fc4 = nn.Linear(in_features, 512)
		self.fc5 = nn.Linear(512, output)

	def forward(self, x):
		x = F.relu(self.fc4(x.view(x.size(0), -1)))
		return self.fc5(x)

"""