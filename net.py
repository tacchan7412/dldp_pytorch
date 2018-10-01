import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, U):
        super(Net, self).__init__()
        self.U = U
        self.fc1 = nn.Linear(60, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # flatten
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        # pca projection
        x = torch.mm(x, self.U)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
