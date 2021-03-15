import torch.nn as nn
import torch.nn.functional as F

class ES_clf(nn.Module):

    def __init__(self, k = 80):
        super().__init__()

        self.fc1 = nn.Linear(k, k)
        self.b1 = nn.BatchNorm1d(k)
        self.fc2 = nn.Linear(k, int(k/2))
        self.b2 = nn.BatchNorm1d(int(k/2))
        self.fc3 = nn.Linear(int(k/2), 16)
        self.b3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16,10)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = F.relu(self.fc3(x))
        x = self.b3(x)
        x = F.relu(self.fc4(x))

        return x