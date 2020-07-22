import torch.nn as nn
import torch.nn.functional as F

class DE(nn.Module):
    def __init__(self):
        super(DE, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.mp = nn.MaxPool2d(3)

    def forward(self, x):
        out = self.mp(self.conv1(x))
        return F.relu(out)

de = DE()