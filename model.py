import torch 
from torch.nn import functional as F
import torch.nn as nn

class SimpleMNISTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_1 = nn.Linear(4*4*64, num_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_1(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.max_pool_2d = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.max_pool_2d(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output
    