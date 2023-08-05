import torch 
import torch.nn as nn
import numpy as np

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
    

if __name__ == "__main__":
    model = SimpleMNISTClassifier((3, 28, 28), 10)
    print(model)