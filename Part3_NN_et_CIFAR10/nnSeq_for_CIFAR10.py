import torch
from torch import nn

class seqModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seqModel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, input):
        output = self.seqModel(input)
        return output

neuralNetCIFAR10 = seqModule()
print(neuralNetCIFAR10)