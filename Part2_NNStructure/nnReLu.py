import torch
from torch import nn

input = torch.tensor([[1, -0.5],
                      [-1, 3  ]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)
print("-> Before ReLu: " + str(input))

class module(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_1 = nn.ReLU(inplace = False) # inplace? -> wether to use the new value to replace the old input

    def forward(self, input):
        output = self.relu_1(input)
        return output

neuralNet = module()
output = neuralNet(input)
print("    After ReLu: " + str(output))