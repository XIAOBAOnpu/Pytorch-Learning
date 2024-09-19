import torch
from torch import nn

# create a NN template (will not be exe before being used)
class nnTemplate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

# create the NN by using the template defined just now
neuralNet = nnTemplate()
x = torch.tensor(1.0)
output = neuralNet(x)
print(output)