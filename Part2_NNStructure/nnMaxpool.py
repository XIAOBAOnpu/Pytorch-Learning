import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype = torch.float32)

input_reshape = torch.reshape(input, (-1, 1, 5, 5))

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(
            kernel_size = 3,
            ceil_mode = True    # if true, then even if the input is not 3*3 -> perform maxpool as well
        )

    def forward(self, input):
        output = self.maxpool(input)
        return output

neuralNet = model()
output = neuralNet(input_reshape)
print(output)