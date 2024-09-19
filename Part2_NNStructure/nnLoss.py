import torch
from torch import nn

input = torch.tensor([1, 2, 3], dtype = torch.float32)
target = torch.tensor([1, 2, 5], dtype = torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = nn.L1Loss()
result = loss(input, target)

lossMSE = nn.MSELoss()
resultMSE = lossMSE(input, target)

print(result)
print(resultMSE)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))    # batch to be 1 and 3 classes
loss_cross = nn.CrossEntropyLoss()
resultCEL = loss_cross(x, y)
print(resultCEL)