import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 4],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
print(input.shape)
print(kernel.shape) # not satisfied the requirement of conv2d, therefore reshape

input_reshape = torch.reshape(input, (1, 1, 5, 5))
kernel_reshape = torch.reshape(kernel, (1, 1, 3, 3))
print(input_reshape.shape)
print(kernel_reshape.shape) # now satisfied

output_stride_1 = F.conv2d(
    input = input_reshape,
    weight = kernel_reshape,
    stride = 1
)
output_stride_2 = F.conv2d(
    input = input_reshape,
    weight = kernel_reshape,
    stride = 2
)
output_padding = F.conv2d(
    input = input_reshape,
    weight = kernel_reshape,
    stride = 1,
    padding = 1
)

print(output_stride_1)
print(output_stride_2)
print(output_padding)