import torch
from torch import nn
from torch.nn import Conv2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    root = "./dataset",  # will automatically create a dataset folder afterwards
    train = False,
    transform = torchvision.transforms.ToTensor(),  # need tensor type
    download = True
)

dataloader = DataLoader(
    dataset = dataset,
    batch_size = 64,     # load 4 samples per batch
    shuffle = True,     # shuffle at every batch
    num_workers = 0,    # no multi-threads
    drop_last = False   # do not drop last batch even if #samples in last batch < batch_size
)

class module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels = 3,    # num of input channels
            out_channels = 6,   # num of output channels
            kernel_size = 3,
            stride = 1,
            padding = 0,
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

neuralNet = module()
print(neuralNet)    # check the structure of the NN that created

writer = SummaryWriter("NN_conv_logs")
steps = 0

for data in dataloader:
    imgs, targets = data
    output = neuralNet(imgs)
    print("-> Input Shape:  " + str(imgs.shape))
    print("   Output Shape: " + str(output.shape))

    writer.add_images("input", imgs, steps)

    # have to reshape since 6 channel is too much, cannot be displayed -> reshape it into 3 channels
    output_show = torch.reshape(output, (-1, 3, 30, 30))    # dont know batch? -> set to -1
    print("   Reshape:      " + str(output_show.shape))
    writer.add_images("output", output_show, steps)
    steps = steps + 1

