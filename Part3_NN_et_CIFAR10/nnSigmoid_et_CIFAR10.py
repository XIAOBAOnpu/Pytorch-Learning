import torch
import torchvision.datasets
from torch import nn
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
        self.relu_1 = nn.ReLU(inplace = False) # inplace? -> wether to use the new value to replace the old input
        self.sigmoid_1 = nn.Sigmoid()

    def forward(self, input):
        output_1 = self.relu_1(input)
        output_2 = self.sigmoid_1(output_1)
        return output_2

neuralNet = module()
writer = SummaryWriter("NN_nonAct_logs")
steps = 0

for data in dataloader:
    imgs, targets = data
    output = neuralNet(imgs)
    print("-> Input Shape:  " + str(imgs.shape))
    print("   Output Shape: " + str(output.shape))

    writer.add_images("input", imgs, steps)
    writer.add_images("output", output, steps)
    steps = steps + 1
