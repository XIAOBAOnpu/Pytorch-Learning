import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
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

writer = SummaryWriter("NN_maxpool_logs")
steps = 0

for data in dataloader:
    imgs, targets = data
    output = neuralNet(imgs)
    print("-> Input Shape:  " + str(imgs.shape))
    print("   Output Shape: " + str(output.shape))

    writer.add_images("input", imgs, steps)
    writer.add_images("output", output, steps)
    steps = steps + 1