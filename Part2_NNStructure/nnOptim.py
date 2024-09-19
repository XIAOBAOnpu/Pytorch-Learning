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
    batch_size = 4,     # load 4 samples per batch
    shuffle = True,     # shuffle at every batch
    num_workers = 0,    # no multi-threads
    drop_last = False   # do not drop last batch even if #samples in last batch < batch_size
)

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
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(neuralNetCIFAR10.parameters(), lr = 0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        output = neuralNetCIFAR10(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss.item()  # convert loss from tensor to scalar

    print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}")