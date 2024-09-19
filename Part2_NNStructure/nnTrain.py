import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

train_data = torchvision.datasets.CIFAR10(
    root="./dataset",  # will automatically create a dataset folder afterwards
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="./dataset",  # will automatically create a dataset folder afterwards
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(train_data_size)
print(test_data_size)

# use dataloader to load data
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# build NN
class nnModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.model(input)
        return output

# use this NN
nnTrain = nnModule()
if torch.cuda.is_available():
    nnTrain = nnTrain.cuda()
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
optim = torch.optim.SGD(nnTrain.parameters(), lr=0.01)

# train this NN
total_train_step = 0
total_test_step = 0
epoch = 50

# tensorboard
writer = SummaryWriter("nnTraining_logs")

for i in range(epoch):

    print("-------------- Round {} starts --------------".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = nnTrain(imgs)
        loss = loss_fn(outputs, targets)
        # optim using optimizer
        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("Training: {} -> Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(),total_train_step)

    # start testing
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = nnTrain(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("Total loss: {}".format(total_test_loss))
    print("Total accuracy: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_train_step + 1

writer.close()
