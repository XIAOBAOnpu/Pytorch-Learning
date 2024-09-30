import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(
    root = "../dataset",  # will automatically create a dataset folder afterwards
    train = False,
    transform = torchvision.transforms.ToTensor(),  # transfer to tensor type
    download = True
)

test_loader = DataLoader(
    dataset = test_data,
    batch_size = 64,     # load 64 samples per batch
    shuffle = True,     # shuffle at every batch
    num_workers = 0,    # no multi-threads
    drop_last = False   # do not drop last batch even if #samples in last batch < batch_size
)

img, target = test_data[0]
print(img.shape)    # OUTPUT: torch.Size([3, 32, 32])
print(target)   # belongs to which class, e.g. ship, plane, cat.... OUTPUT: 3

writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("shuffleEpoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()  # cmd: tensorboard --logdir=D:\\pycharmProject\\pytorchProject\\pythonProject\\dataloader
