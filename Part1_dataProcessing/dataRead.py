from PIL import Image
import os
from torch.utils.data import Dataset

class myData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "../hymenoptera_data/hymenoptera_data/train"     # ../ to go back to upper folder
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = myData(root_dir, ants_label_dir)
bees_dataset = myData(root_dir, bees_label_dir)