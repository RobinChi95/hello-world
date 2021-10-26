import glob
import random
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch
from torchvision.utils import save_image, make_grid

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDatasetPractical(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(root + "/*.*"))
        self.image_arr = os.listdir(root)


    def __getitem__(self, index):

        image_A = Image.open(self.files_A[index % len(self.files_A)])
        single_image_name = self.image_arr[index]
        cuda = torch.cuda.is_available()
        print(image_A)
        print(single_image_name)
        item_A = self.transform(image_A)
        # item_A = item_A.transpose(0, 1),
        if cuda:
            item_A = item_A.cuda()
        # return {"A": item_A}
        return {"A": item_A,"name" :single_image_name}

    def __len__(self):
        return len(self.files_A)

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A_ALL" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B_ALL" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        cuda = torch.cuda.is_available()

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        # print(self.files_A[index % len(self.files_A)])
        # print(self.files_B[index % len(self.files_B)])

        item_A = self.transform(image_A)
        # item_A = item_A.transpose(0, 1),
        item_B = self.transform(image_B)
        # item_B = item_B.transpose(0, 1),
        if cuda:
            item_A = item_A.cuda()
            item_B = item_B.cuda()
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

# transforms_ = [
#     transforms.Resize(int(256 * 1.12), Image.BICUBIC),
#     transforms.RandomCrop((256, 256)),
#     transforms.ToTensor(),
#     # transforms.Normalize(0.5, 0.5),
# ]


