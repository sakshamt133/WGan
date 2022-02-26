from torch.utils.data import Dataset
import os
import cv2 as cv
import utils


class Potato(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = os.listdir(path)
        self.transform = transform

    def __getitem__(self, item):
        img_name = self.files[item]
        img_path = os.path.join(self.path, img_name)

        img = cv.imread(img_path)

        if self.transform:
            img = cv.resize(img, (utils.img_size, utils.img_size))
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)
