import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset


def opencvLoad(imgPath,resizeH,resizeW):
    image = cv2.imread(imgPath)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))
    image = torch.from_numpy(image)
    return image


class LoadDataset(Dataset):
    def __init__(self, data_path):
        data_list = glob.glob(data_path + "*.jpg")
        imgs = []
        for data in data_list:
            label_name = data.split("\\")[-1].split(".")[0]
            image = data
            if label_name == "dog":
                label = 0
                imgs.append((image, label))
            elif label_name == "cat":
                label = 1
                imgs.append((image, label))
        self.imgs = imgs

    def __getitem__(self, item):
        image, label = self.imgs[item]
        img = opencvLoad(image, 227, 227)
        return img, label

    def __len__(self):
        return len(self.imgs)
