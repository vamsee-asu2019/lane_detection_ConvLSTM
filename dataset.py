from torch.utils.data import Dataset
from PIL import Image
import torch
import config
import torchvision.transforms as transforms
import numpy as np
from sklearn import preprocessing
import random

def readTxt(file_path,istrain):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split(';')
            #print(item)
            img_list.append(item)
    file_to_read.close()
    #print(img_list)
    l = len(img_list)
    print(l)
    if istrain:
      l = int(l*0.4)
      img_list = random.sample(img_list, l)
      #img_list =img_list[:l]
    return img_list

class RoadSequenceDataset(Dataset):

    def __init__(self,istrain,file_path, transforms):

        self.img_list = readTxt(file_path,istrain)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[4])
        label = Image.open(img_path_list[5])
        data = self.transforms(data)
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample

# Use this for Conv-LSTM versions of the networks
class RoadSequenceDatasetList(Dataset):

    def __init__(self,istrain, file_path, transforms):

        self.img_list = readTxt(file_path,istrain)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        for i in range(5):
            data.append(torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0))
        data = torch.cat(data, 0)
        label = Image.open(img_path_list[5])
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample


