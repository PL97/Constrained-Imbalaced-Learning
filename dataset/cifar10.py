import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision
import pickle
import os
from torchvision import transforms


class Cifar10(Dataset):
    def __init__(self, X, y, mode="val"):
        """Cifar10 dataset instance

        Args:
            X (numpy array or tensor): features
            y (numpy array or tensor): labels
            mode (str): train or val
        """
        self.mode = mode
        X, y = np.asarray(X), np.asarray(y)
        if isinstance(X, torch.Tensor):
            self.X, self.y = X, y
        else:
            self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
        self.set_ret_idx(ret=True)
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
         
        self.transform = {"train": transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "val": transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])}

    def __len__(self):        
        return self.X.shape[0]
    
    def set_ret_idx(self, ret):
        self.ret_idx = ret
    
    def __getitem__(self, index):
        transformed_X = self.transform[self.mode](self.X[index])
        if self.ret_idx:
            return index, transformed_X, self.y[index]
        else:
            return transformed_X, self.y[index]   
        
def load_cifar10_test():
    """read and load cifar10 test data

    Returns:
        tuple: lists of images and labels
    """
    setname = 'test_batch'
    path_to_DB = "../data/cifar-10-batches-py/"
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
    labelList = DATA[b'labels']
    return imgList, labelList    
    
    
def load_cifar10_train():
    """read and load cifar10 train dataset

    Returns:
        tuple: lists of images, labels and labelnames
    """
    _ = torchvision.datasets.CIFAR10(root='../data/', train=True, download=True)
    path_to_DB = "../data/cifar-10-batches-py/"

    ## fetch labels
    setname = 'batches.meta'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        labelnames = pickle.load(obj, encoding='bytes')
        labelnames = labelnames[b'label_names']
    for i in range(len(labelnames)):
        labelnames[i] = labelnames[i].decode("utf-8") 

    ## fetch images
    setname = 'data_batch_'
    imgList = []
    labelList = []
    for i in range(1, 6):
        with open(os.path.join(path_to_DB, setname+str(i)), 'rb') as obj:
            DATA = pickle.load(obj, encoding='bytes')
        imgList.extend(DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32)))
        labelList.extend(DATA[b'labels'])
    return imgList, labelList, labelnames

def get_data(batch_size=10, random_seed=1997, binary_pos=0):
    _ = torchvision.datasets.CIFAR10(root='../data/', train=True, download=True)
    path_to_DB = "../data/cifar-10-batches-py/"
    
    imgList, labelList, labelNames = load_cifar10_train()
    labelList = np.asarray([0 if l != binary_pos else 1 for l in labelList])
    
    ## split train data into train and val
    imgList_train, imgList_val, labelList_train, labelList_val = train_test_split(imgList, \
                                                      labelList, \
                                                      test_size=0.2, \
                                                      stratify=labelList, \
                                                      random_state=random_seed)
    
    imgList_test, labelList_test = load_cifar10_test()
    labelList_test = np.asarray([0 if l != binary_pos else 1 for l in labelList_test])
    stats = {
        'feature_dim': (3, 32, 32), \
        'label_num': len(np.unique(labelList)), \
        'total_num': labelList.shape[0], \
        'train_num': labelList_train.shape[0],\
        'test_num': labelList_test.shape[0], \
        'val_num': labelList_val.shape[0], \
        'label_distribution': Counter(labelList)
    }

    trainloader = DataLoader(Cifar10(X=imgList_train, y=labelList_train), \
                            batch_size=batch_size, \
                            shuffle=True, \
                            num_workers=8)
    
    valloader = DataLoader(Cifar10(X=imgList_val, y=labelList_val), \
                            batch_size=batch_size, \
                            shuffle=False, \
                            num_workers=8) 
    
    testloader = DataLoader(Cifar10(X=imgList_test, y=labelList_test), \
                            batch_size=batch_size, \
                            shuffle=False, \
                            num_workers=8) 
    
    return trainloader, valloader, testloader, stats


if __name__ == "__main__":
    train, val, test, stats = get_data(batch_size=10, random_seed=1997, binary_pos=0)
    print(stats)
    for idx, x, y in train:
        print(x.shape, y.shape)
        exit("finished")
