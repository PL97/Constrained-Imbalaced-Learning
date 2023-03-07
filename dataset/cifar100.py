import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision
import pickle
import os
from dataset.Fastloader import FastTensorDataLoader 
from torchvision import transforms 
from multiprocessing import Pool, cpu_count

def apply_treansform_train(x):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = {"train": transforms.Compose([
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
    return transform['train'](x)

def apply_treansform_val(x):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = {"train": transforms.Compose([
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
    return transform['val'](x)

class imageFastTensorDataLoader(FastTensorDataLoader):
    def __init__(self, *tensors, batch_size=32, shuffle=False, mode='val'):
        super().__init__(*tensors, batch_size=batch_size, shuffle=shuffle)
        self.mode = mode
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
    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = list(t[self.i:self.i+self.batch_size] for t in self.tensors)
        pool = Pool(cpu_count() - 1)  
        if self.mode == "train":
            test1 = pool.map(apply_treansform_train, batch[-2])
        else:
            test1 = pool.map(apply_treansform_val, batch[-2])
        batch[0] = torch.stack(test1)
        self.i += self.batch_size
        return batch
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self


def load_cifar100_train():
    _ = torchvision.datasets.CIFAR100(root='../data/', train=True, download=True)
    path_to_DB = "../data/cifar-100-python/"

    ## fetch labels
    setname = 'meta'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        labelnames = pickle.load(obj, encoding='bytes')
        labelnames = labelnames[b'fine_label_names']
    for i in range(len(labelnames)):
        labelnames[i] = labelnames[i].decode("utf-8") 

    ## fetch images
    setname = 'train'
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32)).astype(np.float32)
    
    imgList = imgList.astype(np.float32)
    
    labelList = DATA[b'fine_labels']
    coause_labelList = DATA[b'coarse_labels']
    return imgList, labelList, coause_labelList, labelnames

def load_cifar100_test():
    setname = 'test'
    path_to_DB = "../data/cifar-100-python/"
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32)).astype(np.float32)
    labelList = DATA[b'fine_labels']
    coause_labelList = DATA[b'coarse_labels']

    imgList = imgList.astype(np.float32)
    return imgList, labelList, coause_labelList


def get_data(batch_size=10, random_seed=1997, binary_pos=0, with_idx=True):
    _ = torchvision.datasets.CIFAR100(root='../data/', train=True, download=True)
    path_to_DB = "../data/cifar-10-batches-py/"
    
    imgList, labelList, coause_labelList, labelNames = load_cifar100_train()
    labelList = np.asarray([0 if l != binary_pos else 1 for l in labelList])
    
    ## only consider classify class 1 from class 0
    # imgList, labelList = np.asarray(imgList), np.asarray(labelList)
    # binary_idx = np.where(np.asarray(labelList)<=1)[0]
    # imgList = imgList[binary_idx]
    # labelList = labelList[binary_idx]
    
    ## split train data into train and val
    imgList_train, imgList_val, labelList_train, labelList_val = train_test_split(imgList, \
                                                      labelList, \
                                                      test_size=0.2, \
                                                      stratify=labelList, \
                                                      random_state=random_seed)
    
    imgList_test, labelList_test, coause_labelList = load_cifar100_test()
    labelList_test = np.asarray([0 if l != binary_pos else 1 for l in labelList_test])

    ## only consider classify class 1 from class 0
    # imgList_test, labelList_test = np.asarray(imgList_test), np.asarray(labelList_test)
    # binary_idx = np.where(np.asarray(labelList_test)<=1)[0]
    # imgList_test = imgList_test[binary_idx]
    # labelList_test = labelList_test[binary_idx]
    
    stats = {
        'feature_dim': (3, 32, 32), \
        'label_num': len(np.unique(labelList)), \
        'total_num': labelList.shape[0], \
        'train_num': labelList_train.shape[0],\
        'test_num': labelList_test.shape[0], \
        'val_num': labelList_val.shape[0], \
        'label_distribution': Counter(labelList)
    }

    X_train, X_val, X_test = np.asarray(imgList_train), np.asarray(imgList_val), np.asarray(imgList_test)
    y_train, y_val, y_test = np.asarray(labelList_train), np.asarray(labelList_val), np.asarray(labelList_test)
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)
    
    print(X_test.shape, y_test.shape, len(imgList_test))
    
    
    if with_idx:
        trainloader = imageFastTensorDataLoader(np.asarray(range(len(imgList_train))), X_train, y_train, batch_size=batch_size, shuffle=True, mode='train')
        valloader = imageFastTensorDataLoader(np.asarray(range(len(imgList_val))), X_val, y_val, batch_size=batch_size, shuffle=True)
        testloader = imageFastTensorDataLoader(np.asarray(range(len(imgList_test))), X_test, y_test, batch_size=batch_size, shuffle=True)
    else:
        trainloader = imageFastTensorDataLoader(X_train, y_train, batch_size=batch_size, shuffle=True, mode='train')
        valloader = imageFastTensorDataLoader(X_val, y_val, batch_size=batch_size, shuffle=True)
        testloader = imageFastTensorDataLoader(X_test, y_test, batch_size=batch_size, shuffle=True)
    
    
    return trainloader, valloader, testloader, stats


if __name__ == "__main__":
    train, val, test, stats = get_data(batch_size=10, random_seed=1997, binary_pos=0)
    print(stats)
    for x, y in train:
        print(x.shape, y.shape)
        exit("finished")