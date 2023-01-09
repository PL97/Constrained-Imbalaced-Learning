from cProfile import label
from distutils.command.install_egg_info import safe_name
import os, random, time, copy
from cv2 import sort
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image
import pickle
import skimage.transform 

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import random_split

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Subset

from PIL import Image
from kmeans_pytorch import kmeans
from collections import defaultdict
from collections import Counter

from utils.utils import one_hot_embedding, pseudo_labeling
from torch.utils.data.sampler import Sampler


import sys
sys.path.append("../")
sys.path.append("./")
from configs.config import init, finish, print_and_log
from datasets.base_dataset import Base_32_dataset, train_test_split


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector
        

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


class CIFAR10(Base_32_dataset):
    pass


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_train():
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

def load_cifar10_test():
    setname = 'test_batch'
    path_to_DB = "../data/cifar-10-batches-py/"
    with open(os.path.join(path_to_DB, setname), 'rb') as obj:
        DATA = pickle.load(obj, encoding='bytes')
    imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
    labelList = DATA[b'labels']
    return imgList, labelList


def get_cifar10_dataloader(ctx, val=False, testonly=False):
    imgList, labelList, labelNames = load_cifar10_train()

    if ctx['debug']:
        idx = list(range(len(labelList)))
        imgList = np.asarray(imgList)
        labelList = np.asarray(labelList)
        np.random.shuffle(idx)
        imgList = imgList[idx[:500]]
        labelList = labelList[idx[:500]]

    if ctx['dataset']['binary']:
        labelList = [0 if l != ctx['dataset']['idx'] else 1 for l in labelList]



    train_ctx = ctx['train_config']
    cifar10_train = CIFAR10(imageList=imgList, labelList=labelList, ctx=ctx,
        pseudo_only=(not ctx['model']['multi_head'] and ctx['dataset']['pseudolabel']['on']),
        set_name="train",onehot=(train_ctx['loss']=='Huber' or train_ctx['loss']=='Max'or train_ctx['loss']=="AP"))


    counter_dict = Counter(labelList)
    label_freq = [counter_dict[k] for k in range(len(set(labelList)))]
    cifar10_train.label_freq = label_freq



    imgList, labelList = load_cifar10_test()


    if ctx['dataset']['binary']:
        labelList = [0 if l != ctx['dataset']['idx'] else 1 for l in labelList]


    cifar10_test = CIFAR10(imageList=imgList, labelList=labelList, ctx=ctx,
        pseudo_only=(not ctx['model']['multi_head'] and ctx['dataset']['pseudolabel']['on']),
        set_name="val",onehot=(train_ctx['loss']=='Huber' or train_ctx['loss']=='Max'or train_ctx['loss']=="AP"))

    cifar10_test.label_freq = label_freq


    if val:
        trainloader, valloader = train_test_split(cifar10_train, ctx=ctx, ratio=0.1)
    else:
        trainloader = torch.utils.data.DataLoader(cifar10_train,
                                            batch_size=ctx['train_config']['bs'],
                                            shuffle=True,
                                            num_workers=32)
    
    testloader = torch.utils.data.DataLoader(cifar10_test,
                                            batch_size=ctx['train_config']['bs'],
                                            shuffle=False,
                                            num_workers=32)
    dls = {}
    dls['train'] = trainloader
    dls['val'] = valloader if val else None
    dls['test'] = testloader


    return dls




if __name__ == "__main__":
    ctx = init()
    get_cifar10_dataloader(ctx)