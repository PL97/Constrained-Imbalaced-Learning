

from tkinter.tix import Meter
import torch.nn as nn
from torchvision.models import vgg16, resnet18
import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("./")
sys.path.append("../")
from models.shared import shared


class AlexNet(shared):
    def __init__(self, num_classes=10, grayscale=False, input_shape=(1, 1, 32, 32)):
        """AlexNet for image classification

        Args:
            num_classes (int, optional): number of output classes. Defaults to 10.
            grayscale (bool, optional): input grayscale image, set channel as 1. Defaults to False.
            input_shape (tuple, optional): example of input batch shape. Defaults to (1, 1, 32, 32).
        """
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192,kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        if grayscale:
            self.features = nn.Sequential(
                            nn.Conv2d(1, 3, 1),
                            self.features)

    
        self.init_clf(input_shape=input_shape, num_classes=num_classes)
        

if __name__ == "__main__":
    net = AlexNet(num_classes=2, grayscale=False, input_shape=(1, 3, 32, 32))
    print(net)
    test_input = torch.zeros((3, 3, 32, 32))
    print(net(test_input).shape)