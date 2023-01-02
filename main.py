import torch
from torch.optim import SGD, AdamW
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import numpy as np
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import random
import sys
sys.path.append("./")

from AL.FPOR import FPOR
from AL.FROP import FROP
from AL.FPOR_admm import FPOR_admm
from dataset.sythetic import generate_data
from dataset.UCI import get_data
from models.MLP import MLP




if __name__ == "__main__":
    
    # Precision: 0.811951      Recall 0.840000
    random_seed = 2
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


    device = torch.device("cuda")
    dimension = 2
    # X_tensor, y_tenosr, X, y = generate_data(dimension=dimension, device=device)
    # print(X_tensor.shape, y_tenosr.shape)
    # asdf
    X_tensor, y_tenosr, X, y = get_data(name='diabetic', device=device)
    # print(X_tensor.shape, y_tenosr.shape)
    
    # trainer = FPOR(X = X_tensor, y = y_tenosr)
    
    model = MLP(input_dim=X_tensor.shape[1], hidden_dim=100, num_layers=10)
    
    model.train()
    trainer = FPOR(X = X_tensor, y = y_tenosr, device=device, model=model)
    model = trainer.fit()
    



    ## visualize decision boundry for 2D simulation    
    # positive_idx = np.where(y==1)
    # negative_idx = np.where(y!=1)
    # model.eval()
    # a = -model[0].weight.data[0, 0]/model[0].weight.data[0, 1]
    # x_range = 5
    # plt.plot(np.linspace(-x_range, x_range), np.linspace(-x_range, x_range)*a.item()-model[0].bias.data.item()/model[0].weight.data[0, 1].item())
    # plt.scatter(X[positive_idx, 0], X[positive_idx, 1], color='green', label='positive')
    # plt.scatter(X[negative_idx, 0], X[negative_idx, 1], color='red', label='negative')
    # plt.ylim(-5, 5)
    # plt.xlim(-3, 3)
    # plt.savefig("data.png")
    # print(model[0].weight, model[0].bias)
    
    # ## evaluate based on value
    # prediction = (X@model[0].weight.data.detach().cpu().numpy().T + model[0].bias.data.item() > 0).astype(int)
    # TP = int(prediction.T@(y==1).astype(int))
    # print("Precision: {:3f} \t Recall {:3f}".format(1.0*TP/np.sum(prediction), 1.0*TP/np.sum(y==1)))
    
    