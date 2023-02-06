import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys

from AL.AL_base import AL_base
from AL.FPOR import FPOR


class FROP(FPOR):
    
    def objective(self):
        all_y = self.trainloader.targets.to(self.device)
        all_s = self.s
        return -all_s.T@(all_y==1).double()/torch.sum(all_s)


    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        all_y = self.trainloader.targets.to(self.device)
        s = self.s[idx]
        all_s = self.s
        fx = self.model(X)
        ineq = torch.maximum(torch.tensor(0), \
            # self.alpha * torch.sum(self.s) - self.s.T@(self.y==1).double()
            self.alpha - all_s.T@(all_y==1).double() / torch.sum(all_y==1) 
            )
        eqs = torch.maximum(s+fx-1-self.t, torch.tensor(0)) - torch.maximum(-s, fx-self.t)
        
        zero_constrain = torch.mean(torch.abs(self.s[idx]*(1-self.s[idx])))
        return torch.cat([ineq.view(1, 1), (torch.mean(torch.abs(eqs))+zero_constrain).view(1, 1)], dim=0)