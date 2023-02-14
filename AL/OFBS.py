import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys

from AL.AL_base import AL_base
from AL.FPOR import FPOR


class OFBS(FPOR):
    def __init__(self, trainloader, valloader, model, device, args):
        super().__init__(trainloader, valloader, model, device, args)
        self.beta = 1
        
    
    def objective(self):
        all_y = self.trainloader.targets.to(self.device)
        all_s = self.s
        # return -all_s.T@(all_y==1).double()/(all_s.T@(all_y==0).double()+torch.sum((all_y==1).double())*self.beta**2)
        return (all_s.T@(all_y==0).double()+torch.sum((all_y==1).double())*self.beta**2)/all_s.T@(all_y==1).double()

    def AL_func(self):
        """defines the augmented lagrangian function based on the objective function and constrains

        Returns:
            augmented lagrangian function
        """
        X = self.active_set['X']
        X = X.to(self.device)
        # return self.objective() + self.ls.T@self.constrain() \
        #         + (self.rho/2)* torch.sum(self.constrain()**2)
        ## exponential form
        c = 1
        return self.objective() + self.ls.T@self.constrain() \
                + (self.rho/c)* torch.sum(torch.exp(c*self.constrain())-1)

    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        s = self.s[idx]
        fx = self.model(X)
        
        pos_idx = (y==1).flatten()
        eqs_p = torch.maximum(torch.tensor(0), \
            torch.maximum(s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-s[pos_idx], fx[pos_idx]-self.t)
        )
        neg_idx = (y==0).flatten()
        eqs_n = torch.maximum(torch.tensor(0), \
            -torch.maximum(s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-s[neg_idx], fx[neg_idx]-self.t)
        )
        return torch.cat([torch.mean(torch.abs(eqs_n)).view(1, 1), torch.mean(torch.abs(eqs_p)).view(1, 1)], dim=0)
    
    def update_langrangian_multiplier(self):
        """update the lagrangian multipler
        """
        constrain_output = self.constrain()
        self.ls += self.rho*constrain_output
        self.ls = torch.maximum(self.ls, torch.tensor(0))
        self.rho *= self.delta