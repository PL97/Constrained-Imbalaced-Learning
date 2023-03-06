import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys

from AL.AL_base import AL_base
from AL.FPOR import FPOR


class OAP(FPOR):
    def __init__(self, trainloader, valloader, model, device, args):
        super().__init__(trainloader, valloader, model, device, args)
        
        self.u = torch.randn((args.datastats['train_num'], 1), requires_grad=True, \
                            dtype=torch.float64, device=self.device)
    
    def objective(self):
        X, idx = self.active_set['X'].to(self.device), self.active_set['idx']
        all_y = self.trainloader.targets.to(self.device)
        all_u = self.u[idx]
        X, idx = self.active_set['X'].to(self.device), self.active_set['idx']
        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        return -all_u@(all_y==1).double()/torch.sum(all_y) + 0.1*torch.norm(fx *(1-fx))/idx.shape[0]
        

    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        s = self.adjust_s(self.s[idx])
        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        
        pos_idx = (y==1).flatten()
        eqs_p = torch.maximum(torch.tensor(0), \
            torch.maximum(s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-s[pos_idx], fx[pos_idx]-self.t)
        )
        neg_idx = (y==0).flatten()
        eqs_n = torch.maximum(torch.tensor(0), \
            -torch.maximum(s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-s[neg_idx], fx[neg_idx]-self.t)
        )
        
        delta_2 = 0.01
        return torch.cat([torch.log(eqs_n/delta_2 + 1), torch.log(eqs_p/delta_2 + 1)])
    
    def AL_func(self):
        """defines the augmented lagrangian function based on the objective function and constrains

        Returns:
            augmented lagrangian function
        """
        X, idx = self.active_set['X'], self.active_set['idx']
        ls = self.ls[idx]
        X = X.to(self.device)
        return self.objective() + ls.T@self.constrain() \
                + (self.rho/2)* torch.sum(self.constrain()**2)
                
    def update_langrangian_multiplier(self):
        """update the lagrangian multipler
        """
        tmp_constrains = 0
        count_updates = 0
        for idx, X, y in self.trainloader:
            count_updates += 1
            X, y = X.to(self.device), y.to(self.device)
            self.active_set = {"X": X, "y": y, "s": self.s[idx], "idx": idx}
            constrain_output = self.constrain()
            self.ls[idx] += self.rho*constrain_output
            tmp_constrains += torch.norm(constrain_output).item()
        self.rho *= self.delta
        if tmp_constrains > self.pre_constrain:
            self.rho *= self.delta
            self.pre_constrain = tmp_constrains