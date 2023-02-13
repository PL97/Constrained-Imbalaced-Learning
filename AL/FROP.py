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


    def AL_func(self):
        """defines the augmented lagrangian function based on the objective function and constrains

        Returns:
            augmented lagrangian function
        """
        X = self.active_set['X']
        X = X.to(self.device)
        return self.objective() + self.ls.T@self.constrain() \
                + (self.rho/2)* torch.sum(self.constrain()**2)
        ## exponential form
        # c = 1
        return self.objective() + self.ls.T@self.constrain() \
                + (self.rho/c)* torch.sum(torch.exp(c*self.constrain())-1)
        # return self.objective() + self.ls.T@self.constrain() \
        #         + (self.rho/c)* torch.sum(torch.exp(c*self.constrain())-1) + (self.rho/c)* torch.sum(torch.exp(-c*self.constrain())-1)
    

    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    ## another option C(x) <= 0 to log(e(x)+1)
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        all_y = self.trainloader.targets.to(self.device)
        s = self.s[idx]
        all_s = self.s
        fx = self.model(X)
        # ineq = torch.maximum(torch.tensor(0), \
        #     # self.alpha * torch.sum(self.s) - self.s.T@(self.y==1).double()
        #     self.alpha - all_s.T@(all_y==1).double() / torch.sum(all_y==1) 
        #     )

        # c = 1
        # pos_idx = (y==1).flatten()
        # eqs_p = c*torch.maximum(torch.tensor(0), \
        #     torch.maximum(s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-s[pos_idx], fx[pos_idx]-self.t)
        # )
        # neg_idx = (y==0).flatten()
        # eqs_n = c*torch.maximum(torch.tensor(0), \
        #     -torch.maximum(s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-s[neg_idx], fx[neg_idx]-self.t)
        # )

        ineq = self.alpha - all_s.T@(all_y==1).double() / torch.sum(all_y==1) 

        c = 1
        pos_idx = (y==1).flatten()
        eqs_p = c*torch.maximum(s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-s[pos_idx], fx[pos_idx]-self.t)
        # eqs_p = torch.log(torch.exp(s[pos_idx]+fx[pos_idx]-1-self.t,)) - torch.log(torch.exp(fx[pos_idx]-self.t) + torch.exp(-s[pos_idx]))
        neg_idx = (y==0).flatten()
        eqs_n = -c*torch.maximum(s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-s[neg_idx], fx[neg_idx]-self.t)
        # eqs_n = -(torch.log(torch.exp(s[neg_idx]+fx[neg_idx]-1-self.t)) - torch.log(torch.exp(fx[neg_idx]-self.t) + torch.exp(-s[neg_idx])))


        return torch.cat([ineq.view(1, 1), (torch.mean(torch.abs(eqs_n))).view(1, 1), (torch.mean(torch.abs(eqs_p))).view(1, 1)], dim=0)
    