import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys

from AL.AL_base import AL_base
from AL.FPOR import FPOR


## exponetial method of multiplier


class FROP(FPOR):
    
    def objective(self):
        X, idx = self.active_set['X'].to(self.device), self.active_set['idx']
        y = self.active_set['y']
        all_y = self.trainloader.targets.to(self.device)
        all_s = self.adjust_s(self.s)
        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        # return -all_s.T@(all_y==1).double()/torch.sum(all_s) - 0.1*torch.norm(fx *(1-fx))/idx.shape[0]
        
        n_pos = torch.sum(all_y==1)
        n_negs = torch.sum(all_y==0)
        weights = torch.tensor([n_pos/(n_pos+n_negs), n_negs/(n_negs+n_pos)]).to(self.device)
        weights = weights/(n_pos/(n_pos+n_negs))
        reweights = torch.ones(X.shape[0], 1).to(self.device)
        reweights[y==1] = weights[1]
        return -all_s.T@(all_y==1).double()/torch.sum(all_s) - 0.1*torch.norm(reweights * fx *(1-fx))/idx.shape[0]
        

    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    ## another option C(x) <= 0 to log(e(x)+1)
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        all_y = self.trainloader.targets.to(self.device)
        s = self.adjust_s(self.s[idx])
        all_s = self.adjust_s(self.s)
        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)

        ineq = torch.maximum(torch.tensor(0), \
            self.alpha - all_s.T@(all_y==1).double() / torch.sum(all_y==1) 
        )

        pos_idx = (y==1).flatten()
        eqs_p = torch.maximum(torch.tensor(0), \
            torch.maximum(s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-s[pos_idx], fx[pos_idx]-self.t)
        )
        neg_idx = (y==0).flatten()
        eqs_n = torch.maximum(torch.tensor(0), \
            -torch.maximum(s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-s[neg_idx], fx[neg_idx]-self.t)
        )
        
        delta = 1
        delta_2 = 1
        return torch.cat([torch.log(ineq/delta + 1).view(1, 1), torch.log(eqs_n/delta_2 + 1), torch.log(eqs_p/delta_2 + 1)])
    