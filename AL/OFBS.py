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
        all_s = self.adjust_s(self.s)
        eps = 1e-9
        return -all_s.T@(all_y==1).double()/(all_s.T@(all_y==0).double()+torch.sum((all_y==1).double())*self.beta**2) \
                - torch.mean(all_s.T*torch.log2(all_s+eps))
        # return (all_s.T@(all_y==0).double()+torch.sum((all_y==1).double())*self.beta**2)/all_s.T@(all_y==1).double()


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
        
        delta_2 = 0.001
        return torch.cat([torch.log(eqs_n/delta_2 + 1), torch.log(eqs_p/delta_2 + 1)])

    def update_langrangian_multiplier(self):
        """update the lagrangian multipler
        """
        constrain_output = self.constrain()
        self.ls += self.rho*constrain_output
        self.ls = torch.maximum(self.ls, torch.tensor(0))
        self.rho *= self.delta