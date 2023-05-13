import torch
import torch.nn as nn

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
        s = self.adjust_s(self.s[idx])
        fx = self.softmax(self.model(X))[:, 1].view(-1, 1)
        # return -all_s.T@(all_y==1).double()/torch.sum(all_s) - 0.1*torch.norm(fx *(1-fx))/idx.shape[0]
        
        n_pos = torch.sum(all_y==1)
        n_negs = torch.sum(all_y==0)
        weights = torch.tensor([n_pos/(n_pos+n_negs), n_negs/(n_negs+n_pos)]).to(self.device)
        weights = weights/(n_pos/(n_pos+n_negs))
        reweights = torch.ones(X.shape[0], 1).to(self.device)
        reweights[y==1] = weights[1]
        return -s.T@(y==1).double()/torch.sum(all_s) + self.args.reg*torch.norm(reweights * fx *(1-fx))/idx.shape[0]
        

    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    ## another option C(x) <= 0 to log(e(x)+1)
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        all_y = self.trainloader.targets.to(self.device)
        s = self.adjust_s(self.s[idx])
        fx = self.softmax(self.model(X))[:, 1].view(-1, 1)

        ret_val = torch.zeros_like(s).to(self.device)

        ineq = torch.maximum(torch.tensor(0), \
            self.alpha - s.T@(y==1).double() / torch.sum(all_y==1) 
        )

        pos_idx = (y==1).flatten()
        eqs_p = torch.maximum(torch.tensor(0), \
            torch.maximum(s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-s[pos_idx], fx[pos_idx]-self.t)
        )
        neg_idx = (y==0).flatten()
        eqs_n = torch.maximum(torch.tensor(0), \
            -torch.maximum(s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-s[neg_idx], fx[neg_idx]-self.t)
        )
        
        # delta = 1
        # delta_2 = 1
        # return torch.cat([torch.log(ineq/delta + 1).view(1, 1), torch.log(eqs_n/delta_2 + 1), torch.log(eqs_p/delta_2 + 1)])
        ret_val[pos_idx] = eqs_p
        ret_val[neg_idx] = eqs_n
        ret_val = torch.concat([ret_val, ineq.view(1, 1)])
        
        return ret_val
    
    
    def solve_sub_problem(self): 
        """solve the sub problem (stochastic)
        """
        L = 0
        ret_val = 0
        for idx, X, y in self.trainloader:
            X, y = X.to(self.device), y.to(self.device)
            fx = self.softmax(self.model(X))[:, 1].view(-1, 1)
            self.active_set = {"X": X, "y": y, "s": self.s[idx], "idx": idx, "fx": fx}
            tmp_obj, _ = self.AL_func_helper()
            L = tmp_obj
            L.backward()
            with torch.no_grad():
                ret_val += L.item()
        
        all_y = self.trainloader.targets.to(self.device)
        all_s = self.adjust_s(self.s)
        ineq = torch.maximum(torch.tensor(0), \
            self.alpha - all_s.T@(all_y==1).double() / torch.sum(all_y==1) 
        )
        L_ineq = self.ls[-1] * ineq\
                + (self.rho/2)* torch.sum(ineq**2)

        L_ineq.backward()
        self.optim.step()
        self.optim.zero_grad()

        # print(f"solving sub problem: loss {L.item()}")
        with torch.no_grad():
            ret_val += L_ineq.item()
            
        return ret_val
    