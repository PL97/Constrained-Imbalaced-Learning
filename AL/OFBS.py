import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys

from AL.AL_base import AL_base
from AL.FPOR import FPOR


class OFBS(FPOR):
    def __init__(self, trainloader, valloader, testloader, model, device, args):
        super().__init__(trainloader, valloader, testloader, model, device, args)
        self.beta = 1
        
    def regularization(self):
        X, idx = self.active_set['X'].to(self.device), self.active_set['idx']
        y = self.active_set['y']
        all_y = self.trainloader.targets.to(self.device)
        n_pos = torch.sum(all_y==1)
        n_negs = torch.sum(all_y==0)
        weights = torch.tensor([n_pos/(n_pos+n_negs), n_negs/(n_negs+n_pos)]).to(self.device)
        weights = weights/(n_pos/(n_pos+n_negs))
        reweights = torch.ones(X.shape[0], 1).to(self.device)
        reweights[y==1] = weights[1]
        
        fx = self.active_set['fx'] if 'fx' in self.active_set else self.softmax(self.model(X))[:, 1].view(-1, 1)
        return self.args.reg*torch.norm(reweights* fx *(1-fx))/idx.shape[0]

    
    def objective(self, with_reg=True):
        X, idx = self.active_set['X'].to(self.device), self.active_set['idx']
        y = self.active_set['y']
        all_y = self.trainloader.targets.to(self.device)
        all_s = self.adjust_s(self.s)
        s = self.adjust_s(self.s[idx])

        if with_reg:
            reg = self.regularization()
            return -s.T@(y==1).double()/(all_s.T@(all_y==0).double()+torch.sum((all_y==1).double())*self.beta**2) + reg
        else:
            return -s.T@(y==1).double()/(all_s.T@(all_y==0).double()+torch.sum((all_y==1).double())*self.beta**2)


    def full_objective(self):
        all_y = self.trainloader.targets.to(self.device)
        all_s = self.adjust_s(self.s)
        return -all_s.T@(all_y==1).double()/(all_s.T@(all_y==0).double()+torch.sum((all_y==1).double())*self.beta**2)

    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        s = self.adjust_s(self.s[idx])
        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        
        ret_val = torch.zeros_like(s).to(self.device)
        
        pos_idx = (y==1).flatten()
        eqs_p = torch.maximum(torch.tensor(0), \
            torch.maximum(s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-s[pos_idx], fx[pos_idx]-self.t)
        )
        neg_idx = (y==0).flatten()
        eqs_n = torch.maximum(torch.tensor(0), \
            -torch.maximum(s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-s[neg_idx], fx[neg_idx]-self.t)
        )
        
        ret_val[pos_idx] = eqs_p
        ret_val[neg_idx] = eqs_n
        return ret_val
        
        
    def AL_func(self):
        """defines the augmented lagrangian function based on the objective function and constrains

        Returns:
            augmented lagrangian function
        """
        X, idx = self.active_set['X'], self.active_set['idx']
        ls = self.ls[idx]
        X = X.to(self.device)
        constraints = self.constrain()
        return self.objective() + ls.T@constraints\
                + (self.rho/2)* torch.sum(constraints**2)
                

                
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
            

    def solve_sub_problem(self): 
        """solve the sub problem (stochastic)
        """
        L = 0
        ret_val = 0
        for idx, X, y in self.trainloader:
            X, y = X.to(self.device), y.to(self.device)
            fx = self.softmax(self.model(X))[:, 1].view(-1, 1)
            self.active_set = {"X": X, "y": y, "s": self.s[idx], "idx": idx, "fx": fx}
            # L = self.AL_func()
            constraints = self.constrain()
            L = self.regularization() + self.ls[idx].T@constraints\
                + (self.rho/2)* torch.sum(constraints**2)
            L.backward()
            with torch.no_grad():
                ret_val += L.item()
        
        obj = self.full_objective()
        obj.backward()
        
        self.optim.step()
        self.optim.zero_grad()

        # print(f"solving sub problem: loss {L.item()}")
            
        return ret_val