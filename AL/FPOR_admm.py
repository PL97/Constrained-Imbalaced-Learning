import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
import numpy as np
import sys

from AL.AL_base import AL_base


class FPOR_admm(AL_base):
    @torch.no_grad()
    def __init__(self, X, y, model, device):
        self.X, self.y = X, y
        self.device = device
        ## general hyperparameters (fine tune for best performance)
        self.subprob_max_epoch=500
        self.rounds = 50
        self.lr = 0.0001
        self.alpha=0.90
        self.t = 0.5
        self.model = model.to(self.device)
        
        
        ## AL hyperparameters
        self.rho = 1
        self.delta = 1
        self.warm_start = 100
        
        
        ########################### DO NOT TOUCH STARTS FROM HERE ####################
        ## optimization variables: ls are Lagrangian multipliers
        self.s = torch.zeros((self.X.shape[0], 1), requires_grad=True, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            num_constrains = self.constrain().shape[0]
        self.ls = torch.zeros((num_constrains, 1), requires_grad=False, dtype=torch.float64, device=self.device)
        
        
        ## define your solver here
        self.optim_model = AdamW([
                    {'params': self.model.parameters(), 'lr': self.lr*1},
                    # {'params': self.s, 'lr': self.lr*1}
                    ])
        self.optim_s = AdamW([
                    # {'params': self.model.parameters(), 'lr': self.lr*1},
                    {'params': self.s, 'lr': self.lr*1}
                    ])
    
    def objective(self):
        n_pos = torch.sum(self.y==1)
        return -self.s.T@(self.y==1).double()/n_pos


    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        fx = self.model(self.X)
        ineq = torch.maximum(torch.tensor(0), \
            # self.alpha * torch.sum(self.s) - self.s.T@(self.y==1).double()
            self.alpha - self.s.T@(self.y==1).double() / torch.sum(self.s) 
            )
        eqs = torch.maximum(self.s+fx-1-self.t, torch.tensor(0)) - torch.maximum(-self.s, fx-self.t)
        ## two options: 1) treat them as separate constrains, 2) folding
        ## option 1
        # return torch.cat([ineq, eqs], dim=0)
        # return torch.cat([self.X.shape[0]*ineq, eqs], dim=0)
        # return torch.cat([ineq, eqs/self.X.shape[0]], dim=0)
        
        ## option 2
        # return torch.abs(ineq) + torch.sum(torch.abs(eqs))
        return torch.cat([ineq, torch.sum(torch.abs(eqs)).view(1, 1)], dim=0)
        # return torch.maximum(ineq, torch.sum(torch.abs(eqs)).view(1, 1))
    
    
    def warmstart(self):
        optim = AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        for _ in range(self.warm_start):
            L = criterion(self.model(self.X), self.y)
            L.backward()
            optim.step()
            
            with torch.no_grad():
                prediction = (self.model(self.X).detach().cpu().numpy() > self.t).astype(int)
                TP = int(prediction.T@(self.y.detach().cpu().numpy()==1).astype(int))
                
                print(f"========== Warm Start Round {_}/{self.warm_start} ===========")
                print("Precision: {:3f} \t Recall {:3f}".format(1.0*TP/np.sum(prediction>self.t), 1.0*TP/torch.sum(self.y==1)))
                constrains = self.constrain()
                print("Obj: {}\tEQ: {}\tIEQ: {}".format(self.objective().item(), constrains[0].item(), torch.sum(constrains[1:]).item()))
    
    @torch.no_grad()
    def initialize_with_feasiblity(self, quiet=False):
        self.s -= self.s
        self.s += (self.model(self.X) > self.t).int()
        if not quiet:
            print("================= Init with feasibility ===================")
            constrains = self.constrain()
            print("EQ: {}\tIEQ: {}".format(constrains[0], torch.sum(constrains[1:])))
            print("Obj: {}\tConstrain: {}".format(self.objective(), self.constrain()))
            
            
    def solve_sub_problem(self, update_model=True):
        if update_model:
            self.optim_model.zero_grad(set_to_none=True)
        else:
            self.optim_s.zero_grad(set_to_none=True)
        L = self.AL_func()
        L.backward()
        if update_model:
            self.optim_model.step()
        else:
            self.optim_s.step()



    def fit(self):
        if self.warm_start > 0:
            self.warmstart()
            
        self.initialize_with_feasiblity()
        sigmoid = nn.Sigmoid()
        for r in range(self.rounds):
            # for i in range(self.subprob_max_epoch):
            for i in range(100):
                self.solve_sub_problem(update_model=True)
            
            # for i in range(self.subprob_max_epoch):
            for i in range(400):
                self.solve_sub_problem(update_model=False)
                # with torch.no_grad():
                #     self.s.copy_((sigmoid(self.s.data-0.5) > 0.5).int())
                    # self.s.copy_(self.s.data.clamp(0, 1))
                    
            
            ## update lagrangian multiplier and evaluation
            # self.initialize_with_feasiblity(quiet=True)
            with torch.no_grad():
                self.update_langrangian_multiplier()
                prediction = (self.model(self.X).detach().cpu().numpy() > self.t).astype(int)
                TP = int(prediction.T@(self.y.detach().cpu().numpy()==1).astype(int))
                
                print(f"========== Round {r}/{self.rounds} ===========")
                print("Precision: {:3f} \t Recall {:3f}".format(1.0*TP/np.sum(prediction>self.t), 1.0*TP/torch.sum(self.y==1)))
                constrains = self.constrain()
                print("Obj: {}\tIEQ: {}\tEQ: {}".format(self.objective().item(), constrains[0].item(), torch.sum(constrains[1:]).item()))
                # print(self.s)
                self.rho *= self.delta  