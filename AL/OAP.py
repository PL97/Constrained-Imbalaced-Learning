import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys
import wandb

from AL.AL_base import AL_base
from AL.FPOR import FPOR
from models.utils import EarlyStopper


class OAP(AL_base):
    def __init__(self, trainloader, valloader, testloader, model, device, args):
        super().__init__()
        self.args = args
        self.trainloader, self.valloader, self.testloader = trainloader, valloader, testloader
        self.device = device
        self.model = model.to(self.device)
        ## general hyperparameters (fine tune for best performance)
        self.subprob_max_epoch= self.args.subprob_max_epoch  #200
        self.rounds = self.args.rounds                       #100
        self.lr = self.args.learning_rate                    #0.0001
        self.alpha= self.args.alpha                          #0.95
        self.t = self.args.t                                 #0.5
        self.solver = self.args.solver                       #"AdamW"
        self.warm_start = self.args.warm_start               #1000
        self.lr_s = self.args.learning_rate_s                #1
        self.rho = self.args.rho                             #10
        self.delta = self.args.delta                         #1
        self.workspace = self.args.workspace
        self.sto = self.args.sto
        
        self.lr_adaptor = 1
        self.r = 0
        
        
        ## track hyparam
        self.wandb_run = wandb.init(project=self.args.method, \
                   name=self.args.run_name, \
                   dir = self.args.workspace, \
                   config={
                    'dataset': self.args.dataset, \
                    'subprob_max_epoch': self.subprob_max_epoch, \
                    'rounds': self.rounds, \
                    'lr': self.lr, \
                    'lr_s': self.lr_s, \
                    'alpha': self.alpha, \
                    't': self.t, \
                    'solver': self.solver, \
                    'warm_start': self.warm_start, \
                    'rho': self.rho, \
                    'delta': self.delta, \
                    'batch_size': self.args.batch_size
                   })
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        wandb.define_metric("val/*", step_metric="trainer/global_step")
        
        
        ########################### DO NOT TOUCH STARTS FROM HERE ####################
        ## optimization variables: ls are Lagrangian multipliers
        self.s = torch.randn((args.datastats['train_num'], args.datastats['train_num']), requires_grad=True, \
                            dtype=torch.float64, device=self.device)

        # num_constrains = args.num_constrains ** 2
        self.datapoints = args.num_constrains
            
        self.ls = torch.zeros((2, 1), requires_grad=False, \
                            dtype=torch.float64, device=self.device)
        
        self.active_set = None ## this defines a set of activate data(includes indices) that used for optimizing subproblem
        if args.solver.lower() == "AdamW".lower():
            self.optim = AdamW([
                        {'params': self.model.parameters(), 'lr': self.lr},
                        {'params': self.s, 'lr': self.lr_s}  ##best to set as 0.5
                        ])
        else:
            self.optim = LBFGS(list(self.model.parameters()) + list(self.s), history_size=10, max_iter=4, line_search_fn="strong_wolfe")
        
        self.optimizer = args.solver
        
        self.earlystopper = EarlyStopper(patience=5)
        self.beta = 1 ## to calualte the F-Beta score
        self.pre_constrain = np.inf
    
    def objective(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        all_y = self.trainloader.targets.to(self.device)
        all_s = self.adjust(self.s)
        s = self.adujust(self.s[i])

        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        
        n_pos = torch.sum(all_y==1)
        n_negs = torch.sum(all_y==0)
        weights = torch.tensor([n_pos/(n_pos+n_negs), n_negs/(n_negs+n_pos)]).to(self.device)
        weights = weights/(n_pos/(n_pos+n_negs))
        reweights = torch.ones(X.shape[0], 1).to(self.device)
        reweights[y==1] = weights[1]
        
        ret = 0
        for i in idx:
            nominator, denominator = 0, 0
            if all_y[i] == 0:
                continue
            for j in idx:
                if all_y[j] == 1:
                    nominator += all_s[i, j]
                denominator += all_s[i, j]
            ret += (nominator/denominator)
        obj = (1/n_pos)*ret + 0.1*torch.norm(reweights * fx *(1-fx))/idx.shape[0]
        return obj.double()
            
        

        

    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        s = self.adjust_s(self.s[idx])
        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        
        constrains_1 = []
        constrains_2 = []
        for i, idx_tmpi in enumerate(idx):
            for j, idx_tmpj in enumerate(idx):
                if y[i] == 1 and y[j] == 0:
                    c1 = torch.maximum(torch.tensor(0), \
                    - torch.maximum(s[i, j] + fx[j] - fx[i] - 1, torch.tensor(0)) \
                        + torch.maximum(-s[i, j], fx[j] - fx[i])
                    )
                    constrains_1.append(c1)
                elif y[i] ==1 and y[j] == 1:
                    c2 = torch.maximum(torch.tensor(0), \
                    torch.maximum(s[i, j] + fx[j] - fx[i] - 1, torch.tensor(0)) \
                        - torch.maximum(-s[i, j], fx[j] - fx[i])
                    )
                    constrains_2.append(c2)
        constrains_1 = torch.stack(constrains_1)
        constrains_2 = torch.stack(constrains_2)
        delta = 1
        delta_2 = 1
        ret = torch.cat([torch.sum(torch.log(constrains_1/delta + 1)).view(1, 1), torch.sum(torch.log(constrains_2/delta_2 + 1)).view(1, 1)])
        return ret.double()
    
                
        
    def AL_func(self):
        """defines the augmented lagrangian function based on the objective function and constrains

        Returns:
            augmented lagrangian function
        """
        X, idx, y = self.active_set['X'], self.active_set['idx'], self.active_set['y']
        ls = self.ls
        X = X.to(self.device)

        # print(self.objective().shape, ls.shape, self.constrain().shape)
        # asdf
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