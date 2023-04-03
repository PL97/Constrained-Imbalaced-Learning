import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys
import wandb

from AL.AL_base import AL_base
from utils.loss import WCE
from AL.sampler import BatchSampler
from models.utils import EarlyStopper
from sklearn.metrics import average_precision_score
from copy import deepcopy

from utils.utils import log



class FPOR(AL_base):
    @torch.no_grad()
    def __init__(self, trainloader, valloader, testloader, model, device, args):
        """Solver for Fix Precision and Optimize Recall (FPOR)

        Args:
            trainloader (Dataloader): train data loader
            valloader (_type_): valiation data loader
            model (nn.Module): nerual network
            device (_type_): cpu or cuda device
            args (parse object): any arguments that need to input should be put here
        """
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
        
        self.lr_adaptor = 1
        self.r = 0
        self.softmax = nn.Softmax(dim=1)
        
        
        ## track hyparam
        self.wandb_run = wandb.init(project=self.args.run_name, \
                   name=self.args.method, \
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
        self.s = torch.randn((args.datastats['train_num'], 1), requires_grad=True, \
                            dtype=torch.float64, device=self.device)

        num_constrains = args.num_constrains
            
        self.ls = torch.zeros((num_constrains, 1), requires_grad=False, \
                            dtype=torch.float64, device=self.device)
        
        self.active_set = None ## this defines a set of activate data(includes indices) that used for optimizing subproblem
        self.optim = AdamW([
                    {'params': self.model.parameters(), 'lr': self.lr},
                    {'params': self.s, 'lr': self.lr_s}  ##best to set as 0.5
                    ])
        
        self.optimizer = args.solver
        
        self.earlystopper = EarlyStopper(patience=5)
        self.beta = 1 ## to calualte the F-Beta score
        self.pre_constrain = np.inf

        self.scaler = torch.cuda.amp.GradScaler()
    
    def objective(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        s = self.adjust_s(self.s)
        all_y = self.trainloader.targets.to(self.device)
        n_pos = torch.sum(all_y==1)
        fx = self.softmax(self.model(X))[:, 1].view(-1, 1)
        n_pos = torch.sum(all_y==1)
        n_negs = torch.sum(all_y==0)
        weights = torch.tensor([n_pos/(n_pos+n_negs), n_negs/(n_negs+n_pos)]).to(self.device)
        weights = weights/(n_pos/(n_pos+n_negs))
        reweights = torch.ones(X.shape[0], 1).to(self.device)
        reweights[y==1] = weights[1]

        return -s.T@(all_y==1).double()/n_pos + 0.1*torch.norm(reweights * fx *(1-fx))/idx.shape[0]
        

    
    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        all_y = self.trainloader.targets.to(self.device)
        s = self.adjust_s(self.s[idx])
        all_s = self.adjust_s(self.s)
        X = X.float()
        fx = self.softmax(self.model(X))[:, 1].view(-1, 1)
        ineq = torch.maximum(torch.tensor(0), \
            self.alpha - all_s.T@(all_y==1).double() / torch.sum(all_s) 
            )
        
        n_pos = torch.sum(all_y==1)
        n_negs = torch.sum(all_y==0)
        weights = torch.tensor([n_pos/(n_pos+n_negs), n_negs/(n_negs+n_pos)]).to(self.device)
        weights = weights/(n_pos/(n_pos+n_negs))
        
        pos_idx = (y==1).flatten()
        eqs_p = weights[1] * torch.maximum(torch.tensor(0), \
            torch.maximum(s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-s[pos_idx], fx[pos_idx]-self.t)
        )
        neg_idx = (y==0).flatten()
        eqs_n = torch.maximum(torch.tensor(0), \
            -torch.maximum(s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-s[neg_idx], fx[neg_idx]-self.t)
        )
        
        delta = 0.1
        delta_2 = 0.1
        return torch.cat([torch.log(ineq/delta + 1).view(1, 1), torch.log(eqs_n/delta_2 + 1), torch.log(eqs_p/delta_2 + 1)])

    
    @torch.no_grad()
    def initialize_with_feasiblity(self):
        """Another trick that boost the performance, initialize variable s with feasiblity guarantee
        """
        self.s -= self.s
        for idx, X, y in self.trainloader:
            X = X.to(self.device)
            X = X.float()
            self.s[idx] += (self.softmax(self.model(X))[:, 1].view(-1, 1) >= self.t).int()/self.lr_adaptor

    def fit(self):
        best_AP = 0
        if self.warm_start > 0:
            self.warmstart()
        
        # self.initialize_with_feasiblity()
        for r in range(self.rounds):
            self.r = r
            # Log gradients and model parameters
            self.model.train()
            for _ in range(self.subprob_max_epoch):
                # print(f"================= {_} ==============")
                Lag_loss = self.solve_sub_problem()
                # print(f"lagrangian value: {Lag_loss}")
                if self.earlystopper.early_stop(Lag_loss):
                    print(f"train for {_} iterations")
                    break
        
            self.earlystopper.reset()
            with torch.no_grad():
                self.model.eval()
                self.update_langrangian_multiplier()
                
                ## log training performance
                train_metrics = self.test(self.trainloader)
                val_metrics = self.test(self.valloader)
                test_metrics = self.test(self.testloader)
                
                constraints = self.constrain()
                obj = self.objective().item()
                
                log(constraints, obj, train_metrics, val_metrics, test_metrics, verbose=True, r=r, rounds=self.rounds)
                  
                if val_metrics['AP'] > best_AP:
                    final_model = deepcopy(self.model)
                    best_AP = val_metrics['AP']
                
                
        
        final_model_name = f"{self.workspace}/final.pt"
        torch.save(self.model, final_model_name)
        
        # self.draw_graphs()

        return final_model