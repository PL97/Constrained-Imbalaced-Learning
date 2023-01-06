import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys
import wandb

from AL.AL_base import AL_base
from utils.loss import WCE
from AL.sampler import BatchSampler



class FPOR(AL_base):
    @torch.no_grad()
    def __init__(self, X, y, X_val, y_val, model, device, args):
        super().__init__()
        self.args = args
        self.X, self.y, self.X_val, self.y_val = X, y, X_val, y_val
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
        
        
        ## track hyparam
        self.wandb_run = wandb.init(project='debug', \
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
                    'delta': self.delta
                   })
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        wandb.define_metric("val/*", step_metric="trainer/global_step")
        # art_dataset = wandb.Artifact(self.args.dataset, type='dataset')
        # art_dataset.add_file(f"binary_data/{self.args.dataset}.csv")
        # wandb.log_artifact(art_dataset)
        
        self.my_data_sampler = BatchSampler(self.X.shape[0], args.batch_size)
        
        
        
        ########################### DO NOT TOUCH STARTS FROM HERE ####################
        ## optimization variables: ls are Lagrangian multipliers
        self.s = torch.zeros((self.X.shape[0], 1), requires_grad=True, \
                            dtype=torch.float64, device=self.device)
        # with torch.no_grad():
        #     num_constrains = self.constrain().shape[0]
        num_constrains = 2
            
        self.ls = torch.zeros((num_constrains, 1), requires_grad=False, \
                            dtype=torch.float64, device=self.device)
        
        self.active_set = None ## this defines a set of activate data that used for optimizing subproblem
        self.optim = AdamW([
                    {'params': self.model.parameters(), 'lr': self.lr},
                    {'params': self.s, 'lr': self.lr_s}  ##best to set as 0.5
                    ])
    
    def objective(self):
        X, y = self.fetchdata()
        s = self.s[self.active_set]
        n_pos = torch.sum(y==1)
        return -s.T@(y==1).double()/n_pos

    
    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        ## for image data, need to define a function to fetch the images
        X, y = self.fetchdata()
        s = self.s[self.active_set]
        
        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        ineq = torch.maximum(torch.tensor(0), \
            # self.alpha * torch.sum(s) - self.s.T@(y==1).double()
            self.alpha - s.T@(y==1).double() / torch.sum(s) 
            )
        eqs = torch.maximum(s+fx-1-self.t, torch.tensor(0)) - torch.maximum(-s, fx-self.t)
        
        return torch.cat([ineq, torch.sum(torch.abs(eqs)).view(1, 1)], dim=0)
    
    
    def warmstart(self):
        optim = AdamW([
                {'params': self.model.parameters(), 'lr': self.lr}
                ])
        # criterion = WCE(npos=torch.sum(self.y==1).item(), nneg=torch.sum(self.y==0).item(), device=self.device)
        criterion = self.args.criterion
        for _ in range(self.warm_start):
            self.model.train()
            for b in self.my_data_sampler:
                self.active_set = b
                X, y = self.fetchdata()
                
                optim.zero_grad()
                L = criterion(self.model(X), y.flatten().long())
                L.backward()
                optim.step()
                
            with torch.no_grad():
                self.model.eval()
                train_precision, train_recall = self.test(self.X, self.y)
                
                print(f"========== Warm Start Round {_}/{self.warm_start} ===========")
                print("Precision: {:3f} \t Recall {:3f}".format(train_precision, train_recall))

    
    @torch.no_grad()
    def initialize_with_feasiblity(self, quiet=False):
        m = nn.Softmax(dim=1)
        self.s -= self.s
        for b in self.my_data_sampler:
            self.active_set = b
            
            X, y = self.fetchdata()
            self.s[b] += (m(self.model(X))[:, 1].view(-1, 1) >= self.t).int()


    def fit(self):
        if self.warm_start > 0:
            self.warmstart()
        
        self.initialize_with_feasiblity()
        sigmoid = nn.Sigmoid()
        for r in range(self.rounds):
            # 3. Log gradients and model parameters
            for _ in range(self.subprob_max_epoch):
                self.model.train()
                self.solve_sub_problem()
                with torch.no_grad():
                    self.s.data.copy_((sigmoid(self.s.data-0.5) >= 0.5).int())

            wandb.watch(self.model)        
            ## update lagrangian multiplier and evaluation
            self.initialize_with_feasiblity(quiet=True)
            with torch.no_grad():
                self.model.eval()
                self.update_langrangian_multiplier()
                
                ## log training performance
                train_precision, train_recall = self.test(X=self.X, y=self.y)
                val_precision, val_recall = self.test(X=self.X_val, y=self.y_val)
                
                print(f"========== Round {r}/{self.rounds} ===========")
                print("Precision: {:3f} \t Recall {:3f}".format(train_precision, train_recall))
                constrains = self.constrain()
                print("Obj: {}\tIEQ: {}\tEQ: {}".format(self.objective().item(), constrains[0].item(), torch.sum(constrains[1:]).item()))
                print("(val)Precision: {:3f} \t Recall {:3f}".format(val_precision, val_recall))
                self.rho *= self.delta  
                
                
                wandb.log({ "trainer/global_step": r, \
                            "train/Obj": self.objective().item(), \
                            "train/IEQ": constrains[0].item(), \
                            "train/EQ": torch.sum(constrains[1:]).item(), \
                            "train/Precision": train_precision, \
                            "train/Recall": train_recall, \
                            "val/Precision": val_precision, \
                            "val/Recall": val_recall
                            })
                
        
        final_model_name = f"{self.workspace}/final.pt"
        torch.save(self.model, final_model_name)
        # art_model = wandb.Artifact(f"{self.args.dataset}-{self.args.model}-{self.wandb_run.id}", type='model')
        # art_model.add_file(final_model_name)
        # wandb.log_artifact(art_model, aliases=["final"])
        
        try:
            wandb.run.summary["train_precision"] = train_precision
            wandb.run.summary["train_recall"] = train_recall
            wandb.run.summary["val_precision"] = val_precision
            wandb.run.summary["val_recall"] = val_recall
        except:
            print("skip AL...")
        
        return self.model
    
    @torch.no_grad()
    def test(self, X, y):
        tmp_data_sampler = BatchSampler(X.shape[0], self.args.batch_size)
        self.model.eval()
        m = nn.Softmax(dim=1)
        prediction = []
        labels = []
        for b in tmp_data_sampler:
            self.active_set = b
            b_X, b_y = X[b], y[b]
            b_X, b_y = b_X.to(self.device), b_y.to(self.device)
            prediction.extend((m(self.model(b_X))[:, 1].detach().cpu().numpy() >= self.t).astype(int))
            labels.extend(b_y)
            
        prediction = np.stack(prediction, axis=0).reshape(-1, 1)
        labels = torch.stack(labels, axis=0).detach().cpu().numpy()
        TP = int(prediction.T@(labels==1).astype(int))
        precision = 1.0*TP/np.sum(prediction)
        recall = 1.0*TP/np.sum(labels==1)
        return precision, recall