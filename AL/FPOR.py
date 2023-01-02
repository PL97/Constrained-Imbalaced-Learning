import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys
import wandb

from AL.AL_base import AL_base



class FPOR(AL_base):
    @torch.no_grad()
    def __init__(self, X, y, X_val, y_val, model, device, workspace, args):
        self.args = args
        self.X, self.y, self.X_val, self.y_val = X, y, X_val, y_val
        self.device = device
        ## general hyperparameters (fine tune for best performance)
        self.subprob_max_epoch=100
        self.rounds = 1
        self.lr = 0.0001
        self.alpha=0.95
        self.t = 0.5
        self.model = model.to(self.device)
        self.solver = "AdamW"
        self.workspace = workspace
        
        ## AL hyperparameters
        self.rho = 10
        self.delta = 1
        self.warm_start = 100
        
        
        ## track hyparam
        self.wandb_run = wandb.init(project='FPOR', \
                   name=args.run_name, \
                   config={
                    'subprob_max_epoch': self.subprob_max_epoch, \
                    'rounds': self.rounds, \
                    'lr': self.lr, \
                    'alpha': self.alpha, \
                    't': self.t, \
                    'solver': self.solver \
                   })
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/precision", summary="max")
        wandb.define_metric("val/recall", summary="max")
        art_dataset = wandb.Artifact(self.args.dataset, type='dataset')
        art_dataset.add_file(f"binary_data/{self.args.dataset}.csv")
        wandb.log_artifact(art_dataset)
        config = wandb.config
        config.learning_rate = self.lr
        config.alpha=0.95
        
        
        
        ########################### DO NOT TOUCH STARTS FROM HERE ####################
        ## optimization variables: ls are Lagrangian multipliers
        self.s = torch.zeros((self.X.shape[0], 1), requires_grad=True, \
                            dtype=torch.float64, device=self.device)
        with torch.no_grad():
            num_constrains = self.constrain().shape[0]
            
        self.ls = torch.zeros((num_constrains, 1), requires_grad=False, \
                            dtype=torch.float64, device=self.device)
        
        
        ## define your solver here
        if self.solver == "AdamW":
            self.optim = AdamW([
                        {'params': self.model.parameters(), 'lr': self.lr},
                        {'params': self.s, 'lr': 1}  ##best to set as 0.5
                        ])
        else:
            self.optim = LBFGS(list(self.model.parameters()) + list(self.s), \
                        history_size=10, 
                        max_iter=4, 
                        line_search_fn="strong_wolfe")
    
    def objective(self):
        n_pos = torch.sum(self.y==1)
        return -self.s.T@(self.y==1).double()/n_pos


    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self, folding=True):
        fx = self.model(self.X)
        ineq = torch.maximum(torch.tensor(0), \
            # self.alpha * torch.sum(self.s) - self.s.T@(self.y==1).double()
            self.alpha - self.s.T@(self.y==1).double() / torch.sum(self.s) 
            )
        eqs = torch.maximum(self.s+fx-1-self.t, torch.tensor(0)) - torch.maximum(-self.s, fx-self.t)
        
        
        ## inequality format get ride of interior
        # pos_idx = (self.y==1).flatten()
        # eqs_p = torch.maximum(torch.tensor(0), \
        #     torch.maximum(self.s[pos_idx]+fx[pos_idx]-1-self.t, torch.tensor(0)) - torch.maximum(-self.s[pos_idx], fx[pos_idx]-self.t)
        # )
        # neg_idx = (self.y==0).flatten()
        # eqs_n = torch.maximum(torch.tensor(0), \
        #     -torch.maximum(self.s[neg_idx]+fx[neg_idx]-1-self.t, torch.tensor(0)) + torch.maximum(-self.s[neg_idx], fx[neg_idx]-self.t)
        # )
        
        
        ## two options: 1) treat them as separate constrains, 2) folding
        if not folding:
            # option 1
            return torch.cat([ineq, eqs], dim=0)
            # return torch.cat([self.X.shape[0]*ineq, eqs], dim=0)
            # return torch.cat([ineq, eqs/self.X.shape[0]], dim=0)
        else:
            ## option 2
            # return torch.abs(ineq) + torch.sum(torch.abs(eqs))
            return torch.cat([ineq, torch.sum(torch.abs(eqs)).view(1, 1)], dim=0)
            # return torch.maximum(ineq, torch.sum(torch.abs(eqs)).view(1, 1))
            # return torch.cat([ineq, (torch.sum(eqs_p) + torch.sum(eqs_n).view(1, 1))], dim=0)
    
    
    def warmstart(self):
        optim = AdamW([
                {'params': self.model.parameters(), 'lr': self.lr}
                ])
        criterion = nn.BCELoss()
        # criterion = nn.BCEWithLogitsLoss()
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
        self.s += (self.model(self.X) >= self.t).int()
        if not quiet:
            print("================= Init with feasibility ===================")
            constrains = self.constrain()
            print("EQ: {}\tIEQ: {}".format(constrains[0], torch.sum(constrains[1:])))
            print("Obj: {}\tConstrain: {}".format(self.objective(), self.constrain()))
        



    def fit(self):
        best_precision = 0
        best_recall = 0
        if self.warm_start > 0:
            self.warmstart()
    
        self.initialize_with_feasiblity()
        sigmoid = nn.Sigmoid()
        for r in range(self.rounds):
            # 3. Log gradients and model parameters
            for i in range(self.subprob_max_epoch):
                self.solve_sub_problem()
                with torch.no_grad():
                    self.s.data.copy_((sigmoid(self.s.data-0.5) >= 0.5).int())
                    # self.s.copy_(self.s.data.clamp(0, 1))
            
            wandb.watch(self.model)        
            ## update lagrangian multiplier and evaluation
            self.initialize_with_feasiblity(quiet=True)
            with torch.no_grad():
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
                
                
                wandb.log({ "train/step": r, \
                            "train/Obj": self.objective().item(), \
                            "train/IEQ": constrains[0].item(), \
                            "train/EQ": torch.sum(constrains[1:]).item(), \
                            "train/Precision": train_precision, \
                            "train/Recall": train_recall, \
                            "val/Precision": val_precision, \
                            "val/Recall": val_recall
                            })
                
                if best_recall < val_recall:
                    best_recall_model = self.model
                    best_recall = val_recall
                
                if best_precision < val_precision:
                    best_precision_model = self.model
                    best_precision = val_precision
                
        
        final_model_name = f"{self.workspace}/final.pt"
        best_precision_model_name = f"{self.workspace}/best_precision.pt"
        best_recall_model_name = f"{self.workspace}/best_recall.pt"
        torch.save(self.model, final_model_name)
        torch.save(best_precision_model, best_precision_model_name)
        torch.save(best_recall_model, best_recall_model_name)
        art_model = wandb.Artifact(f"{self.args.dataset}-{self.args.model}-{self.wandb_run.id}", type='model')
        art_model.add_file(final_model_name)
        art_model.add_file(best_precision_model_name)
        art_model.add_file(best_recall_model_name)
        wandb.log_artifact(art_model, aliases=["final"])
        
        wandb.run.summary["train_precision"] = train_precision
        wandb.run.summary["train_recall"] = train_recall
        wandb.run.summary["val_precision"] = val_precision
        wandb.run.summary["val_recall"] = val_recall
        
        return self.model
    
    @torch.no_grad()
    def test(self, X, y):
        self.model.eval()
        prediction = (self.model(X).detach().cpu().numpy() >= self.t).astype(int)
        TP = int(prediction.T@(y.detach().cpu().numpy()==1).astype(int))
        precision = 1.0*TP/np.sum(prediction)
        recall = 1.0*TP/torch.sum(self.y==1)
        return precision, recall