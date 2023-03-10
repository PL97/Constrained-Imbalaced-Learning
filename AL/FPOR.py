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
        self.sto = self.args.sto
        
        self.lr_adaptor = 1
        self.r = 0
        
        
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
        # self.s = torch.sigmoid(self.s)
        # self.s.data.copy_(trainloader.targets)

        num_constrains = args.num_constrains
            
        self.ls = torch.zeros((num_constrains, 1), requires_grad=False, \
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
        # s = self.s[idx]
        s = self.adjust_s(self.s)
        # all_y = y.to(self.device)
        all_y = self.trainloader.targets.to(self.device)
        n_pos = torch.sum(all_y==1)
        m = nn.Softmax(dim=1)
        # return -s.T@(all_y==1).double()/n_pos
        fx = m(self.model(X))[:, 1].view(-1, 1)
        # return -s.T@(all_y==1).double()/n_pos - 0.1*torch.mean(fx.T*torch.log2(fx))
        # return -s.T@(all_y==1).double()/n_pos
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
        m = nn.Softmax(dim=1)
        X = X.float()
        fx = m(self.model(X))[:, 1].view(-1, 1)
        ineq = torch.maximum(torch.tensor(0), \
            self.alpha - all_s.T@(all_y==1).double() / torch.sum(all_s) 
            )
        # eqs = torch.maximum(s+fx-1-self.t, torch.tensor(0)) - torch.maximum(-s, fx-self.t)
        # zero_constrain = torch.mean(torch.abs(self.s[idx]*(1-self.s[idx])))
        # return torch.cat([ineq.view(1, 1), (torch.mean(torch.abs(eqs))+zero_constrain).view(1, 1)], dim=0)
        
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
    
    
    def warmstart(self):
        """warm start to get a good initialization, empirically this can speedup the convergence and improve the performance
        """
        optim = AdamW([
                {'params': self.model.parameters(), 'lr': self.lr}
                ])
        # criterion = WCE(npos=torch.sum(self.y==1).item(), nneg=torch.sum(self.y==0).item(), device=self.device)
        criterion = self.args.criterion
        for epoch in range(self.warm_start):
            self.model.train()
            for _, X, y in self.trainloader:
                y = y.view(-1).long()
                X = X.float()
                X, y = X.to(self.device), y.to(self.device)
                optim.zero_grad()
                loss = criterion(self.model(X), y.flatten().long())
                loss.backward()
                optim.step()
                
            with torch.no_grad():
                self.model.eval()
                train_metrics = self.test(self.trainloader)
                
                print(f"========== Warm Start Round {epoch}/{self.warm_start} ===========")
                print("Precision: {:.3f} \t Recall {:.3f} \t F_beta {:.3f} \t AP {:.3f}".format(\
                        train_metrics['precision'], train_metrics['recall'], train_metrics['F_beta'], train_metrics['AP']))

    
    @torch.no_grad()
    def initialize_with_feasiblity(self):
        """Another trick that boost the performance, initialize variable s with feasiblity guarantee
        """
        m = nn.Softmax(dim=1)
        self.s -= self.s
        for idx, X, y in self.trainloader:
            X = X.to(self.device)
            X = X.float()
            self.s[idx] += (m(self.model(X))[:, 1].view(-1, 1) >= self.t).int()/self.lr_adaptor

    def fit(self):
        best_AP = 0
        if self.warm_start > 0:
            self.warmstart()
        
        # self.initialize_with_feasiblity()
        m = nn.Sigmoid()
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
            # wandb.watch(self.model)        
            ## update lagrangian multiplier and evaluation
            # self.initialize_with_feasiblity()
            with torch.no_grad():
                self.model.eval()
                self.update_langrangian_multiplier()
                
                ## log training performance
                train_metrics = self.test(self.trainloader)
                val_metrics = self.test(self.valloader)
                test_metrics = self.test(self.testloader)
                
                print(f"========== Round {r}/{self.rounds} ===========")
                print("Precision: {:3f} \t Recall {:3f} \t F_beta {:.3f} \t AP {:.3f}".format(\
                        train_metrics['precision'], train_metrics['recall'], train_metrics['F_beta'], train_metrics['AP']))
                constrains = self.constrain()
                print("Obj: {}\tIEQ: {}\tEQ: {}".format(self.objective().item(), constrains[0].item(), torch.sum(constrains[1:]).item()))
                print("(val)Precision: {:3f} \t Recall {:3f} F_beta {:.3f} AP:{:.3f}".format(\
                        val_metrics['precision'], val_metrics['recall'], val_metrics['F_beta'], val_metrics['AP']))
                  
                print("(test)Precision: {:3f} \t Recall {:3f} F_beta {:.3f} AP:{:.3f}".format(\
                        test_metrics['precision'], test_metrics['recall'], test_metrics['F_beta'], test_metrics['AP']))
                  
                if val_metrics['AP'] > best_AP:
                    final_model = deepcopy(self.model)
                    best_AP = val_metrics['AP']
                
                
                wandb.log({ "trainer/global_step": r, \
                            "train/Obj": self.objective().item(), \
                            "train/IEQ": constrains[0].item(), \
                            "train/EQ": torch.sum(constrains[1:]).item(), \
                            "train/Precision": train_metrics['precision'], \
                            "train/Recall": train_metrics['recall'], \
                            "train/F_beta": train_metrics['F_beta' ], \
                            "train/AP": train_metrics['AP'], \
                            "val/Precision": val_metrics['precision'], \
                            "val/Recall": val_metrics['recall'], \
                            "val/F_beta": val_metrics['F_beta'], \
                            "val/AP": val_metrics['AP'], \
                            "test/Precision": test_metrics['precision'], \
                            "test/Recall": test_metrics['recall'], \
                            "test/F_beta": test_metrics['F_beta'], \
                            "test/AP": test_metrics['AP']
                            })
                
        
        final_model_name = f"{self.workspace}/final.pt"
        torch.save(self.model, final_model_name)
        # art_model = wandb.Artifact(f"{self.args.dataset}-{self.args.model}-{self.wandb_run.id}", type='model')
        # art_model.add_file(final_model_name)
        # wandb.log_artifact(art_model, aliases=["final"])
        
        try:
            wandb.run.summary["train_precision"] = train_metrics['precision']
            wandb.run.summary["train_recall"] = train_metrics['recall']
            wandb.run.summary["val_precision"] = val_metrics['precision']
            wandb.run.summary["val_recall"] = val_metrics['recall']
        except:
            print("skip AL...")
        
        # self.draw_graphs()

        return final_model