import torch
import torch.nn as nn
from torch.optim import SGD, AdamW, Adam, LBFGS
import numpy as np
import sys
import wandb

from AL.AL_base import AL_base
from AL.FPOR import FPOR
from models.utils import EarlyStopper
from utils.utils import log


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
                    'batch_size': self.args.batch_size, \
                    'reg': args.reg
                   })
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        wandb.define_metric("val/*", step_metric="trainer/global_step")
        
        
        ########################### DO NOT TOUCH STARTS FROM HERE ####################
        ## optimization variables: ls are Lagrangian multipliers
        self.s = torch.randn((args.datastats['train_num'], args.datastats['train_num']), requires_grad=True, \
                            dtype=torch.float32, device=self.device)

        # num_constrains = args.num_constrains ** 2
        self.datapoints = args.num_constrains
            
        self.ls = torch.zeros((2, 1), requires_grad=False, \
                            dtype=torch.float32, device=self.device)
        
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
        self.rebalance_weights = torch.tensor([1., 1., 1.], requires_grad=False, device=self.device)
    
    def objective(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        all_y = self.trainloader.targets.to(self.device)
        all_s = self.adjust_s(self.s)
        
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
            nominator = torch.sum(all_s[i, (all_y==1).flatten()])
            denominator = torch.sum(all_s[i, :])
            ret += (nominator/denominator)
        obj = (1/n_pos)*ret
        
        # return -obj
        reg = reweights*(self.fx - torch.mean(self.fx)) ** 2
        
        return -obj - self.args.reg*torch.norm(reg)/idx.shape[0]



    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        s = self.s[idx, :]
        s = s[:, idx]
        s = self.adjust_s(s=s)
        
        constrains_1 = []
        constrains_2 = []
        for i, idx_tmpi in enumerate(idx):
            if y[i] == 0:
                continue
            j = (y==0).flatten()
            c1 = torch.maximum(torch.tensor(0), \
                    - torch.maximum(s[i, j] + self.fx[j] - self.fx[i] - 1, torch.tensor(0)) \
                        + torch.maximum(-s[i, j], self.fx[j] - self.fx[i])
                    )
            constrains_1.extend(c1)

            j = (y==1).flatten()
            c2 = torch.maximum(torch.tensor(0), \
            torch.maximum(s[i, j] + self.fx[j] - self.fx[i] - 1, torch.tensor(0)) \
                - torch.maximum(-s[i, j], self.fx[j] - self.fx[i])
            )
            constrains_2.extend(c2)


        constrains_1 = torch.stack(constrains_1)
        constrains_2 = torch.stack(constrains_2)
        delta = 0.01
        delta_2 = 0.01
        ret = torch.cat([torch.mean(torch.log(constrains_1/delta + 1)).view(1, 1), torch.mean(torch.log(constrains_2/delta_2 + 1)).view(1, 1)])
        return ret
    
                
        
    def AL_func(self):
        """defines the augmented lagrangian function based on the objective function and constrains

        Returns:
            augmented lagrangian function
        """
        X, idx, y = self.active_set['X'], self.active_set['idx'], self.active_set['y']
        ls = self.ls
        with torch.cuda.amp.autocast():
            self.fx = self.softmax(self.model(X))[:, 1].view(-1, 1)

        const = self.constrain()
        return self.objective() + ls.T@const \
                    + (self.rho/2)* torch.sum(const**2)
                
    def update_langrangian_multiplier(self):
        """update the lagrangian multipler
        """
        tmp_constrains = 0
        count_updates = 0
        for idx, X, y in self.trainloader:
            count_updates += 1
            X, y = X.to(self.device), y.to(self.device)
            self.active_set = {"X": X, "y": y, "s": self.s[idx], "idx": idx}

        self.ls += self.rho * self.constrain()
        self.rho *= self.delta
        if tmp_constrains > self.pre_constrain:
            self.rho *= self.delta
            self.pre_constrain = tmp_constrains


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
                with torch.cuda.amp.autocast():
                    loss = criterion(self.model(X), y.flatten().long())
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                
                
            with torch.no_grad():
                self.model.eval()
                train_metrics = self.test(self.trainloader)
                
                print(f"========== Warm Start Round {epoch}/{self.warm_start} ===========")
                print("Precision: {:.3f} \t Recall {:.3f} \t F_beta {:.3f} \t AP {:.3f}".format(\
                        train_metrics['precision'], train_metrics['recall'], train_metrics['F_beta'], train_metrics['AP']))


    def fit(self):
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
                
        
        final_model_name = f"{self.workspace}/final.pt"
        torch.save(self.model, final_model_name)
        
        self.draw_graphs(pred_only=True)

        return self.model