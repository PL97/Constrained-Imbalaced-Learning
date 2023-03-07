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
        all_s = self.adjust_s(self.s)

        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        
        n_pos = torch.sum(all_y==1)
        n_negs = torch.sum(all_y==0)
        weights = torch.tensor([n_pos/(n_pos+n_negs), n_negs/(n_negs+n_pos)]).to(self.device)
        weights = weights/(n_pos/(n_pos+n_negs))
        reweights = torch.ones(X.shape[0], 1).to(self.device)
        reweights[y==1] = weights[1]
        
        ret = 0
        # print(idx)
        for i in idx:
            nominator, denominator = 0, 0
            if all_y[i] == 0:
                continue
            nominator = torch.sum(all_s[i, (all_y==1).flatten()])
            denominator = torch.sum(all_s[i, :])
            ret += (nominator/denominator)
        obj = (1/n_pos)*ret + 0.1*torch.norm(reweights * fx *(1-fx))/idx.shape[0]
        return obj.double()
            
        

        

    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        X, y, idx = self.active_set['X'].to(self.device), self.active_set['y'].to(self.device), self.active_set['idx']
        s = self.adjust_s(self.s[idx, idx])
        m = nn.Softmax(dim=1)
        fx = m(self.model(X))[:, 1].view(-1, 1)
        
        constrains_1 = []
        constrains_2 = []
        for i, idx_tmpi in enumerate(idx):
            if y[i] == 0:
                continue
            j = (y==0).flatten()
            print(s[i, :].shape, self.s[idx].shape, fx.shape, j.shape)
            c1 = torch.maximum(torch.tensor(0), \
                    - torch.maximum(s[i, j] + fx[j] - fx[i] - 1, torch.tensor(0)) \
                        + torch.maximum(-s[i, j], fx[j] - fx[i])
                    )
            constrains_1.extend(c1)

            j = (y==1).flatten()
            c2 = torch.maximum(torch.tensor(0), \
            torch.maximum(s[i, j] + fx[j] - fx[i] - 1, torch.tensor(0)) \
                - torch.maximum(-s[i, j], fx[j] - fx[i])
            )
            constrains_2.extend(c2)


        constrains_1 = torch.stack(constrains_1)
        constrains_2 = torch.stack(constrains_2)
        delta = 0.1
        delta_2 = 0.1
        ret = torch.cat([torch.mean(torch.log(constrains_1/delta + 1)).view(1, 1), torch.mean(torch.log(constrains_2/delta_2 + 1)).view(1, 1)])
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
            # constrain_output = self.constrain()
            # self.ls[idx] += self.rho*constrain_output
            # tmp_constrains += torch.norm(constrain_output).item()

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

        return self.model