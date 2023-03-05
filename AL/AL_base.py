
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import average_precision_score
import numpy as np

class AL_base:
    @torch.no_grad()
    def __init__(self):
        ## general hyperparameters (fine tune for best performance)
        pass
        
        
    def objective(self):
        """here to define your objective (minimization form)
        """
        pass
    
    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        """here to define your constrains, we conver all constrains into equality constrain
        """
        pass


    def AL_func(self):
        """defines the augmented lagrangian function based on the objective function and constrains

        Returns:
            augmented lagrangian function
        """
        X, idx = self.active_set['X'], self.active_set['idx']
        ls = torch.cat([self.ls[idx], self.ls[-1].reshape(1, 1)])
        X = X.to(self.device)
        return self.objective() + ls.T@self.constrain() \
                + (self.rho/2)* torch.sum(self.constrain()**2)

    
    def adjust_s(self, s):
        return torch.sigmoid(s * self.lr_adaptor)

    
    def solve_sub_problem(self): 
        """solve the sub problem (stochastic)
        """
        m = nn.Sigmoid()
        if not self.sto:
            self.optim.zero_grad()
        for idx, X, y in self.trainloader:
            X, y = X.to(self.device), y.to(self.device)
            self.active_set = {"X": X, "y": y, "s": self.s[idx], "idx": idx}
            from copy import deepcopy
            tmp_s = deepcopy(self.s.data)

            if self.solver.lower() == "AdamW".lower():
                if self.sto:
                    self.optim.zero_grad()
                L = self.AL_func()
                L.backward()
                if self.sto:
                    self.optim.step()
            else:
                # L-BFGS
                def closure():
                    self.optim.zero_grad()
                    L = self.AL_func()
                    L.backward()
                    return L
                self.optim.step(closure)
            
            self.s.requires_grad = True
            # break

            with torch.no_grad():
                for i in idx:
                    tmp_s[i] = self.s.data[i]
                    # tmp_s[i] = torch.sigmoid(self.s.data[i] - self.t) ## correct the bias by shif to right with a threhsold of 0.5
                self.s.data.copy_(tmp_s)
                # self.s.data.copy_(m(self.s.data-self.t) >= self.t)
        if not self.sto:
            self.optim.step()
        

        with torch.no_grad():
            self.model.train()
            ret_val = 0
            for idx, X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                self.active_set = {"X": X, "y": y, "s": self.s[idx], "idx": idx}
                L = self.AL_func()
                ret_val += L.item()
        return ret_val
                
                
        
    
    def update_langrangian_multiplier(self):
        """update the lagrangian multipler
        """
        tmp_constrains = 0
        tmp_ineq = self.ls[-1]
        count_updates = 0
        for idx, X, y in self.trainloader:
            count_updates += 1
            X, y = X.to(self.device), y.to(self.device)
            self.active_set = {"X": X, "y": y, "s": self.s[idx], "idx": idx}
            constrain_output = self.constrain()
            self.ls[idx] += self.rho*constrain_output[:-1]
            self.ls[-1] += self.rho*constrain_output[-1]
            tmp_constrains += torch.norm(constrain_output).item()
        self.ls[-1] = tmp_ineq + (self.ls[-1] - tmp_ineq)/count_updates  ## readjust update steps
        self.rho *= self.delta
        if tmp_constrains > self.pre_constrain:
            self.rho *= self.delta
            self.pre_constrain = tmp_constrains
    
    def fit(self):
        """solve the constrained problem, in each round we iterativly solve the sub problem and update the lagrangian multiplier
        """
        for _ in range(self.rounds):
            for _ in range(self.subprob_max_epoch):
                self.solve_sub_problem()
            with torch.no_grad():
                self.update_langrangian_multiplier()

    @torch.no_grad()
    def test(self, dataloader):
        """_summary_

        Args:
            dataloader (torch.utils.DataLoader): the data that is going to be tested

        Returns:
            precision: precision of the classificaiton model, defined as TP/(TP+FP)
            recall: recall of the classificaiton model, defined as TP/(TP+FN)
        """
        self.model.eval()
        m = nn.Softmax(dim=1)
        prediction = []
        labels = []
        pred_score = []
        indices = []
        for idx, X, y in dataloader:
            X = X.float()
            X, y = X.to(self.device), y.to(self.device)
            tmp_score = m(self.model(X))[:, 1].detach().cpu().numpy()
            pred_score.extend(tmp_score)
            prediction.extend((tmp_score >= self.t).astype(int))
            labels.extend(y)
            indices.extend(idx)
            
        prediction = np.stack(prediction, axis=0).reshape(-1, 1)
        pred_score = np.stack(pred_score, axis=0).reshape(-1, 1)
        indices = np.stack(indices, axis=0).reshape(-1, 1)
        labels = torch.stack(labels, axis=0).detach().cpu().numpy()
        TP = int(prediction.T@(labels==1).astype(int))
        precision = 1.0*TP/np.sum(prediction)
        recall = 1.0*TP/np.sum(labels==1)
        f_beta = (1+self.beta**2) * (precision*recall)/((self.beta**2)*precision+recall)
        AP = average_precision_score(y_true=labels, y_score=prediction)
        metric = {"precision": precision, "recall": recall, "F_beta": f_beta, "AP": AP, \
                "pred_score": pred_score, "labels": labels, "indices": indices}
        return metric
                
    @torch.no_grad()
    def draw_graphs(self):
        ## visualize the prediciton distribution  and algiment with s
        
        train_metric = self.test(self.trainloader)
        pred_score = train_metric['pred_score'].reshape(-1, 1).flatten().tolist()
        labels = train_metric['labels'].reshape(-1, 1).flatten().tolist()
        print(train_metric['precision'], train_metric['recall'], train_metric['F_beta'], train_metric['AP'])
        reordered_s = self.adjust_s(self.s).detach().cpu()[train_metric['indices']].reshape(-1, 1).flatten().tolist()
        
        tmp_df = pd.DataFrame({"prediciton": pred_score, "target": labels, "s": reordered_s})
        print(tmp_df.head())
        TP = np.asarray(tmp_df['s']).reshape(-1, 1).T@np.asarray(tmp_df['target']).reshape(-1, 1)
        pseudo_precision = 1.0*TP / np.sum(tmp_df['s'])
        pseudo_recall = 1.0*TP/np.sum(tmp_df['target'])
        print("pseudo precision and recall", pseudo_precision, pseudo_recall)
        
        
        ax = sns.displot(data=tmp_df, x="prediciton", hue="target")
        ax.fig.set_figwidth(10)
        ax.fig.set_figheight(5)
        plt.savefig("distribution.png")
        plt.close()
        
        ax = sns.scatterplot(data=tmp_df, x='prediciton', y='s', hue="target")
        plt.plot([0.5, 0.5], [0, 1], linewidth=1)
        plt.savefig("alignment.png")
        plt.close()
        
        ax = sns.displot(data=tmp_df, x="s", hue="target")
        ax.fig.set_figwidth(10)
        ax.fig.set_figheight(5)
        plt.savefig("s_distribution.png")
        plt.close()
        tmp_df.to_csv("prediciton.csv")