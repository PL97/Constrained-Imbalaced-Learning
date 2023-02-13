
import torch
import torch.nn as nn

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
        X = self.active_set['X']
        X = X.to(self.device)
        return self.objective() + self.ls.T@self.constrain() \
                + (self.rho/2)* torch.sum(self.constrain()**2)

    
    

    
    def solve_sub_problem(self): 
        """solve the sub problem (stochastic)
        """
        m = nn.Sigmoid()
        for idx, X, y in self.trainloader:
            X, y = X.to(self.device), y.to(self.device)
            self.active_set = {"X": X, "y": y, "s": self.s[idx], "idx": idx}
            from copy import deepcopy
            tmp_s = deepcopy(self.s.data)

            if self.solver.lower() == "AdamW".lower():
                self.optim.zero_grad()
                L = self.AL_func()
                L.backward()
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
            
            # print(self.AL_func().item(), self.objective().item())

            with torch.no_grad():
                for i in idx:
                    tmp_s[i] = self.s.data[i]
                self.s.data.copy_(tmp_s)
                # self.s.data.copy_(m(self.s.data-self.t) >= self.t)
                
        

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
        # self.active_set = {
        #     'X': self.trainloader.data,
        #     'y': self.trainloader.targets, 
        #     's': self.s,
        #     'idx': list(range(self.trainloader.targets.shape[0]))
        # }
        constrain_output = self.constrain()
        self.ls += self.rho*constrain_output
        self.rho *= self.delta
        # if torch.norm(constrain_output) > torch.norm(self.pre_constrain, p=1):
        #     self.rho *= self.delta
        # self.pre_constrain = constrain_output
    
    def fit(self):
        """solve the constrained problem, in each round we iterativly solve the sub problem and update the lagrangian multiplier
        """
        for _ in range(self.rounds):
            for _ in range(self.subprob_max_epoch):
                self.solve_sub_problem()
            with torch.no_grad():
                self.update_langrangian_multiplier()