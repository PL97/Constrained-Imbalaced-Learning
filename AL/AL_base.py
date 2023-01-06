
import torch

class AL_base:
    @torch.no_grad()
    def __init__(self):
        ## general hyperparameters (fine tune for best performance)
        pass
        
        
    def objective(self):
        pass
    
    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        pass


    def AL_func(self):
        X = self.active_set['X']
        X = X.to(self.device)
        return self.objective() + self.ls.T@self.constrain() \
                + (self.rho/2)* torch.sum(self.constrain()**2)
    
    
    
    def solve_sub_problem(self): 
        # L-BFGS: closure to clear the gradient, compute loss  and return it
        for idx, X, y in self.trainloader:
            X, y = X.to(self.device), y.to(self.device)
            self.active_set = {"X": X, "y": y, "s": self.s[idx]}

            self.optim.zero_grad()
            L = self.AL_func()
            L.backward()
        self.optim.step()
    
    def update_langrangian_multiplier(self):
        constrain_output = self.constrain()
        self.ls += self.rho*constrain_output
    
    def fit(self):
        for _ in range(self.rounds):
            for _ in range(self.subprob_max_epoch):
                self.solve_sub_problem()
            with torch.no_grad():
                self.update_langrangian_multiplier()

    
     
    
