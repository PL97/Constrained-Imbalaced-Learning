
import torch

class AL_base:
    @torch.no_grad()
    def __init__(self):
        pass
        
        
    def objective(self):
        pass
    
    ## we convert all C(x) <= 0  to max(0, C(x)) = 0
    def constrain(self):
        pass

    def AL_func(self):
        self.model.to(self.device)
        self.X = self.X.to(self.device)
        return self.objective() + self.ls.T@self.constrain() \
                + (self.rho/2)* torch.sum(self.constrain()**2)
    
    
    
    def solve_sub_problem(self):
        # L-BFGS: closure to clear the gradient, compute loss  and return it
        def closure():
            self.optim.zero_grad()
            L = self.AL_func()
            L.backward()
            return L
        
        if self.solver == "LBFGS":
            self.optim.step(closure)
        else:
            self.optim.zero_grad(set_to_none=True)
            L = self.AL_func()
            L.backward()
            self.optim.step()
    
    def update_langrangian_multiplier(self):
        constrain_output = self.constrain()
        self.ls += self.rho*constrain_output
        # for idx, c in enumerate(constrain_output):
        #     self.ls[idx] += self.rho * c
    
    def fit(self):
        for _ in range(self.rounds):
            for _ in range(self.subprob_max_epoch):
                self.solve_sub_problem()
            with torch.no_grad():
                self.update_langrangian_multiplier()

    
     
    
