import torch
import numpy as np
import random
import sys
sys.path.append("./")

from AL.FPOR import FPOR
from dataset.UCI import get_data
from models.MLP import MLP




if __name__ == "__main__":
    random_seed = 2
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    device = torch.device("cuda")
    X_tensor, y_tenosr, X, y = get_data(name='diabetic', device=device)
    
    model = MLP(input_dim=X_tensor.shape[1], hidden_dim=100, num_layers=10)
    
    model.train()
    trainer = FPOR(X = X_tensor, y = y_tenosr, device=device, model=model)
    model = trainer.fit()