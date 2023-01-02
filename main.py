import torch
import numpy as np
import random
import sys
sys.path.append("./")

from AL.FPOR import FPOR
from dataset.UCI import get_data
from models.MLP import MLP
import argparse
import os
from sklearn.model_selection import train_test_split
import wandb

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=2)
    parser.add_argument('--workspace', type=str, default="checkpoints/FPOR/")
    parser.add_argument('--dataset', type=str, default="diabetic")
    parser.add_argument('--run_name', type=str, default="0")
    parser.add_argument('--model', type=str, default="MLP")
    
    args = parser.parse_args()
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    
    ## create workspace
    try:
        os.removedirs(args.workspace)
        os.makedirs(args.workspace, exist_ok=True)
    except:
        os.makedirs(args.workspace, exist_ok=True)
    return args


if __name__ == "__main__":
    
    
    args = setup()
    

    device = torch.device("cuda")
    X_tensor, y_tenosr, X, y = get_data(name=args.dataset, device=device)
    
    X_train_tensor, X_tmp_tensor, y_train_tenosr, y_tmp_tensor = train_test_split(X_tensor, \
                                                      y_tenosr, \
                                                      test_size=0.2, \
                                                      random_state=args.random_seed)
    X_val_tensor, X_test_tensor, y_val_tensor, y_test_tensor = train_test_split(X_tmp_tensor, \
                                                      y_tmp_tensor, \
                                                      test_size=0.5, \
                                                      random_state=args.random_seed)
    
    
    model = MLP(input_dim=X_tensor.shape[1], hidden_dim=100, num_layers=10)
    
    model.train()
    trainer = FPOR(X = X_train_tensor, y = y_train_tenosr, \
                    X_val = X_val_tensor, y_val = y_val_tensor, \
                    device=device, model=model, workspace=args.workspace, args=args)
    model = trainer.fit()
    test_precision, test_recall = trainer.test(X_test_tensor, y_test_tensor)
    wandb.run.summary["test_precision"] = test_precision
    wandb.run.summary["test_recall"] = test_recall
    wandb.finish()
    