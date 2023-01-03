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
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from trainer.trainer import trainer_base
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader
from utils.loss import WCE

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=2)
    parser.add_argument('--workspace', type=str, default="checkpoints/FPOR/")
    parser.add_argument('--dataset', type=str, default="diabetic")
    parser.add_argument('--run_name', type=str, default="AL_trail1")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--method', type=str, default="AL")
    
    
    ## for AL method only
    parser.add_argument('--subprob_max_epoch', type=int, default=200)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.5)
    parser.add_argument('--solver', type=str, default="AdamW")
    parser.add_argument('--warm_start', type=int, default=1000)
    parser.add_argument('--learning_rate_s', type=float, default=1)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--rho', type=float, default=10)
    
    
    args = parser.parse_args()
    args.workspace = os.path.join(args.workspace, args.dataset, args.method)
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
                                                      stratify=y_tenosr.cpu().numpy(), \
                                                      random_state=args.random_seed)
    X_val_tensor, X_test_tensor, y_val_tensor, y_test_tensor = train_test_split(X_tmp_tensor, \
                                                      y_tmp_tensor, \
                                                      test_size=0.5, \
                                                      stratify=y_tmp_tensor.cpu().numpy(), \
                                                      random_state=args.random_seed)
    
    
    
    
    
    if args.method == "AL":
        model = MLP(input_dim=X_tensor.shape[1], hidden_dim=100, num_layers=10, output_dim=1)
        model.train()
        trainer = FPOR(X = X_train_tensor, y = y_train_tenosr, \
                        X_val = X_val_tensor, y_val = y_val_tensor, \
                        device=device, model=model, args=args)
        model = trainer.fit()
        test_precision, test_recall = trainer.test(X_test_tensor, y_test_tensor)
        wandb.run.summary["test_precision"] = test_precision
        wandb.run.summary["test_recall"] = test_recall
        wandb.finish()
    elif args.method == "WCE":
        model = MLP(input_dim=X_tensor.shape[1], hidden_dim=100, num_layers=10, output_dim=2)
        y_tenosr, y_val_tensor, y_test_tensor = y_tenosr.view(-1).long(), y_val_tensor.view(-1).long(), y_test_tensor.view(-1).long()
        nneg, npos = np.sum(y==0), np.sum(y==1)
        criterion = WCE(npos=npos, nneg=nneg)
        train_loader = DataLoader(TensorDataset(X_tensor, y_tenosr), batch_size=X_tensor.shape[0])
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=X_val_tensor.shape[0])
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=X_test_tensor.shape[0])
        
        wandb_logger = WandbLogger(project="FPOR", \
                                   name=args.run_name, \
                                   save_dir=args.workspace)
        max_epochs = args.rounds
        wandb_logger.experiment.config.update({
            'dataset': args.dataset, \
            'rounds': max_epochs, \
            'lr': args.learning_rate, \
            'solver': "AdamW"
        })
        wandb_logger.watch(model, log="all")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        wandb.define_metric("val/*", step_metric="trainer/global_step")
        args.wandb_logger = wandb_logger
        trainer = pl.Trainer(max_epochs=max_epochs, 
                            accelerator="gpu", 
                            devices=1, 
                            strategy = DDPStrategy(find_unused_parameters=False),
                            # log_every_n_steps=20,
                            auto_scale_batch_size=True,
                            logger=wandb_logger)
        
        MyLightningModule = trainer_base(X = X_train_tensor, y = y_train_tenosr, \
                        X_val = X_val_tensor, y_val = y_val_tensor, \
                        model=model, criterion=criterion, args=args)
        trainer.fit(MyLightningModule, \
                    train_dataloaders=train_loader, \
                    val_dataloaders=val_loader)
        test_precision, test_recall = MyLightningModule.test(X_test_tensor, y_test_tensor)
        wandb.run.summary["test_precision"] = test_precision
        wandb.run.summary["test_recall"] = test_recall
        wandb.finish()

    else:
        exit("not defined method")
    