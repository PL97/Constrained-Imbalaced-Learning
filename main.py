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

import wandb
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from trainer.trainer import trainer_base
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset, DataLoader
from utils.loss import WCE
from dataset.sythetic import generate_data

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=2)
    parser.add_argument('--workspace', type=str, default="checkpoints/FPOR/")
    parser.add_argument('--dataset', type=str, default="diabetic")
    parser.add_argument('--run_name', type=str, default="AL_trail1")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--method', type=str, default="WCE")
    parser.add_argument('--batch_size', type=int, default=50)
    
    
    ## for AL method only
    parser.add_argument('--subprob_max_epoch', type=int, default=200)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.5)
    parser.add_argument('--solver', type=str, default="AdamW")
    parser.add_argument('--warm_start', type=int, default=10)
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

def pytorchlightning_wandb_setup(args):
    wandb_logger = WandbLogger(project="FPOR", \
                                name=args.run_name, \
                                save_dir=args.workspace)
    wandb_logger.experiment.config.update({
        'dataset': args.dataset, \
        'rounds': args.rounds, \
        'lr': args.learning_rate, \
        'solver': "AdamW"
    })
    wandb_logger.watch(model, log="all")
    wandb.define_metric("train/*", step_metric="trainer/global_step")
    wandb.define_metric("val/*", step_metric="trainer/global_step")
    return wandb_logger


if __name__ == "__main__":
    
    args = setup()

    device = torch.device("cuda")
    trainloader, valloader, testloader, stats = get_data(name=args.dataset, \
                                                        batch_size=args.batch_size, \
                                                        random_seed=args.random_seed)
    args.datastats = stats
    
    ## for debug and demo
    # X_tensor, y_tenosr, X, y = generate_data(dimension=2, device=device)
    
    model = MLP(input_dim=stats['feature_dim'], hidden_dim=100, num_layers=10, output_dim=stats['label_num'])
    
    if args.method == "AL":
        criterion = WCE(npos=stats["label_distribution"][1], nneg=stats["label_distribution"][0], device=device)
        args.criterion = criterion
        trainer = FPOR(trainloader, \
                        valloader, \
                        device=device, model=model, args=args)
        model = trainer.fit()
        train_precision, train_recall = trainer.test(trainloader)
        val_precision, val_recall = trainer.test(valloader)
        test_precision, test_recall = trainer.test(testloader)
        
        
        wandb.run.summary.update({"train_precision": train_precision, \
                                  "train_recall": train_recall, \
                                  "val_precision": val_precision, \
                                  "val_recall": val_recall, \
                                  "test_precision": test_precision, \
                                  "test_recall": test_recall})
        wandb.finish()
    elif args.method == "WCE":
        criterion = WCE(npos=stats["label_distribution"][1], nneg=stats["label_distribution"][0])
        trainloader.dataset.set_ret_idx(ret=False)
        valloader.dataset.set_ret_idx(ret=False)
        testloader.dataset.set_ret_idx(ret=False)

        args.wandb_logger = pytorchlightning_wandb_setup(args=args)
        trainer = pl.Trainer(max_epochs=args.rounds, 
                            accelerator="gpu", 
                            devices=1, 
                            strategy = DDPStrategy(find_unused_parameters=False),
                            log_every_n_steps=1,
                            auto_scale_batch_size=True,
                            logger=args.wandb_logger)
        
        MyLightningModule = trainer_base(
                        model=model, criterion=criterion, args=args)
        trainer.fit(MyLightningModule, \
                    train_dataloaders=trainloader, \
                    val_dataloaders=valloader)
        exit()
        
        train_precision, train_recall = MyLightningModule.test(trainloader)
        val_precision, val_recall = MyLightningModule.test(valloader)
        test_precision, test_recall = MyLightningModule.test(testloader)
        
        wandb.run.summary.update({"train_precision": train_precision, \
                                  "train_recall": train_recall, \
                                  "val_precision": val_precision, \
                                  "val_recall": val_recall, \
                                  "test_precision": test_precision, \
                                  "test_recall": test_recall})
        wandb.finish()

    else:
        exit("not defined method")
    