import torch
import numpy as np
import random
import sys
sys.path.append("./")

from AL.FPOR import FPOR
from AL.FROP import FROP
from AL.OFBS import OFBS
from dataset.UCI import get_data as get_uci_data
from dataset.cifar100 import get_data as get_cifar100_data
from models.MLP import MLP
from models.AlexNet import AlexNet
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
from torch.profiler import profile, record_function, ProfilerActivity

def setup():
    """parse arguments in commandline and return a args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=1997)
    parser.add_argument('--workspace', type=str, default="checkpoints/FPOR/")
    parser.add_argument('--dataset', type=str, default="diabetic")
    parser.add_argument('--run_name', type=str, default="AL_trail1")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--method', type=str, default="WCE")
    parser.add_argument('--batch_size', type=int, default=100)
    
    
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
    """initialize wandb looger

    Args:
        args (_type_): argurments containing training hyparams

    Returns:
        wandb_logger: will be used for Pytorch lightning trainer
    """
    wandb_logger = WandbLogger(project="AL_FPOR", \
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
    if args.dataset == "cifar100":
        trainloader, valloader, testloader, stats = get_cifar100_data(
                                                        batch_size=args.batch_size, \
                                                        random_seed=args.random_seed, \
                                                        with_idx=("AL" in args.method))
        model = AlexNet(num_classes=2, grayscale=False, input_shape=(1, 3, 32, 32))
    else:
        trainloader, valloader, testloader, stats = get_uci_data(name=args.dataset, \
                                                        batch_size=args.batch_size, \
                                                        random_seed=args.random_seed, \
                                                        with_idx=("AL" in args.method))
        model = MLP(input_dim=stats['feature_dim'], hidden_dim=100, num_layers=10, output_dim=stats['label_num'])
    
    args.datastats = stats
    print(stats)
    
    ## for debug and demo
    # X_tensor, y_tenosr, X, y = generate_data(dimension=2, device=device)
    
    
    
    if "AL" in args.method:
        criterion = WCE(npos=stats["label_distribution"][1], nneg=stats["label_distribution"][0], device=device)
        args.criterion = criterion
        if args.method == "AL_FPOR":
            # args.num_constrains = 3
            args.num_constrains = stats['train_num'] + 1
            trainer = FPOR(trainloader, \
                        valloader, \
                        device=device, model=model, args=args)  
        elif args.method == "AL_FROP":
            args.num_constrains = stats['train_num'] + 1
            # args.num_constrains = 3
            trainer = FROP(trainloader, \
                        valloader, \
                        device=device, model=model, args=args)
        elif args.method == "AL_OFBS":
            args.num_constrains = 2
            trainer = OFBS(trainloader, \
                        valloader, \
                        device=device, model=model, args=args)
        
        model = trainer.fit()
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     model = trainer.fit()
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        
        train_metrics = trainer.test(trainloader)
        val_metrics = trainer.test(valloader)
        test_metrics = trainer.test(testloader)
        
        
        ## final evaluation on train, val, and test set
        wandb.run.summary.update({"train_precision": train_metrics['precision'], \
                                  "train_recall": train_metrics['recall'], \
                                  "train_F_beta": train_metrics['F_beta'], \
                                  "train_AP": train_metrics['AP'], \
                                  "val_precision": val_metrics['precision'], \
                                  "val_recall": val_metrics['recall'], \
                                  "val_F_beta": val_metrics['F_beta'], \
                                  "val_AP": val_metrics['AP'], \
                                  "test_precision": test_metrics['precision'], \
                                  "test_recall": test_metrics['recall'], \
                                  "test_F_beta": test_metrics['F_beta'], \
                                  "test_AP": test_metrics['AP']})
        wandb.finish()
    
    elif args.method == "WCE":
        criterion = WCE(npos=stats["label_distribution"][1], nneg=stats["label_distribution"][0])
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
        
        
        ## final evaluation on train, val, and test set
        train_metrics = MyLightningModule.test(trainloader)
        val_metrics = MyLightningModule.test(valloader)
        test_metrics = MyLightningModule.test(testloader)
        
        wandb.run.summary.update({"train_precision": train_metrics['precision'], \
                                  "train_recall": train_metrics['recall'], \
                                  "train_F_beta": train_metrics['F_beta'], \
                                  "train_AP": train_metrics['AP'], \
                                  "val_precision": val_metrics['precision'], \
                                  "val_recall": val_metrics['recall'], \
                                  "val_F_beta": val_metrics['F_beta'], \
                                  "val_AP": val_metrics['AP'], \
                                  "test_precision": test_metrics['precision'], \
                                  "test_recall": test_metrics['recall'], \
                                  "test_F_beta": test_metrics['F_beta'], \
                                  "test_AP": test_metrics['AP']})
        wandb.finish()

    else:
        exit("not defined method")
    