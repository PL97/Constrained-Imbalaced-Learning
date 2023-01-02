import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, AdamW, Adam, LBFGS
from sklearn.metrics import precision_score, recall_score
import numpy as np
import wandb

class trainer_base(pl.LightningModule):
    def __init__(self, X, y, X_val, y_val, model, workspace, criterion, args):
        super().__init__()
        self.args = args
        self.X, self.y, self.X_val, self.y_val = X, y, X_val, y_val
        self.workspace = workspace
        self.model = model
        self.criterion = criterion
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = AdamW([
                {'params': self.model.parameters(), 'lr': self.args.learning_rate}
                ])
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train/loss", loss)
        return {'loss': loss, 'preds': pred, 'target': y}
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("test/loss", loss)
        return {'loss': loss, 'preds': pred, 'target': y}
        
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val/loss", loss)
        return {'loss': loss, 'preds': pred, 'target': y}
    
    
    ## define a set of metrics
    def cal_precision(self, pred, y):
        m = nn.Softmax(dim=1)
        norm_pred = (m(pred)[:, 1] > 0.5).int().detach().cpu().numpy()
        
        TP = int(norm_pred.T@(y.detach().cpu().numpy()==1).astype(int))
        precision = 1.0*TP/np.sum(norm_pred)
        return precision
    
    def cal_recall(self, pred, y):
        m = nn.Softmax(dim=1)
        norm_pred = (m(pred)[:, 1] > 0.5).int().detach().cpu().numpy()
        
        TP = int(norm_pred.T@(y.detach().cpu().numpy()==1).astype(int))
        recall = 1.0*TP/torch.sum(self.y==1)
        return recall
        
    
    def training_epoch_end(self, outputs):
        preds, target = [], []
        for out in outputs:
            preds.append(out['preds'])
            target.append(out['target'])
        preds = torch.cat(preds)
        target = torch.cat(target)
        # update and log
        precision = self.cal_precision(preds, target)
        recall = self.cal_recall(preds, target)
        self.log('train/Precision', precision)
        self.log('train/Recall', recall)
        
    def validation_epoch_end(self, outputs):
        preds, target = [], []
        for out in outputs:
            preds.append(out['preds'])
            target.append(out['target'])
        preds = torch.cat(preds)
        target = torch.cat(target)
        # update and log
        precision = self.cal_precision(preds, target)
        recall = self.cal_recall(preds, target)
        self.log('val/Precision', precision)
        self.log('val/Recall', recall)
        