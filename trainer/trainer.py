import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD, AdamW, Adam, LBFGS
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
import wandb

class trainer_base(pl.LightningModule):
    def __init__(self, model, criterion, args):
        super().__init__()
        self.args = args
        self.workspace = self.args.workspace
        self.model = model
        self.criterion = criterion
        self.beta = 1
        
    def forward(self, X):
        return self.model(X)
    
    def configure_optimizers(self):
        optimizer = AdamW([
                {'params': self.model.parameters(), 'lr': self.args.learning_rate}
                ])
        return optimizer
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.view(-1).long()
        pred = self.model(X)
        loss = self.criterion(pred, y)
        self.log("train/loss", loss)
        return {'loss': loss, 'preds': pred, 'target': y}
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # this is the test loop
        return self._shared_eval(batch=batch, batch_idx=batch_idx, prefix="test")
        
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        return self._shared_eval(batch=batch, batch_idx=batch_idx, prefix="val")
    
    @torch.no_grad()
    def _shared_eval(self, batch, batch_idx, prefix):
        X, y = batch
        y = y.view(-1).long()
        pred = self.model(X)
        loss = self.criterion(pred, y)
        self.log(f"{prefix}/loss", loss, sync_dist=True)
        return {'loss': loss, 'preds': pred, 'target': y}
    
    ## define a set of metrics
    def cal_metrics(self, pred, y):
        m = nn.Softmax(dim=1)
        norm_pred = (m(pred)[:, 1] > 0.5).int().detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        TP = int(norm_pred.T@(y==1).astype(int))
        precision = 1.0*TP/np.sum(norm_pred)
        recall = 1.0*TP/np.sum(y==1)
        f_beta = (1+self.beta**2) * (precision*recall)/((self.beta**2)*precision+recall)
        AP = average_precision_score(y_true=y, y_score=norm_pred)
        return {"precision": precision, "recall": recall, "F_beta": f_beta, "AP": AP}
        
    

        
    @torch.no_grad()
    def _shared_epoch_end(self, outputs, prefix):
        preds, target = [], []
        for out in outputs:
            preds.append(out['preds'])
            target.append(out['target'])
        preds = torch.cat(preds)
        target = torch.cat(target)
        # update and log
        metrics = self.cal_metrics(preds, target)
        for k, v in metrics.items():
            self.log(f'{prefix}/{k}', v, sync_dist=True)
        
    @torch.no_grad()
    def training_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, prefix="train")
    
    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        return self._shared_epoch_end(outputs, prefix="val")
    
    @torch.no_grad()
    def test(self, dataloader):
        self.model.eval()
        m = nn.Softmax(dim=1)
        prediction = []
        labels = []
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            prediction.extend((m(self.model(X))[:, 1].detach().cpu().numpy() >= 0.5).astype(int))
            labels.extend(y)
            
        prediction = np.stack(prediction, axis=0).reshape(-1, 1)
        labels = torch.stack(labels, axis=0).detach().cpu().numpy()
        TP = int(prediction.T@(labels==1).astype(int))
        precision = 1.0*TP/np.sum(prediction)
        recall = 1.0*TP/np.sum(labels==1)
        f_beta = (1+self.beta**2) * (precision*recall)/((self.beta**2)*precision+recall)
        AP = average_precision_score(y_true=labels, y_score=prediction)
        metric = {"precision": precision, "recall": recall, "F_beta": f_beta, "AP": AP}
        return metric
        