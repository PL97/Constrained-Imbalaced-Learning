import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## focal loss
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


## LDAMLoss
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


## logit adjustment
def compute_adjustment(train_loader, tro, device):
    """compute the base probabilities"""

    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    return adjustments

class LALoss(nn.Module):
    def __init__(self, train_loader, tro, device):
        super(LALoss, self).__init__()
        self.adjustments = compute_adjustment(train_loader=train_loader, tro=tro, device=device)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, target):
        x = x + self.adjustments
        return self.criterion(x, target)


def AP_loss(logits,targets):
    
    delta=1.0

    grad=torch.zeros(logits.shape).cuda()
    metric=torch.zeros(1).cuda()

    if torch.max(targets)<=0:
        return grad, metric
  
    labels_p=(targets==1)
    fg_logits=logits[labels_p]
    threshold_logit=torch.min(fg_logits)-delta

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n=((targets==0)&(logits>=threshold_logit))
    valid_bg_logits=logits[valid_labels_n] 
    valid_bg_grad=torch.zeros(len(valid_bg_logits)).cuda()
    ########

    fg_num=len(fg_logits)
    prec=torch.zeros(fg_num).cuda()
    order=torch.argsort(fg_logits)
    max_prec=0

    for ii in order:
        tmp1=fg_logits-fg_logits[ii] 
        tmp1=torch.clamp(tmp1/(2*delta)+0.5,min=0,max=1)
        tmp2=valid_bg_logits-fg_logits[ii]
        tmp2=torch.clamp(tmp2/(2*delta)+0.5,min=0,max=1)
        a=torch.sum(tmp1)+0.5
        b=torch.sum(tmp2)
        tmp2/=(a+b)
        current_prec=a/(a+b)
        if (max_prec<=current_prec):
            max_prec=current_prec
        else:
            tmp2*=((1-max_prec)/(1-current_prec))
        valid_bg_grad+=tmp2
        prec[ii]=max_prec 

    grad[valid_labels_n]=valid_bg_grad
    grad[labels_p]=-(1-prec) 

    fg_num=max(fg_num,1)

    grad /= (fg_num)
    
    metric=torch.sum(prec,dim=0,keepdim=True)/fg_num

    return grad, 1-metric


class APLoss_helper(torch.autograd.Function):  
    def __init__(self):
        super().__init__()
    @staticmethod
    def forward(self, x, target):
        clf_grad, clf_loss = AP_loss(x, target)
        self.save_for_backward(clf_grad)
        return clf_loss
    
    @staticmethod
    def backward(self, out_grad1):
        g1, =self.saved_tensors
        return g1*out_grad1, None

class APLoss(nn.Module):
    def __init__(self):
        super(APLoss, self).__init__()
        self.helper_func = APLoss_helper()
        pass

    def forward(self, input, target):
        return self.helper_func.apply(input, target)
    
class WCE(nn.Module):
    def __init__(self, nneg, npos, device=None):
        super(WCE, self).__init__()
        weights = 1.0/torch.Tensor([nneg, npos])
        weights = weights / torch.amin(weights)
        if device:
            weights = weights.to(device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)
    
    def forward(self, input, target):
        return self.criterion(input, target)