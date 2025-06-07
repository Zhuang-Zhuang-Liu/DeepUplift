import torch
import torch.nn.functional as F

from scipy.stats import wasserstein_distance
from models.BaseUnit import TowerUnit
from models.BaseModel import BaseModel


class TarNet(BaseModel):
    def __init__(self, input_dim, share_dim=6,
                 share_hidden_dims =[64,64,64,64,64],base_hidden_dims=[100,100,100,100],
                 share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), 
                 task = 'regression', model_type='tarnet'):
        super().__init__()
        self.model_type = model_type
        self.share_unit = TowerUnit(input_dim, share_hidden_dims, share_dim, share_hidden_func, False)
        self.h_1 = TowerUnit(input_dim=share_dim, hidden_dims=base_hidden_dims, 
                            activation= base_hidden_func, task=task,classi_nums=2)
        self.h_0 = TowerUnit(input_dim=share_dim, hidden_dims=base_hidden_dims, 
                            activation= base_hidden_func, task = task,classi_nums=2)
        if self.model_type == 'dragonnet':
            self.h_t = TowerUnit(input_dim=share_dim, hidden_dims=base_hidden_dims,
                                activation= base_hidden_func,task = 'classification',classi_nums=2)

    def forward(self, x, t):
        phi_x = self.share_unit(x)
        h_1_phi_x = self.h_1(phi_x) 
        h_0_phi_x = self.h_0(phi_x)
        y_preds = [h_0_phi_x,h_1_phi_x]
        
        if self.model_type == 'dragonnet' and self.h_t is not None:
            t_pred = self.h_t(phi_x)
        else:
            t_pred = None 
            
        return t_pred,y_preds,phi_x


def tarnet_loss(t_pred, y_preds, 
                t_true, y_true, phi_x=None, 
                loss_type='tarnet', 
                IPM=False, alpha=0, beta=1.0, tarreg=False, 
                task='regression'):
    
    if task == 'regression':
        loss0 = torch.sum((1. - t_true) * torch.square(y_true - y_preds[0]))  
        loss1 = torch.sum(t_true * torch.square(y_true - y_preds[1]))         
    elif task == 'classification':
            loss0 = torch.sum((1. - t_true) * F.binary_cross_entropy(y_preds[0][:, 1:2], y_true))
            loss1 = torch.sum(t_true * F.binary_cross_entropy(y_preds[1][:, 1:2], y_true))
    else:
        raise ValueError("task must be either 'regression' or 'classification'")
    outcome_loss = loss0 + loss1
    
    if loss_type == 'tarnet':
        if IPM:
            ipw_loss = 0
            for i in range(t_true.shape[1]):   
                phi_x_treated = phi_x[t_true[:, i] == 1].detach().cpu().numpy()
                phi_x_control = phi_x[t_true[:, i] == 0].detach().cpu().numpy()
                if phi_x_treated.size > 0 and phi_x_control.size > 0:
                    for dim in range(phi_x_treated.shape[1]):
                        ipw_loss += wasserstein_distance(phi_x_treated[:, dim], phi_x_control[:, dim])
                    ipw_loss /= phi_x_treated.shape[1]          
            loss = outcome_loss + alpha * ipw_loss
            return loss, outcome_loss, ipw_loss
        else:
            loss = outcome_loss
            return loss, outcome_loss, 0
            
    elif loss_type == 'dragonnet':

        if t_pred is None:
            raise ValueError("t_pred is required for DragonNet loss calculation")
        t_pred_processed = t_pred[:, 1]  
        t_pred_processed = (t_pred_processed + 0.01) / 1.02 
        t_pred_processed = t_pred_processed.unsqueeze(1) 
        treatment_loss = torch.sum(F.binary_cross_entropy(t_pred_processed, t_true))
        
        loss = outcome_loss + alpha * treatment_loss
        
        if tarreg:
            y_pred = t_true * y_preds[1] + (1 - t_true) * y_preds[0]  
            y_pert = y_pred
            targeted_regularization = torch.sum((y_true - y_pert)**2)
            loss = loss + beta * targeted_regularization
            
        return loss, outcome_loss, treatment_loss
    
    else:
        raise ValueError("loss_type must be either 'tarnet' or 'dragonnet'")


