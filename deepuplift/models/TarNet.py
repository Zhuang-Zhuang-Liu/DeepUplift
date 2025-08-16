import torch
import torch.nn as nn

from models.BaseUnit import TowerUnit
from models.BaseModel import BaseModel, BaseLoss


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
            self.epsilon = nn.Linear(in_features=1, out_features=1)
            torch.nn.init.xavier_normal_(self.epsilon.weight)

    def forward(self, x, t):
        phi_x = self.share_unit(x)
        h_1_phi_x = self.h_1(phi_x) 
        h_0_phi_x = self.h_0(phi_x)
        y_preds = [h_0_phi_x,h_1_phi_x]

        if self.model_type == 'dragonnet' and self.h_t is not None:
            t_pred = self.h_t(phi_x)
            eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])  #target regularization
        else:
            t_pred = None 
            eps = None     
        return t_pred,y_preds,phi_x,eps




def tarnet_loss(t_pred, y_preds,t_true, y_true,task='regression'):
    return  BaseLoss(t_pred=t_pred, y_preds=y_preds, 
                       t_true=t_true, y_true=y_true,
                       loss_type='tarnet',IPM = False,task=task)



