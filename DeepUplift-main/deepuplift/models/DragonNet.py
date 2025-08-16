import torch
from models.BaseModel import BaseLoss
from models.TarNet import TarNet


class DragonNet(TarNet):
    def __init__(self, input_dim, share_dim=6,
                 share_hidden_dims =[64,64,64,64,64],base_hidden_dims=[100,100,100,100],
                 share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), 
                 task = 'regression'):
        super().__init__(input_dim=input_dim, share_dim=share_dim,
                         share_hidden_dims=share_hidden_dims, base_hidden_dims=base_hidden_dims,
                         share_hidden_func=share_hidden_func, base_hidden_func=base_hidden_func,
                         task=task, model_type='dragonnet')


def dragonnet_loss(t_pred, y_preds,t_true, y_true,phi_x=None,tarreg_eps=None,alpha=1.0,beta=1.0,tarreg=True, task='regression'):
    return BaseLoss(t_pred=t_pred, y_preds=y_preds, 
                       t_true=t_true, y_true=y_true,
                       loss_type='dragonnet',
                       alpha=alpha, beta=beta, tarreg=tarreg, task=task, tarreg_eps=tarreg_eps)
