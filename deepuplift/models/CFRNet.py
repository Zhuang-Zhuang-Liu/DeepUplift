from models.BaseModel import BaseModel, BaseLoss
from models.TarNet import *




class CFRNet(TarNet):
    '''The same as TARNet, but loss is modified to include IPM'''
    pass

def cfrnet_loss(t_pred, y_preds, t_true, y_true, phi_x,alpha=1.0, task='regression'):
    return BaseLoss(t_pred=t_pred, y_preds=y_preds, 
                       t_true=t_true, y_true=y_true,
                       phi_x=phi_x, 
                       loss_type='tarnet',IPM=True, 
                       alpha=alpha, task=task)