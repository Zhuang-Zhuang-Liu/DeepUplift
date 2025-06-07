from models.TarNet import *


class CFRNet(TarNet):
    '''The same as TARNet, but loss is modified to include IPM'''
    pass

def cfrnet_loss(t_pred, y_preds, tr, y1, phi_x,alpha=1.0, task='regression'):
    return tarnet_loss(t_pred, y_preds, tr, y1, phi_x, 
                       loss_type='tarnet',IPM=True, 
                       alpha=alpha, task=task)