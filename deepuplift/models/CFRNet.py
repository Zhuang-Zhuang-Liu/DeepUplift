
from models.TarNet import *


class CFRNet(TarNet):
    '''The same as TARNet, but loss is modified to include IPM'''
    pass

def cfrnet_loss(t_pred, y_preds,tr, y1,phi_x,IPM=True, alpha=1.0):
    return tarnet_loss(t_pred, y_preds,tr, y1,phi_x,IPM, alpha)