import torch
from scipy.stats import wasserstein_distance

from models.BaseUnit import TowerUnit
from models.BaseModel import BaseModel


class TarNet(BaseModel):
    def __init__(self, input_dim, share_dim=6,
                 share_hidden_dims =[64,64,64,64,64],base_hidden_dims=[100,100,100,100],
                 share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), 
                 task = 'regression'):
        super().__init__()
        self.share_unit = TowerUnit(input_dim, share_hidden_dims, share_dim, share_hidden_func, False)
        self.h_1 = TowerUnit(share_dim, base_hidden_dims, 1, base_hidden_func, False,task)
        self.h_0 = TowerUnit(share_dim, base_hidden_dims, 1, base_hidden_func, False,task)

    def forward(self, x, t):
        phi_x = self.share_unit(x)
        h_1_phi_x = self.h_1(phi_x)  # treat_y
        h_0_phi_x = self.h_0(phi_x)  # control_y
        y_preds = [h_0_phi_x,h_1_phi_x]
        t_pred = None #占位
        return t_pred,y_preds,phi_x   # pred_y


def tarnet_loss(t_pred, y_preds,tr, y1,phi_x,IPM=False, alpha=0):
    y_pred = y_preds[1] * tr + y_preds[0] * (1 - tr) 
    squared_errors = (y_pred - y1) ** 2 # [batch_size, 1]
    factual_loss = (tr * squared_errors).mean()
    IPM_term = 0
    if IPM:
        for i in range(tr.shape[1]):
            # phi_x = model.phi(realization_x)
            # phi: tensor([[-1.0000,  1.0000,  0.9996,  0.9999],
            #              [ 1.0000, -1.0000, -0.9996, -0.9999]]) 
            # tr:  tensor([[1.],[1.]])
            #如果当前样本的tr=1，就把对应的phi_x拿出来，计算wasserstein_distance
            #如果当前样本的tr=0，就把对应的phi_x拿出来，计算wasserstein_distance
            #然后把所有的phi_x的wasserstein_distance求平均，作为IPM_term
            #最后把factual_loss和IPM_term加权求和，作为total_loss    
            phi_x_treated = phi_x[tr[:, i] == 1].detach().cpu().numpy()
            phi_x_control = phi_x[tr[:, i] == 0].detach().cpu().numpy()
            if phi_x_treated.size > 0 and phi_x_control.size > 0:
                for dim in range(phi_x_treated.shape[1]):
                    IPM_term += wasserstein_distance(phi_x_treated[:, dim], phi_x_control[:, dim])
                IPM_term /= phi_x_treated.shape[1]
                        
    loss = factual_loss + alpha * IPM_term 
    return loss, factual_loss, IPM_term  # total_loss , y_loss,t_loss/表征loss


###################

class CFRNet(TarNet):
    '''The same as TARNet, but loss is modified to include IPM'''
    pass

def cfrnet_loss(t_pred, y_preds,tr, y1,phi_x,IPM=True, alpha=1.0):
    return tarnet_loss(t_pred, y_preds,tr, y1,phi_x,IPM, alpha)