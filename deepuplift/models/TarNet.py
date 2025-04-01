import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance


class RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Layer 1
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), # Layer 2
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), # Layer 3
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), # Layer 4
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), # Layer 5
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
    def forward(self, x):
        return self.net(x)

class HypothesisNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Layer 1
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), # Layer 2
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), # layer 3
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), # Layer 4
            nn.ELU(),
            nn.Linear(hidden_dim, 1)  # Predicting a single outcome in Layer 5
        )
        
    def forward(self, x):
        return self.net(x)
    
class TARNet(nn.Module):
    def __init__(self, input_dim, share_dim=12, base_dim=64):
        super().__init__()
        
        self.phi = RepresentationNetwork(input_dim, share_dim)   #共享层 shared layer
        self.h_1 = HypothesisNetwork(share_dim, base_dim) # treated hypothesis, t = 1
        self.h_0 = HypothesisNetwork(share_dim, base_dim) # control hypothesis, t = 0
    
    def forward(self, x, t):
        '''
        INPUTS:
            x [batch_size, 25] = 25 covariates for each factual sample in batch
            t [batch_size, 1]  = binary treatment applied for each factual sample in batch   
        '''
        # Send x through representation network to learn representation covariates, phi_x
        # Input: x [batch_size, 25] -> Output: phi_x [batch_size, hidden_dim]
        phi_x = self.phi(x)
        
        # Send phi_x through hypothesis network to learn h1 and h0 estimates
        # Input: phi_x [batch_size, hidden_dim], Output: h_1_phi_x [batch_size, 1]
        h_1_phi_x = self.h_1(phi_x)  # treat_y
        h_0_phi_x = self.h_0(phi_x)  # control_y
        
        # Mask the h1 estimates and h0 estimates according to t
        # predictions = [batch_size, 1], the h(\phi(x_i), t_i) for each element, i, in the batch

        #y_preds = h_1_phi_x * t + h_0_phi_x * (1 - t) # [batch_size, 1]

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

class CFRNet(TARNet):
    '''The same as TARNet, but loss is modified to include IPM'''
    pass

def cfrnet_loss(t_pred, y_preds,tr, y1,phi_x,IPM=True, alpha=1.0):
    return tarnet_loss(t_pred, y_preds,tr, y1,phi_x,IPM, alpha)
