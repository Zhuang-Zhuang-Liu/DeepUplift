import torch
import torch.nn as nn
import math
from geomloss import SamplesLoss
import sys


def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def sigmod2(y):
    # y = torch.clamp(0.995 / (1.0 + torch.exp(-y)) + 0.0025, 0, 1)
    # y = torch.clamp(y, -16, 16)
    y=torch.sigmoid(y)
    # y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025

    return y

def safe_sqrt(x):
    ''' Numerically safe version of Pytoch sqrt '''
    return torch.sqrt(torch.clip(x, 1e-9, 1e+9))

class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, 
                 do_rate=0.2, use_batch_norm1d=True, normalization="divide", device='cpu'):
        super(ShareNetwork, self).__init__()
        self.do_rate = do_rate
        self.device = device
        if use_batch_norm1d:
            print("use BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=self.do_rate),
                nn.Linear(share_dim, share_dim),
                # nn.BatchNorm1d(share_dim),
                nn.ELU(),
                nn.Dropout(p=self.do_rate),
                nn.Linear(share_dim, base_dim),
                # nn.BatchNorm1d(base_dim),
                nn.ELU(),
                nn.Dropout(p=self.do_rate)
            )
        else:
            print("No BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=do_rate),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=do_rate),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
            )

        self.DNN.apply(init_weights)
        self.normalization = normalization
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        h_rep = self.DNN(x)
        if self.normalization == "divide":
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep_norm

# 去掉cfg，把里面涉及的参数，都放到模型输入里
class BaseModel(nn.Module):
    def __init__(self, base_dim, do_rate):
        super(BaseModel, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate)
        )
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits

class BaseModel4MetaLearner(nn.Module):
    def __init__(self, input_dim, base_dim, do_rate, device):
        super(BaseModel4MetaLearner, self).__init__()
        self.DNN = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate),
            nn.Linear(base_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate),
            nn.Linear(base_dim, 1),
        )
        self.DNN.apply(init_weights)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        logit = self.DNN(x)
        return logit


class PrpsyNetwork(nn.Module):
    """propensity network"""
    def __init__(self, base_dim, do_rate):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, do_rate)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, do_rate):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, do_rate)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, do_rate):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, do_rate)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class TauNetwork(nn.Module):
    """pseudo tau network"""
    def __init__(self, base_dim, do_rate):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, do_rate)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        # return self.tanh(p)
        return tau_logit

class ESX(nn.Module):
    """ESX"""
    def __init__(self, prpsy_network: PrpsyNetwork, \
                 mu1_network: Mu1Network, mu0_network: Mu0Network, tau_network: TauNetwork, shareNetwork: ShareNetwork, device):
        super(ESX, self).__init__()
        # self.feature_extractor = feature_extractor
        self.shareNetwork = shareNetwork.to(device)
        self.prpsy_network = prpsy_network.to(device)
        self.mu1_network = mu1_network.to(device)
        self.mu0_network = mu0_network.to(device)
        self.tau_network = tau_network.to(device)
        self.device = device
        self.to(device)

    def forward(self, inputs):
        shared_h = self.shareNetwork(inputs)

        # propensity output_logit
        p_prpsy_logit = self.prpsy_network(shared_h)

        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.05, 0.95)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1 # Refer to the naming in TARnet/CFR
        p_h0 = p_mu0 # Refer to the naming in TARnet/CFR

        # entire space
        p_estr = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)

        return p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_h1, p_h0, shared_h


def wasserstein_torch(X,t ):
    """ Returns the Wasserstein distance between treatment groups """
    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)
    return imbalance_loss

def mmd2_torch(X, t):
    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    samples_loss = SamplesLoss(loss="energy", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)
    return imbalance_loss



#######################################

class ESX_Model(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, 
                    do_rate=0.2, use_batch_norm1d=True, normalization="divide",device ='cpu'):
        super().__init__()   # 用途：调用父类的初始化方法，确保子类也能正确初始化父类的属性。
        self.share_network = ShareNetwork(input_dim, share_dim, base_dim, 
                                          do_rate=do_rate, use_batch_norm1d=use_batch_norm1d, normalization=normalization, device=device)
        self.prpsy_network = PrpsyNetwork(base_dim, do_rate)
        self.mu1_network = Mu1Network(base_dim, do_rate)
        self.mu0_network = Mu0Network(base_dim, do_rate)
        self.tau_network = TauNetwork(base_dim, do_rate)

        self.esx = ESX(self.prpsy_network, self.mu1_network, self.mu0_network, 
                       self.tau_network, self.share_network,device)

    def forward(self, inputs,tr=None):
        """tr: 与efin对齐, 无实际意义"""
        p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_h1, p_h0, shared_h = self.esx(inputs)
        return p_prpsy,[p_h1, p_h0],p_estr, p_escr, tau_logit, mu1_logit, mu0_logit,p_prpsy_logit, shared_h



def esx_loss(   p_prpsy,y_preds,
                t_labels, y_labels, 
                p_estr, p_escr, tau_logit, mu1_logit, mu0_logit,p_prpsy_logit, shared_h,  # *eps
                # , e_labels,
                prpsy_w=1.0, escvr1_w=1.0, escvr0_w=1.0, h1_w=1.0, h0_w=1.0, 
                mu1hat_w=1.0, mu0hat_w=1.0,imb_dist_w=0.0, imb_dist="wass"):
    """"
    t_pred = p_prpsy_logit
    y_preds = [p_h1, p_h0] 
    """
    p_h1 = y_preds[1]
    p_h0 = y_preds[0]

    p_t = 0.5
    sample_weight = torch.ones_like(t_labels) 
    # define loss function
    loss_w_fn = nn.BCELoss(weight=sample_weight)
    loss_fn = nn.BCELoss()
    loss_w_with_logit_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(1 / (2 * p_t)))  # for propensity loss

    # loss for propensity: 交叉熵
    t_labels = t_labels.squeeze(1)
    prpsy_loss = prpsy_w * loss_w_with_logit_fn(p_prpsy_logit.squeeze(1), t_labels.float()  )     
    
    # loss for ESTR, ESCR
    # p_estr: torch.Size([64, 1]) y_labels: torch.Size([64, 1]) t_labels: torch.Size([64])
    y_labels = y_labels.squeeze(1)
    t_labels = t_labels.float()
    loss_w_fn = nn.BCELoss(weight=sample_weight)
    estr_loss = escvr1_w * loss_w_fn(p_estr, (y_labels * t_labels).float().unsqueeze(1))
    escr_loss = escvr0_w * loss_w_fn(p_escr,(y_labels * (1 - t_labels)).float().unsqueeze(1))

    # loss for TR, CR
    loss_fn = nn.BCELoss()

    tr_loss = h1_w * loss_fn(p_h1,y_labels.unsqueeze(1))  # * (1 / (2 * p_t))
    cr_loss = h0_w * loss_fn(p_h0,y_labels.unsqueeze(1))  # * (1 / (2 * (1 - p_t)))
    # loss for cross TR: mu1_prime, cross CR: mu0_prime
    cross_tr_loss = mu1hat_w * loss_fn(torch.sigmoid(mu0_logit + tau_logit),y_labels.unsqueeze(1))
    cross_cr_loss = mu0hat_w * loss_fn(torch.sigmoid(mu1_logit - tau_logit),y_labels.unsqueeze(1))
    imb_dist_val = 0
    if imb_dist_w > 0:
        if imb_dist == "wass":
            imb_dist_val = wasserstein_torch(X=shared_h, t=t_labels)
        elif imb_dist == "mmd":
            imb_dist_val = mmd2_torch(shared_h, t_labels)
        else:
            sys.exit(1)

    imb_dist_loss = imb_dist_w * imb_dist_val
    loss = prpsy_loss + estr_loss + escr_loss \
                 + tr_loss + cr_loss \
                 + cross_tr_loss + cross_cr_loss \
                 + imb_dist_loss
    outcome_loss = tr_loss + cr_loss
    treatment_loss  = prpsy_loss

    return loss, outcome_loss, treatment_loss
