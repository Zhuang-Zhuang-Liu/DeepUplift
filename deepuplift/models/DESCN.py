import torch
import torch.nn as nn
import sys

from models.BaseModel import BaseModel
from models.BaseUnit import TowerUnit
from utils.matrics import wasserstein_torch, mmd2_torch


class PrpsyNetwork(nn.Module):
    """Propensity network"""
    def __init__(self, base_dim, do_rate):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = TowerUnit(input_dim=base_dim, 
                                 hidden_dims=[base_dim, base_dim],
                                 share_output_dim=base_dim,
                                 activation=nn.ELU(),
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=do_rate,
                                 task='share')
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, do_rate):
        super(Mu0Network, self).__init__()
        self.baseModel = TowerUnit(input_dim=base_dim, 
                                 hidden_dims=[base_dim, base_dim],
                                 share_output_dim=base_dim,
                                 activation=nn.ELU(),
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=do_rate,
                                 task='share')
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, do_rate):
        super(Mu1Network, self).__init__()
        self.baseModel = TowerUnit(input_dim=base_dim, 
                                 hidden_dims=[base_dim, base_dim],
                                 share_output_dim=base_dim,
                                 activation=nn.ELU(),
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=do_rate,
                                 task='share')
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class TauNetwork(nn.Module):
    """Pseudo tau network"""
    def __init__(self, base_dim, do_rate):
        super(TauNetwork, self).__init__()
        self.baseModel = TowerUnit(input_dim=base_dim, 
                                 hidden_dims=[base_dim, base_dim],
                                 share_output_dim=base_dim,
                                 activation=nn.ELU(),
                                 use_batch_norm=True,
                                 use_dropout=True,
                                 dropout_rate=do_rate,
                                 task='share')
        self.logitLayer = nn.Linear(base_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        return tau_logit


class ESX(BaseModel):
    """ESX model: Predicts treatment effects through shared networks and multiple sub-networks.
    Contains propensity network, treatment and control networks, and treatment effect network.
    """
    def __init__(self, input_dim, share_dim, base_dim, task="classification",
                do_rate=0.2, use_batch_norm1d=True, normalization="divide", device='cpu'):
        super().__init__()
        
        # Validate task type
        if task != "classification":
            raise ValueError(f"Unsupported task type: '{task}'. This model only supports 'classification' task.")
        
        # Shared network
        self.share_network = TowerUnit(input_dim=input_dim, 
                                     hidden_dims=[share_dim, share_dim],
                                     share_output_dim=base_dim,
                                     activation=nn.ELU(),
                                     use_batch_norm=use_batch_norm1d,
                                     use_dropout=True,
                                     dropout_rate=do_rate,
                                     task='share',
                                     device=device)
        # Sub-networks
        self.prpsy_network = PrpsyNetwork(base_dim, do_rate).to(device)
        self.mu1_network = Mu1Network(base_dim, do_rate).to(device)
        self.mu0_network = Mu0Network(base_dim, do_rate).to(device)
        self.tau_network = TauNetwork(base_dim, do_rate).to(device)
        
        self.device = device
        self.to(device)

    def forward(self, x, tr=None):
        """
        Args:
            x: Input features
            tr: Treatment indicator variable (only for EFIN interface alignment, no actual use)
        Returns:
            tuple: Tuple containing multiple prediction results
                - p_prpsy: Propensity score
                - [p_h1, p_h0]: Treatment and control group prediction results
                - p_estr: Expected score for treatment group
                - p_escr: Expected score for control group
                - tau_logit: Logit value for treatment effect
                - mu1_logit: Logit value for treatment group
                - mu0_logit: Logit value for control group
                - p_prpsy_logit: Logit value for propensity score
                - shared_h: Output of shared network
        """
        # Shared network output
        shared_h = self.share_network(x)

        # Propensity score
        p_prpsy_logit = self.prpsy_network(shared_h)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # Treatment and control group predictions
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)
        p_mu1 = torch.sigmoid(mu1_logit)
        p_mu0 = torch.sigmoid(mu0_logit)
        p_h1 = p_mu1  # Treatment group prediction
        p_h0 = p_mu0  # Control group prediction

        # Treatment effect
        tau_logit = self.tau_network(shared_h)

        # Calculate expected scores
        p_estr = torch.mul(p_prpsy, p_h1)  # Treatment group expectation
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)  # Control group expectation

        return p_prpsy, [p_h1, p_h0], p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy_logit, shared_h



#######################################
def esx_loss(   p_prpsy,y_preds,
                t_labels, y_labels, 
                p_estr, p_escr, tau_logit, mu1_logit, mu0_logit,p_prpsy_logit, shared_h,  # *eps
                # , e_labels,
                prpsy_w=1.0, escvr1_w=1.0, escvr0_w=1.0, h1_w=1.0, h0_w=1.0, 
                mu1hat_w=1.0, mu0hat_w=1.0,imb_dist_w=0.0, imb_dist="wass",task='classification'):
    """"
    t_pred = p_prpsy_logit
    y_preds = [p_h1, p_h0] 
    """
    # Validate task type
    if task != "classification":
        raise ValueError(f"Unsupported task type: '{task}'. This model only supports 'classification' task.")
        

    p_h1 = y_preds[1]
    p_h0 = y_preds[0]

    p_t = 0.5
    sample_weight = torch.ones_like(t_labels) 
    # Define loss function
    loss_w_fn = nn.BCELoss(weight=sample_weight)
    loss_fn = nn.BCELoss()
    loss_w_with_logit_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(1 / (2 * p_t)))  # For propensity loss

    # Loss for propensity: cross-entropy
    t_labels = t_labels.squeeze(1)
    prpsy_loss = prpsy_w * loss_w_with_logit_fn(p_prpsy_logit.squeeze(1), t_labels.float()  )     
    
    # Loss for ESTR, ESCR
    # p_estr: torch.Size([64, 1]) y_labels: torch.Size([64, 1]) t_labels: torch.Size([64])
    y_labels = y_labels.squeeze(1)
    t_labels = t_labels.float()
    loss_w_fn = nn.BCELoss(weight=sample_weight)
    estr_loss = escvr1_w * loss_w_fn(p_estr, (y_labels * t_labels).float().unsqueeze(1))
    escr_loss = escvr0_w * loss_w_fn(p_escr,(y_labels * (1 - t_labels)).float().unsqueeze(1))

    # Loss for TR, CR
    loss_fn = nn.BCELoss()

    tr_loss = h1_w * loss_fn(p_h1,y_labels.unsqueeze(1))  # * (1 / (2 * p_t))
    cr_loss = h0_w * loss_fn(p_h0,y_labels.unsqueeze(1))  # * (1 / (2 * (1 - p_t)))
    # Loss for cross TR: mu1_prime, cross CR: mu0_prime
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
