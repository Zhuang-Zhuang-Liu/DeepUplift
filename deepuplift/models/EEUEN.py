import torch
import torch.nn.functional as F

from models.BaseModel import BaseModel
from models.BaseUnit import TowerUnit


class EEUEN(BaseModel):
    def __init__(self, input_dim, hc_dim=64, hu_dim=64, he_dim=64, 
                 share_hidden_dims=[64,64,64,64], base_hidden_dims=[100,100,100,100],
                 share_hidden_func=torch.nn.ELU(), base_hidden_func=torch.nn.ELU(),
                 is_self=True, l2_reg=0.0, task='regression'):
        super().__init__()
        
        # Control Network
        self.control_net = TowerUnit(
            input_dim=input_dim,
            hidden_dims=share_hidden_dims,
            share_output_dim=hc_dim,
            activation=share_hidden_func,
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.2,
            task='share'
        )
        self.c_logit = TowerUnit(
            input_dim=hc_dim,
            hidden_dims=base_hidden_dims,
            activation=base_hidden_func,
            task=task,
            classi_nums=1
        )
        self.c_tau = TowerUnit(
            input_dim=hc_dim,
            hidden_dims=base_hidden_dims,
            activation=base_hidden_func,
            task=task,
            classi_nums=1
        )

        # Treat Exposure Network
        self.exposure_net = TowerUnit(
            input_dim=input_dim,
            hidden_dims=share_hidden_dims,
            share_output_dim=he_dim,
            activation=share_hidden_func,
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.2,
            task='share'
        )
        self.e_logit = TowerUnit(
            input_dim=he_dim,
            hidden_dims=base_hidden_dims,
            activation=base_hidden_func,
            task=task,
            classi_nums=1
        )

        # Uplift Network
        self.uplift_net = TowerUnit(
            input_dim=input_dim,
            hidden_dims=share_hidden_dims,
            share_output_dim=hu_dim,
            activation=share_hidden_func,
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.2,
            task='share'
        )
        self.t_logit = TowerUnit(
            input_dim=hu_dim,
            hidden_dims=base_hidden_dims,
            activation=base_hidden_func,
            task=task,
            classi_nums=1
        )
        self.u_tau = TowerUnit(
            input_dim=hu_dim,
            hidden_dims=base_hidden_dims,
            activation=base_hidden_func,
            task=task,
            classi_nums=1
        )

    def forward(self, x, tr=None):
        # Control Network
        c_hidden = self.control_net(x)
        c_logit = self.c_logit(c_hidden)
        c_tau = self.c_tau(c_hidden)

        # Treat Exposure Network
        e_hidden = self.exposure_net(x)
        e_logit = self.e_logit(e_hidden)

        # Uplift Network
        u_hidden = self.uplift_net(x)
        t_logit = self.t_logit(u_hidden)
        u_tau = self.u_tau(u_hidden)

        # Calculate final predictions - using sigmoid activation function
        uc = torch.sigmoid(c_logit)
        ut = torch.sigmoid(t_logit)

        y_preds = [uc, ut]  # 控制组和处理组的预测
        return t_logit, y_preds, e_logit




def eeuen_loss(t_pred, y_preds, tr, y1, e_logit=None, alpha=1.0, beta=1.0, task='regression'):
    """EEUEN loss function
    
    Args:
        t_pred: Treatment prediction logit output
        y_preds: [uc, ut] Control and treatment group predictions
        tr: Actual treatment assignment
        y1: Actual outcome
        e_logit: Exposure prediction logit output
        alpha: Weight for treatment loss
        beta: Weight for exposure loss
        task: 'regression' or 'classification'
    """
    # Treatment prediction loss
    tr_labels = tr.long().squeeze()
    treatment_loss = F.binary_cross_entropy_with_logits(t_pred, tr.float())
    
    # Outcome prediction loss
    uc, ut = y_preds
    if task == 'regression':
        outcome_loss = torch.mean((1 - tr) * torch.square(y1 - uc) + 
                                tr * torch.square(y1 - ut))
    else:  # classification
        outcome_loss = torch.mean((1 - tr) * F.binary_cross_entropy(uc, y1) +
                                tr * F.binary_cross_entropy(ut, y1))
    
    # Exposure loss
    if e_logit is not None:
        exposure_loss = torch.mean(tr * F.binary_cross_entropy_with_logits(e_logit, y1))
        total_loss = outcome_loss + alpha * treatment_loss + beta * exposure_loss
    else:
        total_loss = outcome_loss + alpha * treatment_loss
        
    return total_loss, outcome_loss, treatment_loss 