import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence
from models.BaseModel import BaseModel
from models.BaseUnit import TowerUnit


class CEVAE(BaseModel):
    def __init__(self, input_dim, h_dim, x_repr_dim, z_repr_dim, rep_norm=True, is_self=True, act_type="elu", task='regression'):
        super().__init__()
        # Validate task type
        if task not in ['regression']:
            raise ValueError(f"Unsupported task type: '{task}'. This model supports 'regression' tasks.")

        self.h_dim = h_dim
        self.x_repr_dim = x_repr_dim
        self.z_repr_dim = z_repr_dim
        self.rep_norm = rep_norm
        self.is_self = is_self
        self.act_type = act_type
        
        # R(X) Base Network
        self.rep_net = TowerUnit(
            input_dim=input_dim,
            hidden_dims=[h_dim],
            share_output_dim=x_repr_dim,
            activation=nn.ELU() if act_type == "elu" else nn.ReLU(),
            task='share'
        )
        
        # P(X|Z) Network
        self.p_x_z = TowerUnit(
            input_dim=z_repr_dim,
            hidden_dims=[h_dim, h_dim],
            share_output_dim=h_dim,
            activation=nn.ELU() if act_type == "elu" else nn.ReLU(),
            task='share'
        )
        self.p_x_z_mu = nn.Linear(h_dim, x_repr_dim)
        self.p_x_z_sigma = nn.Sequential(
            nn.Linear(h_dim, x_repr_dim),
            nn.Softplus()
        )
        
        # P(T|Z) Network
        self.p_t_z = TowerUnit(
            input_dim=z_repr_dim,
            hidden_dims=[h_dim],
            activation=nn.ELU() if act_type == "elu" else nn.ReLU(),
            task='regression'
        )
        
        # P(Y|Z,T) Network
        hidden_dims = [h_dim, h_dim//2, h_dim//4]
        if is_self:
            hidden_dims.append(h_dim//8)
            
        self.p_y_zt_t0 = TowerUnit(
            input_dim=z_repr_dim,
            hidden_dims=hidden_dims,
            activation=nn.ELU() if act_type == "elu" else nn.ReLU(),
            task='regression'
        )
        
        self.p_y_zt_t1 = TowerUnit(
            input_dim=z_repr_dim,
            hidden_dims=hidden_dims,
            activation=nn.ELU() if act_type == "elu" else nn.ReLU(),
            task='regression'
        )
        
        # Q(T|X) Network
        self.q_t_x = TowerUnit(
            input_dim=x_repr_dim,
            hidden_dims=[h_dim, h_dim//2],
            activation=nn.ELU() if act_type == "elu" else nn.ReLU(),
            task='regression'
        )
        
        # Q(Y|X,T) Network
        self.q_y_xt = TowerUnit(
            input_dim=x_repr_dim,
            hidden_dims=hidden_dims,
            activation=nn.ELU() if act_type == "elu" else nn.ReLU(),
            task='share'
        )
        self.q_y_xt_t0_mu = nn.Linear(h_dim//4 if not is_self else h_dim//8, 1)
        self.q_y_xt_t1_mu = nn.Linear(h_dim//4 if not is_self else h_dim//8, 1)
        
        # Q(Z|T,Y,X) Network
        hidden_dims_z = [h_dim, h_dim]
        if is_self:
            hidden_dims_z.append(h_dim)
            
        self.q_z_tyx = TowerUnit(
            input_dim=x_repr_dim + 1,  # +1 for y
            hidden_dims=hidden_dims_z,
            activation=nn.ELU() if act_type == "elu" else nn.ReLU(),
            task='share'
        )
        
        self.q_z_tyx_t0_mu = nn.Linear(h_dim, z_repr_dim)
        self.q_z_tyx_t1_mu = nn.Linear(h_dim, z_repr_dim)
        self.q_z_tyx_t0_sigma = nn.Sequential(
            nn.Linear(h_dim, z_repr_dim),
            nn.Softplus()
        )
        self.q_z_tyx_t1_sigma = nn.Sequential(
            nn.Linear(h_dim, z_repr_dim),
            nn.Softplus()
        )

    def rep_x4_emb(self, x):
        x_repr = self.rep_net(x)
        if self.rep_norm:
            x_repr_norm = x_repr / torch.sqrt(torch.sum(x_repr**2, dim=1, keepdim=True) + 1e-8)
        else:
            x_repr_norm = x_repr
        return x_repr_norm
    
    def rep_x(self, x):
        if self.rep_norm:
            x_repr_norm = x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + 1e-8)
        else:
            x_repr_norm = x
        return x_repr_norm
    
    def p_x_z_forward(self, z):
        h = self.p_x_z(z)
        mu = self.p_x_z_mu(h)
        sigma = self.p_x_z_sigma(h)
        if self.is_self:
            sigma = sigma + 1e-4
        else:
            sigma = torch.maximum(sigma, torch.tensor(0.4))
        return Normal(mu, sigma)
    
    def p_t_z_forward(self, z):
        logits = self.p_t_z(z)
        p = torch.sigmoid(logits)
        return Bernoulli(probs=torch.maximum(p, torch.tensor(1e-4))), logits, p
    
    def p_y_zt_forward(self, z, t):
        h_t0 = self.p_y_zt_t0(z)
        mu_t0 = self.p_y_zt_t0_mu(h_t0)
        
        h_t1 = self.p_y_zt_t1(z)
        mu_t1 = self.p_y_zt_t1_mu(h_t1)
        
        mu = (1 - t) * mu_t0 + t * mu_t1
        return Normal(mu, torch.ones_like(mu))
    
    def q_t_x_forward(self, x):
        logits = self.q_t_x(x)
        p = torch.sigmoid(logits)
        return Bernoulli(probs=torch.maximum(p, torch.tensor(1e-4))), logits, p
    
    def q_y_xt_forward(self, x, t):
        h = self.q_y_xt(x)
        mu_t0 = self.q_y_xt_t0_mu(h)
        mu_t1 = self.q_y_xt_t1_mu(h)
        mu = (1 - t) * mu_t0 + t * mu_t1
        return Normal(mu, torch.ones_like(mu))
    
    def q_z_tyx_forward(self, x, y, t):
        xy = torch.cat([x, y], dim=1)
        h = self.q_z_tyx(xy)
        
        mu_t0 = self.q_z_tyx_t0_mu(h)
        mu_t1 = self.q_z_tyx_t1_mu(h)
        sigma_t0 = self.q_z_tyx_t0_sigma(h)
        sigma_t1 = self.q_z_tyx_t1_sigma(h)
        
        if self.is_self:
            sigma = (1 - t) * sigma_t0 + t * sigma_t1 + 1e-4
        else:
            sigma = torch.maximum((1 - t) * sigma_t0 + t * sigma_t1, torch.tensor(0.4))
        
        mu = (1 - t) * mu_t0 + t * mu_t1
        return Normal(mu, sigma)
    
    def z_prior(self, z_latent):
        return Normal(torch.zeros_like(z_latent), torch.ones_like(z_latent))
    
    def forward(self, x, t):
        # Get X representation
        if self.is_self:
            x_repr = self.rep_x4_emb(x)
            x_repr = x_repr.detach()
        else:
            x_repr = self.rep_x(x)
        
        # Predict y and t based on x
        t_x_bin, tx_logit, tx_p = self.q_t_x_forward(x_repr)
        y_xt_prob = self.q_y_xt_forward(x_repr, t)
        
        # Get prior latent z
        z_latent = self.q_z_tyx_forward(x_repr, y_xt_prob.sample(), t)
        sample_z = z_latent.rsample()
        z_latent_prior = self.z_prior(sample_z)
        
        # Get x, y and t described by giving z
        x_z_repr = self.p_x_z_forward(sample_z)
        t_z_bin, _, _ = self.p_t_z_forward(sample_z)
        y_zt_prob = self.p_y_zt_forward(sample_z, t)
        
        return t_x_bin, [y_xt_prob, y_zt_prob], [x_z_repr, t_z_bin, z_latent, z_latent_prior]




def cevae_loss(t_pred, y_preds, t_true, y_true, eps,task='regression'):
    """
    Parameters
    ----------
    t_pred: Bernoulli
        Predicted treatment distribution
    y_preds: [Normal, Normal]
        [y_xt_prob, y_zt_prob] - Predicted outcome distributions
    t_true: torch.Tensor
        Actual treatment variable
    y_true: torch.Tensor
        Actual outcome variable
    eps: [Normal, Bernoulli, Normal, Normal]
        [x_z_repr, t_z_bin, z_latent, z_latent_prior] - Additional distributions for loss calculation
    
    Returns
    -------
    loss: torch.Tensor
        Total loss
    outcome_loss: torch.Tensor
        Outcome prediction loss
    treatment_loss: torch.Tensor
        Treatment prediction loss
    """
    # Validate task type
    if task not in ['regression']:
        raise ValueError(f"Unsupported task type: '{task}'. This model supports 'regression' tasks.")
    
    x_z_repr, t_z_bin, z_latent, z_latent_prior = eps
    y_xt_prob, y_zt_prob = y_preds
    
    # Calculate losses
    loss_x = torch.sum(x_z_repr.log_prob(y_true), dim=1)
    loss_x = torch.where(torch.isnan(loss_x), -1e-8 * torch.ones_like(loss_x), loss_x)
    
    loss_y = torch.sum(y_zt_prob.log_prob(y_true), dim=1)
    loss_y = torch.where(torch.isnan(loss_y), -1e-8 * torch.ones_like(loss_y), loss_y)
    
    loss_t = torch.sum(t_z_bin.log_prob(t_true), dim=1)
    loss_t = torch.where(torch.isnan(loss_t), -1e-8 * torch.ones_like(loss_t), loss_t)
    
    loss_z = torch.sum(-kl_divergence(z_latent, z_latent_prior), dim=1)
    loss_z = torch.where(torch.isnan(loss_z), -1e-8 * torch.ones_like(loss_z), loss_z)
    
    loss_pt = torch.sum(t_pred.log_prob(t_true), dim=1)
    loss_pt = torch.where(torch.isnan(loss_pt), -1e-8 * torch.ones_like(loss_pt), loss_pt)
    
    loss_py = torch.sum(y_xt_prob.log_prob(y_true), dim=1)
    loss_py = torch.where(torch.isnan(loss_py), -1e-8 * torch.ones_like(loss_py), loss_py)
    
    loss_elbo = loss_x + loss_y + loss_t + loss_z
    loss_pred = loss_pt + loss_py
    
    total_loss = torch.mean(-loss_elbo - loss_pred)
    outcome_loss = torch.mean(-loss_y - loss_py)
    treatment_loss = torch.mean(-loss_t - loss_pt)
    
    return total_loss, outcome_loss, treatment_loss 