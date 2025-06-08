import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseModel, BaseLoss
from models.BaseUnit import TowerUnit


class DeepFM(nn.Module):
    def __init__(self, feat_size, embedding_size=4, hidden_dims=[64,32], num_continuous=2):
        super().__init__()
        self.feat_size = feat_size
        self.embedding_size = embedding_size
        self.num_continuous = num_continuous
        self.num_discrete = feat_size - num_continuous
        
        # Discrete feature embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(2, embedding_size) for _ in range(self.num_discrete)
        ])
        
        # Continuous feature processing layer
        self.cont_linear = nn.Linear(num_continuous, embedding_size * num_continuous)
        
        # FM linear part
        self.fm_linear = nn.Linear(feat_size, 1)
        
        # DNN part
        dnn_input_dim = (self.num_discrete + num_continuous) * embedding_size
        layers = []
        for dim in hidden_dims:
            layers.extend([nn.Linear(dnn_input_dim, dim), nn.ReLU()])
            dnn_input_dim = dim
        self.dnn = nn.Sequential(*layers)
        self.dnn_fc = nn.Linear(hidden_dims[-1], 1)  # Output scalar for FM part
        
        # Final output dimension (FM scalar + DNN features)
        self.output_dim = 1 + hidden_dims[-1]

    def forward(self, x):
        # Separate discrete and continuous features
        discrete_feats = x[:, :self.num_discrete]  
        cont_feats = x[:, self.num_discrete:]
        
        # Process discrete features
        if self.num_discrete > 0:
            disc_embeds = [emb(discrete_feats[:,i].long()) for i, emb in enumerate(self.embeddings)]
            disc_embeds = torch.stack(disc_embeds, dim=1)  # [B, num_disc, emb]
        else:
            disc_embeds = torch.empty(x.size(0), 0, self.embedding_size, device=x.device)
        
        # Process continuous features
        cont_embeds = self.cont_linear(cont_feats)  # [B, num_cont*emb]
        cont_embeds = cont_embeds.view(-1, self.num_continuous, self.embedding_size)  # [B, num_cont, emb]
        
        # Merge all embeddings
        all_embeds = torch.cat([disc_embeds, cont_embeds], dim=1)  # [B, feat_size, emb]
        
        # FM linear part - first order
        fm_first = self.fm_linear(x)  # [B, 1]

        # FM crossing part - second order
        num_embeds = all_embeds.size(1)
        crossing_terms = []
        for i in range(num_embeds):
            for j in range(i + 1, num_embeds):
                crossing_term = torch.sum(all_embeds[:, i] * all_embeds[:, j], dim=1, keepdim=True)
                crossing_terms.append(crossing_term)
        fm_second = torch.sum(torch.cat(crossing_terms, dim=1), dim=1, keepdim=True)  # [B, 1]
        
        # DNN part
        dnn_input = all_embeds.view(x.size(0), -1)  # [B, (disc+cont)*emb]
        dnn_out = self.dnn(dnn_input)               # [B, hidden_dim]
        
        # Combine FM and DNN outputs
        combined = torch.cat([fm_first + fm_second, dnn_out], dim=1)  # [B, 1 + hidden_dim]
        return combined

class DragonDeepFM(BaseModel):
    def __init__(self, input_dim, embedding_size=4,share_dim=32,num_continuous=2, 
                 base_hidden_dims=[100,100,100,100],base_hidden_func = torch.nn.ELU(), 
                 task='classification', num_treatments=2):
        super().__init__()
        # Shared layer (output dimension guaranteed by DeepFM as 1+hidden_dim[-1])
        self.shared_layer = DeepFM(input_dim,embedding_size,[64, share_dim-1],num_continuous)
        # Propensity head*1: for predicting which treatment the current sample belongs to (multi-classification)
        self.treatment_head = TowerUnit(input_dim= share_dim,
                                        hidden_dims= base_hidden_dims,activation= base_hidden_func, 
                                        use_batch_norm= False,
                                        task = 'classification',classi_nums = num_treatments )
        # Outcome head*n: for predicting the outcome of each treatment (regression or binary classification)
        self.outcome_heads = nn.ModuleList([
                             TowerUnit(input_dim= share_dim,
                                        hidden_dims= base_hidden_dims,activation= base_hidden_func, 
                                        use_batch_norm= False,
                                        task = task,classi_nums = 2 
                            )  for _ in range(num_treatments)
                                            ])

    def forward(self, x, tr=None):
        shared = self.shared_layer(x)  
        t_pred = self.treatment_head(shared)  
        y_preds = [head(shared) for head in self.outcome_heads] 
        #y_preds = [y[:, 1] for y in y_preds]  # Take treatment=1 probability as sigmoid output
        return t_pred,y_preds



def dragonnet_loss(t_pred, y_preds,t_true, y_true,phi_x=None,alpha=1.0,beta=1.0,tarreg=True, task='regression'):
    return BaseLoss(t_pred=t_pred, y_preds=y_preds, 
                       t_true=t_true, y_true=y_true,
                       loss_type='dragonnet',
                       alpha=alpha, beta=beta, tarreg=tarreg, task=task)