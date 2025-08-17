import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit

class Tlearner(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,
                 base_hidden_dims=[100,100,100,100],output_activation_base=torch.nn.Sigmoid(),
                 base_hidden_func = torch.nn.ELU(),task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],model_type='Tarnet',device='cpu'):
        super(Tlearner, self).__init__()
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.treatment_model = nn.ModuleDict()
        self.treatment_label_list = treatment_label_list
        input_dim = input_dim - len(discrete_size_cols) + len(discrete_size_cols)*embedding_dim

        # embedding 
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim).to(device) for size in discrete_size_cols
        ]).to(device)

        for treatment_label in self.treatment_label_list:
            # treatment tower
            self.treatment_model[str(treatment_label)] = TowerUnit(input_dim = input_dim, 
                 hidden_dims=base_hidden_dims, 
                 share_output_dim=None, 
                 activation=base_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task=task, 
                 classi_nums=classi_nums, 
                 output_activation=output_activation_base,
                 device=device, 
                 use_xavier=True)
    
    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)

        pre = []
        ate = []

        base_predcit_pro = self.treatment_model['0'](x).squeeze().unsqueeze(1)

        for treatment_label in self.treatment_label_list:
            predcit_pro = self.treatment_model[str(treatment_label)](x).squeeze().unsqueeze(1)
            pre.append(predcit_pro)
            if treatment_label != 0:
                ate.append(predcit_pro -base_predcit_pro)
        return torch.cat(ate, dim=1),pre,None

def tlearn_loss(y_preds,t, y_true,task='regression',loss_type=None,classi_nums=2, treatment_label_list=None):
    if task is None:
        raise ValueError("task must be 'classification' or 'regression'")

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    y_pred = torch.gather(torch.cat(y_preds, dim=1), dim=1, index=t.long()).squeeze().unsqueeze(1)

    y_true_dict = {}
    y_pred_dict = {}
    for treatment in treatment_label_list:
        mask = (t == treatment)
        y_true_dict[treatment] = y_true[mask]
        y_pred_dict[treatment] = y_pred[mask]

    # 计算每个treatment的损失
    if task == 'classification':
        if loss_type == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_type =='BCELoss':
            criterion = nn.BCELoss()
        else:
            raise ValueError("loss_type must be 'BCEWithLogitsLoss' or 'BCELoss'")
    elif task == 'regression':
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type =='huberloss':
            criterion = nn.SmoothL1Loss() 
        else:
            raise ValueError("loss_type must be 'mse' or 'huberloss'")
    else:
        raise ValueError("task must be 'classification' or'regression'")
    
    loss_treat = 0
    for treatment in treatment_label_list:
        loss_treat += criterion(y_pred_dict[treatment], y_true_dict[treatment])
    loss = loss_treat
    return loss, loss_treat, None



        
        