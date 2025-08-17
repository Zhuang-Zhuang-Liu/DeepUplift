import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit

class ESN_Tarnet(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,share_dim=6,
                 share_hidden_dims =[64,64,64,64,64],
                 base_hidden_dims=[100,100,100,100],output_activation_base=torch.nn.Sigmoid(),
                 ipw_hidden_dims=[64,64,64,64,64],output_activation_ipw=torch.nn.Sigmoid(),
                 share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), ipw_hidden_func = torch.nn.ELU(),
                 task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],model_type='ESN_Tarnet',device='cpu'):
        super(ESN_Tarnet, self).__init__()
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
        
        # share tower
        self.share_tower = TowerUnit(input_dim = input_dim, 
                 hidden_dims=share_hidden_dims, 
                 share_output_dim=share_dim, 
                 activation=share_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task='share', 
                 classi_nums=None, 
                 device=device, 
                 use_xavier=True)

        for treatment_label in self.treatment_label_list:
            # treatment tower
            self.treatment_model[str(treatment_label)] = TowerUnit(input_dim = share_dim, 
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
        
        # ipw tower
        self.ipw_tower = TowerUnit(input_dim = input_dim, 
                 hidden_dims=ipw_hidden_dims, 
                 share_output_dim=None, 
                 activation=ipw_hidden_func, 
                 use_batch_norm=True, 
                 use_dropout=True, 
                 dropout_rate=0.3, 
                 task=task, 
                 classi_nums=self.treatment_nums, 
                 output_activation=output_activation_ipw,
                 device=device, 
                 use_xavier=True)

    
    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)
        # print(f'输入{x}')

        share_out = self.share_tower(x)
        # print(f'share{share_out}')
        ipw = self.ipw_tower(x)
        # print(f'ipw{ipw}')
        treatment_proba = torch.softmax(ipw, dim=1)

        pre = []
        ate = []

        base_predcit_pro = self.treatment_model['0'](share_out).squeeze().unsqueeze(1)
        base_ipw = treatment_proba[:,0].squeeze().unsqueeze(1)

        for treatment_label in self.treatment_label_list:
            predcit_pro = self.treatment_model[str(treatment_label)](share_out).squeeze().unsqueeze(1)
            ipw_score = treatment_proba[:,treatment_label].squeeze().unsqueeze(1)
            # print(f'predcit_pro_{predcit_pro}')
            # print(f'ipw_score{ipw_score}')
            pre.append(predcit_pro*ipw_score)
            if treatment_label != 0:
                ate.append(predcit_pro -base_predcit_pro)
        pre.append(ipw)
        return torch.cat(ate, dim=1),pre,None

def esn_tarnet_loss(y_preds,t, y_true,task='classification',loss_type=None,classi_nums=2, treatment_label_list=None):
    if task is None:
        raise ValueError("task must be 'classification'")

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    y_pred = torch.gather(torch.cat(y_preds[:-1], dim=1), dim=1, index=t.long()).squeeze().unsqueeze(1)

    ipw = y_preds[-1]

    y_true_dict = {}
    y_pred_dict = {}
    for treatment in treatment_label_list:
        mask = (t == treatment)
        y_true_dict[treatment] = y_true[mask]
        y_pred_dict[treatment] = y_pred[mask]

    # 计算ipw损失
    if len(treatment_label_list) == 2:
        ipw_criterion = nn.BCEWithLogitsLoss()
        loss_ipw = ipw_criterion(ipw, t)

    if len(treatment_label_list) > 2: 
        ipw_criterion = nn.CrossEntropyLoss()
        loss_ipw = ipw_criterion(ipw, t.squeeze())

    # 计算每个treatment的损失
    if task == 'classification':
        if loss_type == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        elif loss_type =='BCELoss':
            criterion = nn.BCELoss()
        else:
            raise ValueError("loss_type must be 'BCEWithLogitsLoss' or 'BCELoss'")
    else:
        raise ValueError("task must be 'classification'")
    
    loss_treat = 0
    for treatment in treatment_label_list:
        loss_treat += criterion(y_pred_dict[treatment], y_true_dict[treatment])
    loss = loss_treat + loss_ipw
    return loss, loss_treat, loss_ipw



        
        