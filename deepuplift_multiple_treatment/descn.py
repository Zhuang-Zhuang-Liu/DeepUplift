import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit

class Descn(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,share_dim=6,
                 share_hidden_dims =[64,64,64,64,64],
                 base_hidden_dims=[100,100,100,100],output_activation_base=torch.nn.Sigmoid(),
                 ipw_hidden_dims=[64,64,64,64,64],output_activation_ipw=torch.nn.Sigmoid(),
                 pseudo_hidden_dims=[64,64,64,64,64],output_activation_pseudo=torch.nn.Sigmoid(),
                 share_hidden_func = torch.nn.ELU(),base_hidden_func = torch.nn.ELU(), ipw_hidden_func = torch.nn.ELU(),pseudo_hidden_func = torch.nn.ELU(),
                 task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],model_type='ESN_Tarnet',device='cpu'):
        super(Descn, self).__init__()
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.treatment_model = nn.ModuleDict()
        self.pseudo_treatment_model = nn.ModuleDict()
        self.treatment_label_list = treatment_label_list
        input_dim = input_dim - len(discrete_size_cols) + len(discrete_size_cols)*embedding_dim

        # embedding 
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim).to(device) for size in discrete_size_cols
        ])
        
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

        # treatment tower
        for treatment_label in self.treatment_label_list:
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

        # Pseudo Treatment Effect tower
        for treatment_label in self.treatment_label_list[1:]:
            self.pseudo_treatment_model[str(treatment_label)] = TowerUnit(input_dim = input_dim, 
                    hidden_dims=pseudo_hidden_dims, 
                    share_output_dim=None, 
                    activation=pseudo_hidden_func, 
                    use_batch_norm=True, 
                    use_dropout=True, 
                    dropout_rate=0.3, 
                    task='regression', 
                    classi_nums=None, 
                    output_activation=output_activation_pseudo,
                    device=device, 
                    use_xavier=True)

    
    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)
        share_out = self.share_tower(x)
        
        #计算每个样本划入各个treatment的概率 -- loss ipw
        ipw = self.ipw_tower(x)
        treatment_proba = torch.softmax(ipw, dim=1)

        pre = []
        ite = []
        origin_pre = []
        pseudo = []

        for treatment_label in self.treatment_label_list:
            #计算每个样本在指定treatment下的目标响应概率
            predcit_pro = self.treatment_model[str(treatment_label)](share_out).squeeze().unsqueeze(1)
            origin_pre.append(predcit_pro)
            if treatment_label == 0:
                    base_predcit_pro = predcit_pro

            #计算每个样本划入指定treatment的概率
            ipw_score = treatment_proba[:,treatment_label].squeeze().unsqueeze(1)

            #计算每个样本指定treatment下的伪响应概率
            if treatment_label !=0 :
                pseudo_treatment = self.pseudo_treatment_model[str(treatment_label)](x).squeeze().unsqueeze(1)
                #计算每个样本指定treatment下的cross伪响应概率 -- loss crossXr
                pseudo.append([torch.sigmoid(torch.logit(predcit_pro, eps=1e-6) - torch.logit(pseudo_treatment, eps=1e-6)),
                               torch.sigmoid(torch.logit(base_predcit_pro, eps=1e-6) + torch.logit(pseudo_treatment, eps=1e-6))])
                
            #每个样本指定treatment下的目标响应概率与划入指定treatment的概率相乘 -- loss esXr
            pre.append(predcit_pro*ipw_score)

            #不同treatment下ITE输出
            if not self.training:
                if treatment_label != 0:
                    ite.append((base_predcit_pro + pseudo_treatment) - (predcit_pro - pseudo_treatment))

        # pre [(batch_size,1),(batch_size,1),...]
        # pseudo [[(batch_size,1),(batch_size,1)],...]
        # ipw (batch_size,treatment_nums)
        return torch.cat(ite, dim=1),[pre,pseudo,ipw],None

def descn_loss(y_preds,t, y_true,task='classification',loss_type=None,classi_nums=2, treatment_label_list=None):
    if task is None:
        raise ValueError("task must be 'classification'")

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    pre = y_preds[0]
    pseudo = y_preds[1]
    ipw = y_preds[2]
    mask_0 = (t == 0)

    y_pred = torch.gather(torch.cat(pre, dim=1), dim=1, index=t.long()).squeeze().unsqueeze(1)
    y_true_dict = {}
    y_pred_dict = {}
    for treatment in treatment_label_list:
        mask = (t == treatment)
        y_true_dict[treatment] = y_true[mask]
        y_pred_dict[treatment] = y_pred[mask]

    # loss ipw
    if len(treatment_label_list) == 2:
        ipw_criterion = nn.BCEWithLogitsLoss()
        loss_ipw = ipw_criterion(ipw, t)

    if len(treatment_label_list) > 2: 
        ipw_criterion = nn.CrossEntropyLoss()
        loss_ipw = ipw_criterion(ipw, t.squeeze())

    # loss esXr
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

    #loss crossXr
    loss_pseudo = 0
    huberloss = nn.SmoothL1Loss()
    for treatment in treatment_label_list[1:]:
        mask = (t == treatment)
        loss_pseudo += huberloss(pseudo[treatment-1][0][mask_0],y_true_dict[0])
        loss_pseudo += huberloss(pseudo[treatment-1][1][mask],y_true_dict[treatment])

    loss = loss_treat + loss_ipw + loss_pseudo
    return loss, [loss_treat, loss_ipw, loss_pseudo],None



        
        