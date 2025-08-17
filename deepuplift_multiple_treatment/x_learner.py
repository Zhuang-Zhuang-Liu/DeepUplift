import torch.nn as nn
import torch
from basemodel import BaseModel
from baseunit import TowerUnit

class Xlearner(BaseModel):
    def __init__(self, input_dim=100,discrete_size_cols=[2,3,4,5,2],embedding_dim=64,
                 base_hidden_dims=[100,100,100,100],output_activation_base=torch.nn.Sigmoid(),base_hidden_func = torch.nn.ELU()
                 ,lift_hidden_dims=[100,100,100,100],lift_activation_base=None,lift_hidden_func = torch.nn.ELU()
                 ,task = 'classification',classi_nums=2, treatment_label_list=[0,1,2,3],treatment_label_weight=[0.25,0.25,0.25,0.25],lift_weight=[0.5,0.5,0.5,0.5],model_type='Tarnet',device='cpu'):
        super(Xlearner, self).__init__()
        self.model_type = model_type
        self.layers = []
        self.treatment_nums = len(treatment_label_list)
        self.treatment_model = nn.ModuleDict()
        self.treatment_uplift_model = nn.ModuleDict()
        self.treatment_label_list = treatment_label_list
        self.treatment_label_weight = treatment_label_weight
        self.lift_weight = lift_weight
        input_dim = input_dim - len(discrete_size_cols) + len(discrete_size_cols)*embedding_dim

        # embedding 
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim).to(device) for size in discrete_size_cols
        ]).to(device)

        # treatment 响应塔
        for treatment_label in self.treatment_label_list:
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
        
        # treatment 伪响应塔
        for treatment_label in self.treatment_label_list:
            if treatment_label != 0:
                self.treatment_uplift_model[str(treatment_label)] = TowerUnit(input_dim = input_dim, 
                    hidden_dims=lift_hidden_dims, 
                    share_output_dim=None, 
                    activation=lift_hidden_func, 
                    use_batch_norm=True, 
                    use_dropout=True, 
                    dropout_rate=0.3, 
                    task='regression', 
                    classi_nums=classi_nums, 
                    output_activation=lift_activation_base,
                    device=device, 
                    use_xavier=True)
            else:
                for treatment_label in self.treatment_label_list:
                    if treatment_label != 0:
                        self.treatment_uplift_model['0_' + str(treatment_label)] = TowerUnit(input_dim = input_dim, 
                            hidden_dims=lift_hidden_dims, 
                            share_output_dim=None, 
                            activation=lift_hidden_func, 
                            use_batch_norm=True, 
                            use_dropout=True, 
                            dropout_rate=0.3, 
                            task='regression', 
                            classi_nums=classi_nums, 
                            output_activation=lift_activation_base,
                            device=device, 
                            use_xavier=True)
        
    
    def forward(self, X, t, X_discrete=None, X_continuous=None):
        embedded = [emb(X_discrete[:, i].long()) for i, emb in enumerate(self.embeddings)]
        X_discrete_emb = torch.cat(embedded, dim=1)  # 拼接所有embedding
        x = torch.cat((X_continuous,X_discrete_emb), dim=1)
        
        # 计算每个treatment的响应
        y_preds = []
        for treatment_label in self.treatment_label_list:
            treatment_pred = self.treatment_model[str(treatment_label)](x).squeeze().unsqueeze(1)
            y_preds.append(self.treatment_label_weight[treatment_label] * treatment_pred)

        # 计算每个treatment的伪响应
        uplift_preds = {}
        for treatment in self.treatment_label_list:
            if treatment != 0:
                uplift_preds[str(treatment)] = self.treatment_label_weight[treatment_label] * self.treatment_uplift_model[str(treatment)](x).squeeze().unsqueeze(1)
            else:
                for treatment_label in self.treatment_label_list:
                    if treatment_label != 0:
                        uplift_preds['0_' + str(treatment_label)] = self.treatment_label_weight[treatment_label] * self.treatment_uplift_model['0_' + str(treatment_label)](x).squeeze().unsqueeze(1)
        
        ate = []
        if not self.training:
            for i in range(len(self.treatment_label_list)):
                if self.treatment_label_list[i] != 0:
                    exp = self.treatment_uplift_model[str(self.treatment_label_list[i])](x).squeeze().unsqueeze(1)
                    base = self.treatment_uplift_model['0_' + str(self.treatment_label_list[i])](x).squeeze().unsqueeze(1)
                    ate.append(self.lift_weight[0]*base + self.lift_weight[i]*exp)
                    
        return torch.cat(ate, dim=1) if len(ate) !=0 else None,[y_preds,uplift_preds],None

def xlearn_loss(y_preds,t, y_true,task='regression',loss_type=None,classi_nums=2, treatment_label_list=None):
    if task is None:
        raise ValueError("task must be 'classification' or 'regression'")
    
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

    t = t.squeeze().unsqueeze(1).long()
    y_true = y_true.squeeze().unsqueeze(1)
    #获取每个样本的所有treatment的响应
    y_pred_all = torch.cat(y_preds[0], dim=1)
    #获取每个样本的真实受到的treatment的响应
    y_pred = torch.gather(y_pred_all, dim=1, index=t.long()).squeeze().unsqueeze(1)
    lift_pred = y_preds[1]
    mask_0 = (t == 0)
    huberloss = nn.SmoothL1Loss()

    # 计算每个treatment的损失,包括响应损失和伪响应损失
    loss_treat = 0
    loss_uplift = 0
    for i in range(len(treatment_label_list)):
        # 响应损失
        mask = (t == treatment_label_list[i])
        loss_treat += criterion(y_pred[mask], y_true[mask])
        if treatment_label_list[i] != 0:
            #对照组的伪响应loss
            loss_uplift += huberloss(lift_pred['0_' + str(treatment_label_list[i])][mask_0], y_pred_all[mask_0.squeeze(),treatment_label_list[i]] - y_true[mask_0])
            #实验组的伪响应loss
            loss_uplift += huberloss(lift_pred[str(treatment_label_list[i])][mask], y_true[mask] - y_pred_all[mask.squeeze(),0])

    loss = loss_treat + loss_uplift
    return loss, loss_treat, loss_uplift




        
        