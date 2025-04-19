import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseModel
from models.BaseUnit import TowerUnit


class DeepFM(nn.Module):
    def __init__(self, feat_size, embedding_size=4, hidden_dims=[64,32], num_continuous=2):
        super().__init__()
        self.feat_size = feat_size
        self.embedding_size = embedding_size
        self.num_continuous = num_continuous
        self.num_discrete = feat_size - num_continuous
        
        # 离散特征嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(2, embedding_size) for _ in range(self.num_discrete)
        ])
        
        # 连续特征处理层
        self.cont_linear = nn.Linear(num_continuous, embedding_size * num_continuous)
        
        # FM线性部分
        self.fm_linear = nn.Linear(feat_size, 1)
        
        # DNN部分
        dnn_input_dim = (self.num_discrete + num_continuous) * embedding_size
        layers = []
        for dim in hidden_dims:
            layers.extend([nn.Linear(dnn_input_dim, dim), nn.ReLU()])
            dnn_input_dim = dim
        self.dnn = nn.Sequential(*layers)
        self.dnn_fc = nn.Linear(hidden_dims[-1], 1)  # 输出标量用于FM部分
        
        # 最终输出维度（FM标量 + DNN特征）
        self.output_dim = 1 + hidden_dims[-1]

    def forward(self, x):
        # 分离离散和连续特征
        discrete_feats = x[:, :self.num_discrete]  
        cont_feats = x[:, self.num_discrete:]
        
        # 处理离散特征
        if self.num_discrete > 0:
            disc_embeds = [emb(discrete_feats[:,i].long()) for i, emb in enumerate(self.embeddings)]
            disc_embeds = torch.stack(disc_embeds, dim=1)  # [B, num_disc, emb]
        else:
            disc_embeds = torch.empty(x.size(0), 0, self.embedding_size, device=x.device)
        
        # 处理连续特征
        cont_embeds = self.cont_linear(cont_feats)  # [B, num_cont*emb]
        cont_embeds = cont_embeds.view(-1, self.num_continuous, self.embedding_size)  # [B, num_cont, emb]
        
        # 合并所有嵌入
        all_embeds = torch.cat([disc_embeds, cont_embeds], dim=1)  # [B, feat_size, emb]
        
        # FM 的linear part部分-一阶
        fm_first = self.fm_linear(x)  # [B, 1]

        # FM的crossing part部分-二阶
        num_embeds = all_embeds.size(1)
        crossing_terms = []
        for i in range(num_embeds):
            for j in range(i + 1, num_embeds):
                crossing_term = torch.sum(all_embeds[:, i] * all_embeds[:, j], dim=1, keepdim=True)
                crossing_terms.append(crossing_term)
        fm_second = torch.sum(torch.cat(crossing_terms, dim=1), dim=1, keepdim=True)  # [B, 1]
        
        # DNN部分
        dnn_input = all_embeds.view(x.size(0), -1)  # [B, (disc+cont)*emb]
        dnn_out = self.dnn(dnn_input)               # [B, hidden_dim]
        
        # 合并FM和DNN输出
        combined = torch.cat([fm_first + fm_second, dnn_out], dim=1)  # [B, 1 + hidden_dim]
        return combined

class DragonDeepFM(BaseModel):
    def __init__(self, input_dim, embedding_size=4,share_dim=32,num_continuous=2, 
                 base_hidden_dims=[100,100,100,100],base_hidden_func = torch.nn.ELU(), 
                 task='classification', num_treatments=2):
        super().__init__()
        # 共享层（输出维度由DeepFM保证为1+hidden_dim[-1]）
        self.shared_layer = DeepFM(input_dim,embedding_size,[64, share_dim-1],num_continuous)
        # 倾向性头*1：用于预测当前样本属于哪个treatment（多分类 ）
        self.treatment_head = TowerUnit(input_dim= share_dim,
                                        hidden_dims= base_hidden_dims,activation= base_hidden_func, 
                                        use_batch_norm= False,
                                        task = task,classi_nums = num_treatments )
        # outcome头*n：用于预测每个treatment的outcome（回归或二分类）
        self.outcome_heads = nn.ModuleList([
                             TowerUnit(input_dim= share_dim,
                                        hidden_dims= base_hidden_dims,activation= base_hidden_func, 
                                        use_batch_norm= False,
                                        task = task,classi_nums = 2 
                            )  for _ in range(num_treatments)
                                            ])

    def forward(self, x, tr=None):
        shared = self.shared_layer(x)  
        t_pred = self.treatment_head(shared)  # softmax 2个输出
        y_preds = [head(shared) for head in self.outcome_heads] # softmax输出
        y_preds = [y[:, 1] for y in y_preds]  # 取treatment=1概率作为sigmoid输出
        return t_pred,y_preds




def dragon_loss(t_pred, y_preds, treatment, outcome, eps=None, alpha=1.0,beta=1.0,tarreg=False, task='regression'):
    """
    参数:
        t_pred: [batch_size, num_treatments] 处理预测的softmax输出, 支持单treatment与多treatment
        y_preds: [batch_size, 1] * num_treatments 每个treatment塔的sigmoid输出，支持分类和回归
        treatment: [batch_size] 带有处理索引的张量 (0 到 num_treatments-1)
        outcome: [batch_size] 带有二元标签的张量
        eps: torch.Tensor, 可训练的epsilon参数
        alpha: float, treatment loss的权重
        beta: float, targeted regularization的权重
        tarreg: bool, 是否使用targeted regularization
        task: str, 'regression' 或 'classification'
    返回:
        total_loss: 结果预测损失和处理预测损失的组合
    """
    # 1. 将 y_preds 转换为张量 [batch_size, num_treatments]
    y_pred_matrix = torch.stack(y_preds, dim=1).squeeze(-1)  # [batch_size, num_treatments]

    # 2. 结果预测损失: 仅使用实际处理的预测
    treatment = treatment.long()  # 转换为long类型以支持one_hot
    treatment_mask = F.one_hot(treatment, num_classes=len(y_preds)).float().squeeze(1)  # [batch_size, num_treatments]
    selected_y_pred = (y_pred_matrix * treatment_mask).sum(dim=1)  # [batch_size]  只保留实际处理的预测
    
    if task == 'regression':
        outcome_loss = torch.mean((selected_y_pred.unsqueeze(1) - outcome.float())**2)
    elif task == 'classification':
        outcome_loss = F.binary_cross_entropy(selected_y_pred.unsqueeze(1), outcome.float())
    else:
        raise ValueError("task must be either 'regression' or 'classification'")
    
    # 3. 处理预测损失: 交叉熵  
    treatment_loss = F.cross_entropy(t_pred, treatment.squeeze())
    
    # 4. 组合损失
    loss = outcome_loss + alpha * treatment_loss

    if tarreg:
        y_pred = treatment * y_preds[1] + (1 - treatment) * y_preds[0]   # 按实际的t预测的y：只支持二分类的t

        #y_pert = y_pred + eps # 可渐进一致 ：2个塔的loss一起下降, 去掉eps无法渐进一致  ********************待修改
        y_pert = y_pred # **********

        targeted_regularization = torch.sum((outcome - y_pert)**2)
        loss = loss + beta * targeted_regularization  

    return loss, outcome_loss, treatment_loss