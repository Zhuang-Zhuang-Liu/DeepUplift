import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


class PSM:
    """
    倾向得分匹配（PSM）类，用于根据指定特征和处理变量进行匹配分析。
    """
    def __init__(self, df, features, treatment_col='treatment', outcome_col='outcome', model='logistic', **model_params):
        """
        初始化PSM类实例。

        参数:
        - df: 包含数据的DataFrame
        - features: 用于匹配的特征列表
        - treatment_col: 处理组标识列名，默认为'treatment'
        - outcome_col: 结果列名，默认为'outcome'
        - model: 用于估计倾向得分的模型，可选值为'logistic'、'random_forest'、'gbm'，默认为'logistic'
        - **model_params: 模型的参数
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        X = df[features]
        y = df[self.treatment_col]
        
        if model == 'logistic':
            self.model = LogisticRegression(**model_params)
        else:
            raise ValueError(f"Unsupported model: {model}")


        print('PSM Model Training...')
        self.model.fit(X, y)
        print('PSM Model Ready')
        self.ps = self.model.predict_proba(X)[:, 1]
    
    def match(self, df, method='nearest', k=1, caliper=None, replace=False):
        """
        执行匹配操作。

        参数:
        - df: 包含数据的DataFrame
        - method: 匹配方法，可选值为'nearest'、'caliper'，默认为'nearest'
        - k: 最近邻的数量，默认为1
        - caliper: 卡尺匹配的阈值，默认为None
        - replace: 是否允许有放回匹配，默认为False

        返回:
        - 匹配后的DataFrame
        """
        treated = df[df[self.treatment_col] == 1].copy()
        control = df[df[self.treatment_col] == 0].copy()
        
        treated['ps'] = self.ps[df[self.treatment_col] == 1]
        control['ps'] = self.ps[df[self.treatment_col] == 0]
        
        if method == 'nearest':
            # 最近邻匹配实现
            nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(control[['ps']])
            distances, indices = nbrs.kneighbors(treated[['ps']])
            
            if not replace: # 无放回匹配
                matched_control = control.iloc[np.unique(indices)].copy()
            else: # 有放回匹配
                matched_control = control.iloc[indices.flatten()].copy()
                
            matched_df = pd.concat([treated, matched_control])
            
        elif method == 'caliper':
            # 卡尺匹配实现
            if caliper is None:
                caliper = 0.2 * np.std(self.ps)
                
            nbrs = NearestNeighbors(n_neighbors=k, radius=caliper, metric='euclidean').fit(control[['ps']])
            distances, indices = nbrs.kneighbors(treated[['ps']])
            
            valid = distances <= caliper
            matched_indices = [i[v] for i, v in zip(indices, valid)]
            
            matched_control = pd.concat([control.iloc[i] for i in matched_indices if len(i) > 0])
            matched_df = pd.concat([treated, matched_control])

        return matched_df
    



def balance_check(df, treatment_col,outcome_col,features=None, plot_vars_kde=True):
    """
    检查处理组和对照组的特征分布平衡性

    参数:
    - df: 包含数据的DataFrame
    - treatment_col: 处理组标识列名(str)
    - outcome_col: 结果列名(str)，新增参数说明
    - features: 要检查的特征列表(list)，如果为None则使用所有数值型列
    - plot_vars_kde: 是否绘制特征的核密度估计图(bool)，原注释中没有该参数，更新注释添加

    返回:
    - balance_df: 包含SMD和平衡状态的DataFrame
    - 如果plot_vars_kde为True，会显示特征的核密度估计图
    """
    
    # sample imbalance
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    group_res = df.groupby(treatment_col)[outcome_col].agg(['count', 'mean'])
    print(f"""==== Group Summary ====""")
    print(group_res)

    # treatment bias
    smd_results = []
    for feature in features:
        # 计算均值和标准差
        mean_treated = treated[feature].mean()
        mean_control = control[feature].mean()
        std_pooled = np.sqrt((treated[feature].var() + control[feature].var()) / 2)
        # 计算SMD
        smd = abs(mean_treated - mean_control) / std_pooled if std_pooled != 0 else 0
        # 执行t检验
        t_stat, p_value = ttest_ind(treated[feature], control[feature])
        # 确定是否平衡
        balanced = 'Balanced' if smd <= 0.1 else 'Not Balanced'
        
        smd_results.append({
            'Feature': feature,
            'SMD': smd,
            'Treated_Mean': mean_treated,
            'Control_Mean': mean_control,
            'Absolute_Difference': abs(mean_treated - mean_control),
            'P_value': p_value,
            'Balanced': balanced
        })
    
    balance_res = pd.DataFrame(smd_results)
    print(f"""==== Balance Check Summary ====""")
    print(balance_res)
    
    if plot_vars_kde:
        num_features = len(features)
        fig, axes = plt.subplots(num_features, 2, figsize=(10, 3 * num_features))
        for i, feature in enumerate(features):
            # 绘制treatment=0的概率密度曲线
            sns.kdeplot(df[df[treatment_col] == 0][feature], ax=axes[i, 0])
            axes[i, 0].set_title(f'{feature} (Treatment = 0)')
            axes[i, 0].set_xlabel(feature)
            axes[i, 0].set_ylabel('Density')
            # 绘制treatment=1的概率密度曲线
            sns.kdeplot(df[df[treatment_col] == 1][feature], ax=axes[i, 1])
            axes[i, 1].set_title(f'{feature} (Treatment = 1)')
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel('Density')
        plt.tight_layout()
    
    return balance_res