from pylift.eval import UpliftEval,get_scores
import pandas as pd
import numpy as np
from sklift.metrics import qini_auc_score,uplift_at_k
from sklearn.metrics import roc_auc_score
import torch
import shap
import random
from causalml.metrics import qini_score,plot_qini
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


class MyModelWrapper(torch.nn.Module):
        def __init__(self, model, feature_list_discrete, device, treatment):
            super().__init__()
            self.model = model
            self.feature_list_discrete = feature_list_discrete
            self.device = device
            self.treatment = treatment
        def forward(self, X):
            X_discrete = X[:, :len(self.feature_list_discrete)]
            X_continuous = X[:, len(self.feature_list_discrete):]
            self.model.eval()
            uplift_predictions, y_preds, *eps = self.model(None, None, X_discrete=X_discrete, X_continuous=X_continuous)
            return uplift_predictions[:,self.treatment-1].unsqueeze(1)

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def split_df_into_n_random_parts(df, n=10, random_state=None):
    # 1. 打乱索引
    shuffled_indices = df.sample(frac=1, random_state=random_state).index
    # 2. 计算每份大小（尽量均匀）
    total_len = len(df)
    part_size = total_len // n
    remainder = total_len % n
    dfs = []
    start = 0
    for i in range(n):
        # 处理最后几份多一个元素的情况
        end = start + part_size + (1 if i < remainder else 0)
        part_indices = shuffled_indices[start:end]
        dfs.append(df.loc[part_indices])
        start = end
    return dfs

def get_qini(
    df,
    outcome_col="y",
    treatment_col="w",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
):
    """Get Qini of model estimates in population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): deprecated

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """
    assert (
        (outcome_col in df.columns and df[outcome_col].notnull().all())
        and (treatment_col in df.columns and df[treatment_col].notnull().all())
        or (
            treatment_effect_col in df.columns
            and df[treatment_effect_col].notnull().all()
        )
    ), "{outcome_col} and {treatment_col}, or {treatment_effect_col} should be present without null.".format(
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        treatment_effect_col=treatment_effect_col,
    )

    df = df.copy()

    model_names = [
        x
        for x in df.columns
        if x not in [outcome_col, treatment_col, treatment_effect_col]
    ]

    qini = []
    for i, col in enumerate(model_names):
        sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)
        sorted_df.index = sorted_df.index + 1
        sorted_df["cumsum_tr"] = sorted_df[treatment_col].cumsum()

        if treatment_effect_col in sorted_df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population.
            l = (
                sorted_df[treatment_effect_col].cumsum()
                / sorted_df.index
                * sorted_df["cumsum_tr"]
            )
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population.
            sorted_df["cumsum_ct"] = sorted_df.index.values - sorted_df["cumsum_tr"]
            sorted_df["cumsum_y_tr"] = (
                sorted_df[outcome_col] * sorted_df[treatment_col]
            ).cumsum()
            sorted_df["cumsum_y_ct"] = (
                sorted_df[outcome_col] * (1 - sorted_df[treatment_col])
            ).cumsum()

            l = (
                sorted_df["cumsum_y_tr"]
                - sorted_df["cumsum_y_ct"]
                * sorted_df["cumsum_tr"]
                / sorted_df["cumsum_ct"]
            )

        qini.append(l)

    qini = pd.concat(qini, join="inner", axis=1)
    qini.loc[0] = np.zeros((qini.shape[1],))
    qini = qini.sort_index().interpolate()

    qini.columns = model_names

    if normalize:
        qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)

    return qini

def evaluate(df=None,y_true=None,uplift_predictions=None,treatment=None,divide_feature=None,n=100):
    if divide_feature is None:
        sm_qini_auc_score_model = get_qini(df,outcome_col=y_true,treatment_col=treatment)
        print('qini结果:')
        print(sm_qini_auc_score_model.sum(axis=0).sort_values(ascending=False))

        plot_qini(df,outcome_col=y_true,treatment_col=treatment,n=n)

     
    else:
        divide_features = sorted(df[divide_feature].unique().tolist())
        print(divide_features)

        model_res = []
        for each in divide_features:
            print(each)
            df_tmp = df[df[divide_feature] == each]
            del df_tmp[divide_feature]
            
            sm_qini_auc_score_model = get_qini(df_tmp,outcome_col=y_true,treatment_col=treatment)
            print(sm_qini_auc_score_model.sum(axis=0).sort_values(ascending=False))
            plot_qini(df,outcome_col=y_true,treatment_col=treatment,n=n)
            print('---------------------------------')
            model_res.append(sm_qini_auc_score_model.sum(axis=0).sort_values(ascending=False)) 
        
        df = pd.DataFrame(model_res)
        mean_series = df.mean()
        print(f'平均结果:{mean_series}')

def feature_importance_with_shap(model,df_train,df_test,feature_list,feature_list_discrete,device,treatment):
    X_train = df_train[feature_list_discrete + [_ for _ in feature_list if _ not in feature_list_discrete]].values
    X_test = df_test[feature_list_discrete + [_ for _ in feature_list if _ not in feature_list_discrete]].values

    my_model = MyModelWrapper(model, feature_list_discrete, device, treatment)
    background = torch.tensor(X_train, dtype=torch.float32)
    explainer = shap.GradientExplainer(my_model, background)
    shap_values = explainer.shap_values(torch.tensor(X_test, dtype=torch.float32))
    shap.summary_plot(
        shap_values,         # 解释结果
        X_test,              # 原始输入数据
        feature_names=feature_list  # 可选：特征名
    )
    