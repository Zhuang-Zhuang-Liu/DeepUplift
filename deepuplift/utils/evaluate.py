import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

def uplift_metric(df,outcome_col="outcome",treatment_col="treatment",treatment_value = 1,
                    kind = 'qini',if_plot = True,):
    """
    paper：Causal Inference and Uplift Modeling A review of the literature
    auuc: uplift curve的面积, uplift curve = (实验组转化率-对照组转化率)*(实验组人数+对照组人数)
    qini: qini curve的面积,当实验组和对照组样本不一致时, 应选择qini, qini curve = 实验组累计转化 - 对照组累计转化 *（实验组累计人数/对照组累计人数）
    # 注意：AUUC指标计算不能用来自观测数据，因为实验组和对照组的分数分布不一致
    """
    df = df.copy()
    model_names = [x for x in df.columns if x not in [outcome_col, treatment_col]]

    # 添加随机curve
    model_names.append('random')
    df['random'] = np.random.rand(df.shape[0])

    all_metric = {}
    for model_name in model_names:
        # 01-排序
        subset_df = df.copy()   # 筛选+复制：避免修改原数据
        sorted_df = subset_df.sort_values(model_name, ascending=False).reset_index(drop=True) # 降序排序
        sorted_df.index = sorted_df.index + 1 # 索引+1，方便计算
        # 02-累加统计：实验组、对照组的累计数
        sorted_df["cumsum_tr"] = (sorted_df[treatment_col] == treatment_value).astype(int).cumsum()
        sorted_df["cumsum_ct"] = sorted_df.index.values - sorted_df["cumsum_tr"]
        # 03-累加统计：实验组、对照组的累计转化数
        sorted_df["cumsum_y_tr"] = ( sorted_df[outcome_col] * (sorted_df[treatment_col] == treatment_value).astype(int)
                                    ).cumsum()
        sorted_df["cumsum_y_ct"] = (sorted_df[outcome_col] * (sorted_df[treatment_col] == 0).astype(int)
                                    ).cumsum()
        # 04- Qini Curve ：假设对照组都干预的增量
        qini_value = ( sorted_df["cumsum_y_tr"]  - sorted_df["cumsum_y_ct"] 
                      * ( sorted_df["cumsum_tr"]/ sorted_df["cumsum_ct"]) )
        # 04- Uplift Curve：假设100%都干预的增量
        uplift_curve = (
                        ( (sorted_df["cumsum_y_tr"]/sorted_df["cumsum_tr"]) - (sorted_df["cumsum_y_ct"]/sorted_df["cumsum_ct"]))
                        * ( sorted_df["cumsum_tr"] + sorted_df["cumsum_ct"] )
                        )
        if kind  == 'qini':    all_metric[f"{model_name}"] = qini_value
        elif kind == 'auuc':    all_metric[f"{model_name}"] = uplift_curve
        else:    raise ValueError("kind must be 'qini' or 'auuc'")

    # df
    metric_df = pd.concat(all_metric.values(), axis=1)
    metric_df.loc[0] = np.zeros((metric_df.shape[1],))
    metric_df = metric_df.sort_index().interpolate()
    metric_df.columns = all_metric.keys()

    # plot
    if if_plot:
        plt.figure(figsize=(10, 6))
        for col in metric_df.columns:
            plt.plot(metric_df.index, metric_df[col], label=col)
        plt.xlabel("Number of Samples")
        plt.ylabel("Cumulative Gain")
        plt.title(f"{kind} Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    # calculate score
    metric_score = {}
    for col in metric_df.columns:
        metric_score[col] = metric_df[col].sum()

    return metric_df, pd.Series(metric_score)


def plot_bins_uplift(df, uplift_col, treatment_col, outcome_col,figsize=(8, 8),bins = 10,if_plot = False):
    """
    可视化的统计模型分箱后的转化率及增益转化率
    """
    # calculate
    df['bins'] = pd.qcut(df[uplift_col], bins)
    result = df.groupby([treatment_col, 'bins'])[outcome_col].agg(['mean', 'sum','count'])
    result.reset_index(inplace=True)
    
    # plot
    if if_plot:
        unique_treatment = result[treatment_col].unique()
        plt.figure(figsize=figsize)
        for val in unique_treatment:
            subset = result[result[treatment_col] == val]
            line = plt.plot(subset['bins'].astype(str), subset['mean'], label=f'treatment_{val}')[0]
            for x, y in zip(subset['bins'].astype(str), subset['mean']):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
        
        # plot uplift
        treatment_1 = result[result[treatment_col] == 1]
        treatment_0 = result[result[treatment_col] == 0]
        if len(treatment_1) == len(treatment_0):
            diff = treatment_1['mean'].values - treatment_0['mean'].values
            line = plt.plot(treatment_1['bins'].astype(str), diff, label='uplift', color='red')[0]
            for x, y in zip(treatment_1['bins'].astype(str), diff):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            print("处理组和控制组的分箱数量不一致，无法计算差值。")

            plt.xlabel(uplift_col)
            plt.ylabel(f'Mean of {outcome_col}')
            plt.title(f'Mean of {outcome_col} by {uplift_col} bins and {treatment_col}')
            plt.xticks(rotation=45)
            plt.legend()

    return result


def calculate_metrics_by_treatment(df, outcome_col,treatment_col,threshold=0.5,if_plot = True):
    """
    根据 T_test 的值（0 或 1）拆分数据，分别评估 y0_pred 和 y1_pred 与 Y_test 的差异。
    参数:
    df (pd.DataFrame): 包含 'outcome', 'y0_pred', 'y1_pred', 't_pred' 的DataFrame
    threshold (float): 用于计算混淆矩阵的阈值，默认为0.5
    返回:
    dict: 包含AUC和混淆矩阵的字典
    """
    def calculate_metrics(df, outcome, pred, threshold, ax_roc, ax_cm):
        auc = roc_auc_score(df[outcome], df[pred])
        pred_class = (df[pred] >= threshold).astype(int)
        cm = confusion_matrix(df[outcome], pred_class)
        if if_plot:
            # 绘制auc
            fpr, tpr, thresholds = roc_curve(df[outcome], df[pred])
            ax_roc.plot(fpr, tpr, label=f'{pred} (AUC = {auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend(loc='lower right')
            # 绘制混淆矩阵
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_title(f'Confusion Matrix for {pred}')
        return auc, cm

    df_t0 = df[df[treatment_col] == 0]
    df_t1 = df[df[treatment_col] == 1]
    if if_plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 增加图形大小
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 调整子图之间的间距
        auc_y0, cm_y0 = calculate_metrics(df_t0, outcome_col, 'y0_pred', threshold, axes[0, 0], axes[1, 0])
        auc_y1, cm_y1 = calculate_metrics(df_t1, outcome_col, 'y1_pred', threshold, axes[0, 1], axes[1, 1])
    else:
        auc_y0, cm_y0 = calculate_metrics(df_t0, outcome_col, 'y0_pred', threshold, None, None)
        auc_y1, cm_y1 = calculate_metrics(df_t1, outcome_col, 'y1_pred', threshold, None, None)

    return {
        'auc_y0': auc_y0,
        'auc_y1': auc_y1,
        'confusion_matrix_y0': cm_y0,
        'confusion_matrix_y1': cm_y1
    }