import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

def uplift_metric(df,outcome_col="outcome",treatment_col="treatment",treatment_value = 1,
                    kind = 'qini',if_plot = True,):
    """
    Reference: Causal Inference and Uplift Modeling A review of the literature
    auuc: area under uplift curve, uplift curve = (treatment group conversion rate - control group conversion rate) * (treatment group count + control group count)
    qini: area under qini curve, when treatment and control groups have different sample sizes, qini should be chosen, qini curve = treatment group cumulative conversions - control group cumulative conversions * (treatment group cumulative count / control group cumulative count)
    # Note: AUUC metric calculation cannot be used for observational data because the score distributions of treatment and control groups are inconsistent
    """
    df = df.copy()
    model_names = [x for x in df.columns if x not in [outcome_col, treatment_col]]

    # Add random curve
    model_names.append('random')
    df['random'] = np.random.rand(df.shape[0])

    all_metric = {}
    for model_name in model_names:
        # 01-Sort
        subset_df = df.copy()   # Filter and copy: avoid modifying original data
        sorted_df = subset_df.sort_values(model_name, ascending=False).reset_index(drop=True) # Sort in descending order
        sorted_df.index = sorted_df.index + 1 # Index + 1 for easier calculation
        # 02-Cumulative statistics: cumulative count of treatment and control groups
        sorted_df["cumsum_tr"] = (sorted_df[treatment_col] == treatment_value).astype(int).cumsum()
        sorted_df["cumsum_ct"] = sorted_df.index.values - sorted_df["cumsum_tr"]
        # 03-Cumulative statistics: cumulative conversions of treatment and control groups
        sorted_df["cumsum_y_tr"] = ( sorted_df[outcome_col] * (sorted_df[treatment_col] == treatment_value).astype(int)
                                    ).cumsum()
        sorted_df["cumsum_y_ct"] = (sorted_df[outcome_col] * (sorted_df[treatment_col] == 0).astype(int)
                                    ).cumsum()
        # 04- Qini Curve: incremental gain assuming all control group are treated
        qini_value = ( sorted_df["cumsum_y_tr"]  - sorted_df["cumsum_y_ct"] 
                      * ( sorted_df["cumsum_tr"]/ sorted_df["cumsum_ct"]) )
        # 04- Uplift Curve: incremental gain assuming 100% are treated
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
    Visualize conversion rates and uplift rates after binning the statistical model
    """
    # Calculate
    df['bins'] = pd.qcut(df[uplift_col], bins,duplicates='drop')
    result = df.groupby([treatment_col, 'bins'])[outcome_col].agg(['mean', 'sum','count'])
    result.reset_index(inplace=True)
    
    # Plot
    if if_plot:
        unique_treatment = result[treatment_col].unique()
        plt.figure(figsize=figsize)
        for val in unique_treatment:
            subset = result[result[treatment_col] == val]
            line = plt.plot(subset['bins'].astype(str), subset['mean'], label=f'treatment_{val}')[0]
            for x, y in zip(subset['bins'].astype(str), subset['mean']):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot uplift
        treatment_1 = result[result[treatment_col] == 1]
        treatment_0 = result[result[treatment_col] == 0]
        if len(treatment_1) == len(treatment_0):
            diff = treatment_1['mean'].values - treatment_0['mean'].values
            line = plt.plot(treatment_1['bins'].astype(str), diff, label='uplift', color='red')[0]
            for x, y in zip(treatment_1['bins'].astype(str), diff):
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            print("The number of bins for treatment and control groups are inconsistent, unable to calculate difference.")

            plt.xlabel(uplift_col)
            plt.ylabel(f'Mean of {outcome_col}')
            plt.title(f'Mean of {outcome_col} by {uplift_col} bins and {treatment_col}')
            plt.xticks(rotation=45)
            plt.legend()

    return result


def calculate_metrics_by_treatment(df, outcome_col,treatment_col,threshold=0.5,if_plot = True):
    """
    Split data based on T_test values (0 or 1), and evaluate the differences between y0_pred and y1_pred with Y_test respectively.
    Parameters:
    df (pd.DataFrame): DataFrame containing 'outcome', 'y0_pred', 'y1_pred', 't_pred'
    threshold (float): Threshold for calculating confusion matrix, default is 0.5
    Returns:
    dict: Dictionary containing AUC and confusion matrix
    """
    def calculate_metrics(df, outcome, pred, threshold, ax_roc, ax_cm):
        auc = roc_auc_score(df[outcome], df[pred])
        pred_class = (df[pred] >= threshold).astype(int)
        cm = confusion_matrix(df[outcome], pred_class)
        if if_plot:
            # Plot AUC
            fpr, tpr, thresholds = roc_curve(df[outcome], df[pred])
            ax_roc.plot(fpr, tpr, label=f'{pred} (AUC = {auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend(loc='lower right')
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_title(f'Confusion Matrix for {pred}')
        return auc, cm

    df_t0 = df[df[treatment_col] == 0]
    df_t1 = df[df[treatment_col] == 1]
    if if_plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  
        plt.subplots_adjust(wspace=0.3, hspace=0.3)  
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