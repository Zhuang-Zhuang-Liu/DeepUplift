import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


class PSM:
    """
    Propensity Score Matching (PSM) class for matching analysis based on specified features and treatment variables.
    """
    def __init__(self, df, features, treatment_col='treatment', outcome_col='outcome', model='logistic', **model_params):
        """
        Initialize PSM class instance.

        Parameters:
        - df: DataFrame containing the data
        - features: List of features for matching
        - treatment_col: Treatment group identifier column name, default is 'treatment'
        - outcome_col: Outcome column name, default is 'outcome'
        - model: Model for estimating propensity scores, options are 'logistic', 'random_forest', 'gbm', default is 'logistic'
        - **model_params: Model parameters
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
        Perform matching operation.

        Parameters:
        - df: DataFrame containing the data
        - method: Matching method, options are 'nearest', 'caliper', default is 'nearest'
        - k: Number of nearest neighbors, default is 1
        - caliper: Threshold for caliper matching, default is None
        - replace: Whether to allow matching with replacement, default is False

        Returns:
        - DataFrame after matching
        """
        treated = df[df[self.treatment_col] == 1].copy()
        control = df[df[self.treatment_col] == 0].copy()
        
        treated['ps'] = self.ps[df[self.treatment_col] == 1]
        control['ps'] = self.ps[df[self.treatment_col] == 0]
        
        if method == 'nearest':
            # Nearest neighbor matching implementation
            nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(control[['ps']])
            distances, indices = nbrs.kneighbors(treated[['ps']])
            
            if not replace: # Matching without replacement
                matched_control = control.iloc[np.unique(indices)].copy()
            else: # Matching with replacement
                matched_control = control.iloc[indices.flatten()].copy()
                
            matched_df = pd.concat([treated, matched_control])
            
        elif method == 'caliper':
            # Caliper matching implementation
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
    Check the balance of feature distributions between treatment and control groups

    Parameters:
    - df: DataFrame containing the data
    - treatment_col: Treatment group identifier column name (str)
    - outcome_col: Outcome column name (str), newly added parameter description
    - features: List of features to check (list), if None, use all numeric columns
    - plot_vars_kde: Whether to plot kernel density estimation plots for features (bool), added parameter to update comments

    Returns:
    - balance_df: DataFrame containing SMD and balance status
    - If plot_vars_kde is True, displays kernel density estimation plots for features
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
        # Calculate mean and standard deviation
        mean_treated = treated[feature].mean()
        mean_control = control[feature].mean()
        std_pooled = np.sqrt((treated[feature].var() + control[feature].var()) / 2)
        # Calculate SMD
        smd = abs(mean_treated - mean_control) / std_pooled if std_pooled != 0 else 0
        # Perform t-test
        t_stat, p_value = ttest_ind(treated[feature], control[feature])
        # Determine if balanced
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
            # Plot probability density curve for treatment=0
            sns.kdeplot(df[df[treatment_col] == 0][feature], ax=axes[i, 0])
            axes[i, 0].set_title(f'{feature} (Treatment = 0)')
            axes[i, 0].set_xlabel(feature)
            axes[i, 0].set_ylabel('Density')
            # Plot probability density curve for treatment=1
            sns.kdeplot(df[df[treatment_col] == 1][feature], ax=axes[i, 1])
            axes[i, 1].set_title(f'{feature} (Treatment = 1)')
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel('Density')
        plt.tight_layout()
    
    return balance_res