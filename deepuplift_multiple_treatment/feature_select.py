import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score

def features_select(df=None,features=None,label=None,n_top=None,selected_path=None,task='classification'):
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    if task=='classification':
        # 2. 训练原始模型（使用全部特征）
        model_full = xgb.XGBClassifier(n_estimators = 150,use_label_encoder=False, eval_metric='logloss',n_jobs=-1)
        model_full.fit(df_train[features].values, df_train[label].values)

        # 3. 预测并计算原始模型AUC
        y_pred_proba_full = model_full.predict_proba(df_test[features].values)[:, 1]
        auc_full = roc_auc_score(df_test[label].values, y_pred_proba_full)
        print(f"AUC with all features: {auc_full:.4f}")

        # 4. 计算特征重要性
        importance = model_full.feature_importances_
        importance_df = pd.DataFrame({'feature': features, 'importance': importance})
        print(importance_df.shape)
        importance_df = importance_df[importance_df['importance'] > 0]
        print(importance_df.shape)
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        top_features = importance_df['feature'].head(n_top).tolist()
        print(len(top_features))
        # 5. 用选出的特征重新训练模型
        model_selected = xgb.XGBClassifier(n_estimators = 150,use_label_encoder=False, eval_metric='logloss',n_jobs=-1)
        model_selected.fit(df_train[top_features].values, df_train[label].values)

        # 6. 预测测试集，计算AUC
        y_pred_proba = model_selected.predict_proba(df_test[top_features])[:, 1]
        auc_score = roc_auc_score(df_test[label].values, y_pred_proba)
        print(f"Selected top {n_top} features: {top_features}")
        print(f"AUC on test set with selected features: {auc_score:.4f}")
    else:
        if task == 'regression':
            # 1. 训练全特征模型
            model_full = xgb.XGBRegressor(n_estimators=150, n_jobs=-1)
            model_full.fit(df_train[features].values, df_train[label].values)

            # 2. 预测并计算原始模型回归指标
            y_pred_full = model_full.predict(df_test[features].values)
            mse_full = mean_squared_error(df_test[label].values, y_pred_full)
            r2_full = r2_score(df_test[label].values, y_pred_full)
            print(f"MSE with all features: {mse_full:.4f}")
            print(f"R2 with all features: {r2_full:.4f}")

            # 3. 计算特征重要性
            importance = model_full.feature_importances_
            importance_df = pd.DataFrame({'feature': features, 'importance': importance})
            print(importance_df.shape)
            importance_df = importance_df[importance_df['importance'] > 0]
            print(importance_df.shape)
            importance_df = importance_df.sort_values(by='importance', ascending=False)
            top_features = importance_df['feature'].head(n_top).tolist()
            print(len(top_features))

            # 4. 用选出的特征重新训练模型
            model_selected = xgb.XGBRegressor(n_estimators=150, n_jobs=-1)
            model_selected.fit(df_train[top_features].values, df_train[label].values)

            # 5. 预测测试集，计算回归指标
            y_pred_selected = model_selected.predict(df_test[top_features].values)
            mse_selected = mean_squared_error(df_test[label].values, y_pred_selected)
            r2_selected = r2_score(df_test[label].values, y_pred_selected)
            print(f"Selected top {n_top} features: {top_features}")
            print(f"MSE on test set with selected features: {mse_selected:.4f}")
            print(f"R2 on test set with selected features: {r2_selected:.4f}")
        else:
            raise ValueError("task must be 'classification' or'regression'")

    import pickle
    # 将列表保存到文件
    with open(selected_path, 'wb') as f:
        pickle.dump(top_features, f)

    return top_features

    
    