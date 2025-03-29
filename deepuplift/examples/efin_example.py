import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.modules.fold import F

from trainer import Trainer
from models.efin import EFIN, WrapperModel
from functools import partial
from utils.evaluate import *


if __name__ == "__main__":

  import models
  print('__version__:',models.__version__)

  # define
  treatment = 'treatment'
  outcome = 'visit'   
  features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']

  # io
  csv_name = r'deepuplift/dataset/criteo-uplift-v2.1-50w.csv'
  df = pd.read_csv(csv_name).head(200000)
  group_stats = df.groupby(treatment).agg({outcome: ['mean', 'count','sum']})
  print('整体:',group_stats)

  # Split 
  df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)  
  print(df_train.shape, df_test.shape)
  # 统计训练集和测试集的按treatment分组后的outcome的总数和均值
  train_stats = df_train.groupby(treatment).agg({outcome: ['mean', 'count','sum']})
  test_stats = df_test.groupby(treatment).agg({outcome: ['mean', 'count','sum']})
  print("训练集统计结果:",train_stats)
  print("\n测试集统计结果:",test_stats)
  
  X_train,Y_train,T_train = df_train[features],df_train[outcome],df_train[treatment]
  X_test,Y_test,T_test = df_test[features],df_test[outcome],df_test[treatment]

  # fit
  model = EFIN(input_dim=12, hc_dim=16, hu_dim=32, is_self=False, act_type="elu")
  model = WrapperModel(model).to( "cpu") 
  model = Trainer(model = model, 
                  epochs=2,  
                  loss_f = None  
                  )  
  print("开始训练^^")      
  model.fit(X_train, Y_train, T_train,valid_perc=0.2)
  print("训练完成^^")

  # predict - valid
  y0_pred, y1_pred, t_pred, _ = model.predict(X_test)
  df_result = pd.concat([T_test, Y_test, 
                        pd.DataFrame({'y_diff': (y1_pred - y0_pred).flatten(),
                                      'y0_pred': y0_pred.flatten(),
                                      'y1_pred': y1_pred.flatten(),
                                      't_pred': t_pred.flatten()
                                      }, index=T_test.index)], axis=1)

  aa = plot_bins_uplift(df_result, 
                         uplift_col='y_diff', 
                         treatment_col=treatment, 
                         outcome_col=outcome,
                         figsize=(8, 8),
                         bins = 5,
                         if_plot = False,
                         )

  # evaluate - uplift
  qini, qini_scores = uplift_metric( df=df_result[[treatment,outcome,'y_diff',]],
                                      kind='qini',
                                      outcome_col=outcome,
                                      treatment_col=treatment,
                                      treatment_value = 1,
                                      if_plot = False
                                      )

  auuc, auuc_scores = uplift_metric( df=df_result[[treatment,outcome,'y_diff',]],
                                      kind='auuc',
                                      outcome_col=outcome,
                                      treatment_col=treatment,
                                      treatment_value = 1,
                                      if_plot =  False,
                                      )

  # evaluate - response
  res = calculate_metrics_by_treatment(df_result,outcome_col=outcome,treatment_col=treatment, threshold=0.5)

  # show
  plt.show()

