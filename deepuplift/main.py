import pandas as pd
from sklearn.model_selection import train_test_split
from trainer import Trainer
from models.DragonNet import DragonNet,tarreg_loss
from models.DragonDeepFM import *
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
  print("训练集统计结果:",train_stats,"\n测试集统计结果:",test_stats)

  
  X_train,Y_train,T_train = df_train[features],df_train[outcome],df_train[treatment]
  X_test,Y_test,T_test = df_test[features],df_test[outcome],df_test[treatment]


  # choose model
  est1 = Trainer(model = DragonNet(len(features)),
                 epochs=20,batch_size=64,
                 loss_f = partial(tarreg_loss, alpha=1.0, beta=1.0) 
                )
  est2 = Trainer(model = DragonNetDeepFM(input_dim=12,num_continuous=12
                                         ,embedding_size=8,shared_dim=32,
                                         kind='classi', num_treatments=2), 
                 epochs=1,batch_size=64,
                 loss_f = partial(Dragon_loss, alpha=1) 
                ) 

  model = est1
  model.fit(X_train, Y_train, T_train,valid_perc=0.2)
  t_pred,y_preds, *_ = model.predict(X_test)

  # predict - validation
  df_result = pd.concat([T_test, Y_test, 
                        pd.DataFrame({'y_diff': (y_preds[1] - y_preds[0]).flatten(),
                                      'y0_pred': y_preds[0].flatten(),
                                      'y1_pred': y_preds[1].flatten(),
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
                                      if_plot =  False,
                                      )

  auuc, auuc_scores = uplift_metric(  df=df_result[[treatment,outcome,'y_diff',]],
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









                     

