import pandas as pd
from functools import partial
from utils.evaluate import *

from models.TarNet import *
from models.CFRNet import *
from models.DragonNet import *
from models.DragonDeepFM import *
from models.EFIN import *
from models.DESCN import *
from models.EUEN import *
from models.CEVAE import * 
from models.EEUEN import *
from models.GANITE import *


if __name__ == "__main__":

  # config
  treatment = 'treatment'
  outcome = 'visit'   
  features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
  task ='regression' 


  # io
  df = pd.read_csv('dataset/criteo-uplift-v2.1-un-biaised-sample50w.csv').head(200000)
  df_train,df_test = df.iloc[:100000],df.iloc[100000:200000]  
  train_stats = df_train.groupby(treatment).agg({outcome: ['mean', 'count','sum']})
  test_stats = df_test.groupby(treatment).agg({outcome: ['mean', 'count','sum']})
  print("Training set statistics:",train_stats,"\nTest set statistics:",test_stats)


  X_train,Y_train,T_train = df_train[features],df_train[outcome],df_train[treatment]
  X_test,Y_test,T_test = df_test[features],df_test[outcome],df_test[treatment]


  # models
  est1 = TarNet(input_dim=len(features),share_dim=12,task=task)
  est2 = CFRNet(input_dim=len(features),share_dim=12,task=task)
  est3 = DragonNet(input_dim=len(features),share_dim=64, task=task)
  est4 = DragonDeepFM(input_dim=12,num_continuous=12,share_dim=32,embedding_size=8,task=task, num_treatments=2)
  est5 = EFIN(input_dim=12, hc_dim=16, hu_dim=8, is_self=True,task=task) 
  est6 = ESX(input_dim=len(features),share_dim=12,base_dim=12,task='classification')
  est7 = EUEN(input_dim=len(features), hc_dim=64, hu_dim=64, is_self=False,task='regression')
  est8 = CEVAE(input_dim=len(features), h_dim=64, x_repr_dim=32, z_repr_dim=16, rep_norm=True, is_self=True,task='regression')
  est9 = EEUEN(input_dim=len(features), hc_dim=64, hu_dim=64, he_dim=64, is_self=True,task=task)
  est10 = GANITE(input_dim=len(features), h_dim=64, is_self=True,task='classification')
  
  loss1 = partial(tarnet_loss,task=task)
  loss2 = partial(cfrnet_loss,task=task)
  loss3 = partial(dragonnet_loss, alpha=1.0, beta=1.0,tarreg=True,task=task)
  loss4 = partial(dragonnet_loss, alpha=1.0, beta=1.0,tarreg=True,task=task) 
  loss5 = partial(efin_loss,task=task) 
  loss6 = partial(esx_loss,task='classification')
  loss7 = partial(euen_loss,task='regression') 
  loss8 = partial(cevae_loss,task='regression')
  loss9 = partial(eeuen_loss, alpha=1.0, beta=1.0, task=task)
  loss10 = partial(ganite_loss, task='classification')
 

  # Train 
  model,loss_f = est3,loss3
  model.fit(X_train, Y_train, T_train, valid_perc=0.2, epochs=10, batch_size=64, learning_rate=1e-5, loss_f=loss_f)



  # Prediction
  t_pred, y_preds, *_ = model.predict(X_test, T_test)
  df_result = pd.concat([T_test, Y_test, pd.DataFrame({'y_diff': (y_preds[1] - y_preds[0]).flatten(),
                                                         'y0_pred': y_preds[0].flatten(),
                                                         'y1_pred': y_preds[1].flatten(),
                                                        }, index=T_test.index)], axis=1)

  # Ploy uplift 
  plot_bins_uplift(df_result, 
                   uplift_col='y_diff', treatment_col=treatment, outcome_col=outcome,
                   figsize=(8, 8),bins = 5,if_plot = False,)

  # Evaluate uplift
  qini, qini_scores = uplift_metric( df=df_result[[treatment,outcome,'y_diff',]],
                                      kind='qini',
                                      outcome_col=outcome,treatment_col=treatment,
                                      treatment_value = 1,
                                      if_plot = False,
                                      )
  print('》》》qini_scores:\n',qini_scores)

  auuc, auuc_scores = uplift_metric(  df=df_result[[treatment,outcome,'y_diff',]],
                                      kind='auuc',
                                      outcome_col=outcome,treatment_col=treatment,
                                      treatment_value = 1,
                                      if_plot = False,
                                      )
  print('》》》auuc_scores:\n',auuc_scores)

  # Evaluate response
  calculate_metrics_by_treatment(df_result,outcome_col=outcome,treatment_col=treatment, threshold=0.5)









                     

