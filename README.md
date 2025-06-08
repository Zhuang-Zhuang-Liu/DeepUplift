# 📈 DeepUplift 
DeepUplift is a PyTorch-based project of deep-learning heterogeneous causal effect models along with common evaluation metrics and training components.You can easily use all models with model.fit() and model.predict().


## 🌟 Features
- Deep Uplift Models：
  - ✅ TarNet: Estimating individual treatment effect: generalization bounds and algorithms, 2016.
  - ✅ CFRNet: Estimating individual treatment effect: generalization bounds and algorithms, 2016.
  - ✅ CEVAE: Causal effect inference with deep latent-variable models. In Advances in Neural Information Processing Systems, 2017.
  - ✅ GANITE：Estimation of Individualized Treatment Effects using Generative Adversarial Nets, International Conference on Learning Representations (ICLR), 2018.
  - ✅ DragonNet: Adapting Neural Networks for the Estimation of Treatment Effects, 2019.
  - ✅ DragonDeepFM: Adapting Neural Networks for the Estimation of Treatment Effects, 2019.
  - ✅ EUEN: Addressing Exposure Bias in Uplift Modeling forLarge-scale Online Advertising, IEEE International Conference on Data Mining (ICDM), 2021.
  - ✅ EEUEN: Addressing Exposure Bias in Uplift Modeling forLarge-scale Online Advertising, IEEE International Conference on Data Mining (ICDM), 2021.
  - ✅ DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation, SIGKDD, 2022.
  - ✅ EFIN: Explicit Feature Interaction-aware Uplift Network for Online Marketing, SIGKDD, 2023.
  - 🔄 SNet: Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms, 2021

- Evaluation Metrics：
  - ✅ QINI/AUUC Curves
  - ✅ Causal Effect Evaluation Metrics
  - ✅ Model Performance Evaluation Tools

- Tool Support：
  - ✅ Model Trainer
  - ✅ Propensity Score Matching (PSM)
  - ✅ TensorBoard Visualization
  - 🔄 Data Preprocessing Tools

### 🔧 Installation Dependencies
```bash
pip install pandas==2.1.4 torch==1.12.1 geomloss==0.2.6 sklearn==1.3.2 matplotlib==3.8.2 
```

## 🚀 Quick Start
```python
from deepuplift.models.DESCN import ESX
from deepuplift.utils.evaluate import uplift_metric

# Model
model,loss_f = ESX(input_dim=len(features), share_dim=12, base_dim=12),partial(esx_loss)

# Training
model.fit(X_train, Y_train, T_train,
          valid_perc=0.2,epochs=2,batch_size=64,learning_rate=1e-5,
          loss_f=loss_f,tensorboard=True)

# Prediction
t_pred, y_preds, *_ = model.predict(X_test, T_test)

# Evaluation
qini, qini_scores = uplift_metric(df, kind='qini')
```


## 📊 Download Demo Data
- Un-biaised dataset
    - Download Link : https://pan.quark.cn/s/6408800b0b8e (Quark Cloud Drive)
    - File Name : criteo-uplift-v2.1-un-biaised-sample50w.csv
    - Data source: https://ailab.criteo.com/criteo-uplift-prediction-dataset
    - If the Download Link fails, pls contact author: https://github.com/Zhuang-Zhuang-Liu/DeepUplift
- Biaised dataset
    - Download Link : https://pan.baidu.com/share/init?surl=CKJvzow7UFGwrdXbkt1mQA (Baidu Drive with code: 75hr )
    - Data source: https://github.com/kailiang-zhong/DESCN/tree/main/data/Lazada_dataset


## 📁 Project Structure
```
deepuplift/
├── models/         
├── utils/          
│   ├── evaluate.py    
│   ├── matrics.py      
│   └── psm.py          
├── dataset/       
│   └── data_link.md    
└── main.py       
```


## 🤝 Contribution
If you are interested in this project, you are welcome to contribute code, raise issues, or make suggestions. You can participate in development by submitting Pull Requests or Issues.


## 💬 Contact Me
- 📮 Email: zhuangzhuangliu_v1@qq.com
- 💚 Wechat: Wave_1024
- 📚 公众号: 壮壮的三味书屋

## 📄 License
This project is licensed under the [MIT License](LICENSE).
