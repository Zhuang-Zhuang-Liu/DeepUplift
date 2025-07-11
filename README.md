# 📈 DeepUplift 
DeepUplift is a **PyTorch-based** project of deep-learning heterogeneous causal effect models along with common evaluation metrics and training components.You can easily use uplift models with **model.fit()** and **model.predict()**.


## 🌟 Features
- Deep Uplift Models：
  - ✅ TarNet: U. Shalit, F. D. Johansson, and D. Sontag. Estimating individual treatment effect: generalization bounds and algorithms.2016.
    - Link: https://arxiv.org/abs/1606.03976    
  - ✅ CFRNet: U. Shalit, F. D. Johansson, and D. Sontag. Estimating individual treatment effect: generalization bounds and algorithms.2016.
    - Link: https://arxiv.org/abs/1606.03976
  - ✅ DragonNet: Claudia Shi, David M Blei, and Victor Veitch. 2019. Adapting neural networks for the estimation of treatment effects. In Proceedings of the 33rd International Conference on Neural Information Processing Systems. 2507–2517.
    - Link: https://arxiv.org/pdf/1906.02120 
  - ✅ DragonDeepFM: Claudia Shi, David M Blei, and Victor Veitch. 2019. Adapting neural networks for the estimation of treatment effects. In Proceedings of the 33rd International Conference on Neural Information Processing Systems. 2507–2517.
  - ✅ EUEN: Wenwei Ke, Chuanren Liu, Xiangfu Shi, Yiqiao Dai, S Yu Philip, and XiaoqiangZhu. 2021. Addressing exposure bias in uplift modeling for large-scale online advertising. In Proceedings of the 2021 IEEE International Conference on Data Mining.1156–1161.
    - Link: https://github.com/aifor/eeuen
  - ✅ EEUEN: Wenwei Ke, Chuanren Liu, Xiangfu Shi, Yiqiao Dai, S Yu Philip, and XiaoqiangZhu. 2021. Addressing exposure bias in uplift modeling for large-scale online advertising. In Proceedings of the 2021 IEEE International Conference on Data Mining. 1156–1161.
    - Link: https://github.com/aifor/eeuen
  - ✅ DESCN: Kailiang Zhong, Fengtong Xiao, Yan Ren, Yaorong Liang, Wenqing Yao, Xiaofeng Yang, and Ling Cen. 2022. DESCN: Deep entire space cross networks for individual treatment effect estimation. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 4612–4620.
    - Link: https://arxiv.org/abs/2207.09920 
  - ✅ EFIN: Explicit Feature Interaction-aware Uplift Network for Online Marketing, SIGKDD, 2023.
    - Link: https://arxiv.org/abs/2306.00315
  - ✅ CEVAE: C. Louizos, U. Shalit, J. M. Mooij, D. Sontag, R. Zemel, and M. Welling.Causal effect inference with deep latent-variable models. NEURIPS. 2017.
    - Link: https://github.com/AMLab-Amsterdam/CEVAE  
  - ✅ GANITE：Jinsung Yoon, James Jordon, and Mihaela Van Der Schaar. 2018. GANITE: Estimation of individualized treatment effects using generative adversarial nets. In Proceedings of the 6th International Conference on Learning Representations.
    - Link: https://github.com/jsyoon0823/GANITE
  - 🔄 SNet: Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms, 2021.

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
          valid_perc=0.2,epochs=2,batch_size=64,learning_rate=1e-5,loss_f=loss_f,tensorboard=True)

# Prediction
t_pred, y_preds, *_ = model.predict(X_test, T_test)

# Evaluation
qini, qini_scores = uplift_metric(df, kind='qini')
```


## 📊 Download Demo Data
- Un-biaised dataset
    - Download Link : https://pan.quark.cn/s/6408800b0b8e (Quark Cloud Drive)
    - Data source: https://ailab.criteo.com/criteo-uplift-prediction-dataset
- Biaised dataset
    - Download Link : https://pan.baidu.com/share/init?surl=CKJvzow7UFGwrdXbkt1mQA (Baidu Drive with code: 75hr )
    - Data source: https://github.com/kailiang-zhong/DESCN/tree/main/data/Lazada_dataset


## 📁 Project Structure
```
deepuplift/
├── models/
│   ├── BaseModel.py
│   ├── BaseUnit.py
│   ├── ...             
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
