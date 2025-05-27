# Step 1: 导入库
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
import plotly.graph_objects as go

# Step 2: 载入数据
X_train = pd.read_csv('data/SourceCM.csv', index_col=0)
X_test = pd.read_csv('data/QueryCM.csv', index_col=0)
y_train = pd.read_csv('data/SourceLabel.csv', index_col=0)
y_test = pd.read_csv('data/QueryLabel.csv', index_col=0)

# Step 3: 预处理丰度数据（属水平）
def preprocess_abundance(abu, level):
    abu = abu.T  # 转置
    abu = abu.loc[:, abu.sum(axis=0) > 0]  # 去除全为0的特征
    abu.columns = abu.columns.str.split(';', expand=True)  # 将分类信息拆分为多列
    abu.columns = abu.columns.get_level_values(level)  # 选择指定层级的分类信息
    abu = abu.loc[:, abu.columns.notnull()]  # 去除分类信息为空的特征
    abu = abu.groupby(abu.columns, axis=1).sum()  # 按分类信息聚合
    return abu

X_train = preprocess_abundance(X_train, 5)  # 属水平
X_test = preprocess_abundance(X_test, 5)

# Step 4: 特征对齐 + CLR 转换
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_train = X_train.div(X_train.sum(axis=1), axis=0)
X_test = X_test.div(X_test.sum(axis=1), axis=0)

def clr_transform(df):
    pseudocount = 1e-6
    log_df = np.log(df + pseudocount)
    gm = log_df.mean(axis=1)
    clr = log_df.sub(gm, axis=0)
    return clr

X_train = clr_transform(X_train)
X_test = clr_transform(X_test)

# Step 5: 标签处理
y_train = y_train['Env'].str.split(':', expand=True).iloc[:, -1]
y_test = y_test['Env'].str.split(':', expand=True).iloc[:, -1]
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Step 6: 训练模型
model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train_enc)

# Step 7: 预测
y_score = model.predict_proba(X_test)

# Step 8: AUC 评估
auc_ovr = roc_auc_score(y_test_enc, y_score, multi_class='ovr')
auc_ovo = roc_auc_score(y_test_enc, y_score, multi_class='ovo')

# Step 9: ROC 曲线绘图（micro-average）
n_classes = y_score.shape[1]
y_test_bin = np.eye(n_classes)[y_test_enc]
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr,
                         mode='lines',
                         name=f'micro-average ROC (AUC = {auc_ovr:.2f})'))
fig.update_layout(title='Micro-average ROC Curve (XGBoost + CLR)',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  width=700, height=500)
fig.show()
