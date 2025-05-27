import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.graph_objects as go

# Step 1: 读取数据
X_train = pd.read_csv('data/SourceCM.csv', index_col=0)
X_test = pd.read_csv('data/QueryCM.csv', index_col=0)
y_train = pd.read_csv('data/SourceLabel.csv', index_col=0)
y_test = pd.read_csv('data/QueryLabel.csv', index_col=0)

# Step 2: 数据处理（取属水平，计算相对丰度）
def preprocess_abundance(abu, level):
    abu = abu.T  # 转置
    abu = abu.loc[:, abu.sum(axis=0) > 0]  # 去除全为0的特征
    abu.columns = abu.columns.str.split(';', expand=True)  # 将分类信息拆分为多列
    abu.columns = abu.columns.get_level_values(level)  # 选择指定层级的分类信息
    abu = abu.loc[:, abu.columns.notnull()]  # 去除分类信息为空的特征
    abu = abu.groupby(abu.columns, axis=1).sum()  # 按分类信息聚合
    return abu

# 取属水平（G）
X_train = preprocess_abundance(X_train, 5)
X_test = preprocess_abundance(X_test, 5)

# 对齐测试集的列顺序
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 相对丰度归一化（每行归一）
X_train = X_train.div(X_train.sum(axis=1), axis=0)
X_test = X_test.div(X_test.sum(axis=1), axis=0)

# Step 3: 标签处理
y_train = y_train['Env'].str.split(':', expand=True).iloc[:, -1]
y_test = y_test['Env'].str.split(':', expand=True).iloc[:, -1]
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
n_classes = len(le.classes_)

# Step 4: 特征标准化（SVM要求）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: SVM 模型训练
model = OneVsRestClassifier(SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
model.fit(X_train_scaled, y_train_enc)
y_score = model.predict_proba(X_test_scaled)

# Step 6: ROC & AUC
y_test_bin = np.eye(n_classes)[y_test_enc]
auc_ovr = roc_auc_score(y_test_bin, y_score, multi_class='ovr')

fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                         name=f'SVM ROC (AUC = {auc_ovr:.2f})'))
fig.update_layout(title='Micro-average ROC Curve (SVM)',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  width=700, height=500)
fig.show()
