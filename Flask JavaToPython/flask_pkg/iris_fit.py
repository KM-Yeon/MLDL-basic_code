
from sklearn.datasets import load_iris

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, fbeta_score

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold

dataset = load_iris()
print(dataset.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

print(dataset.feature_names)    #X컬럼이름
print(dataset.data[:5])         #X값

print(dataset.target_names)
print(dataset.target[:10])      #y값

# print(dataset.DESCR)

#sepal length (cm)  sepal width (cm)  ...  petal width (cm)

df = pd.DataFrame(data=dataset.data,
                  #columns=dataset.feature_names
                  columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                  )
df["target"] = dataset.target
print(df.head())
print(df.tail())

print(df.info())
print(df.shape)

X = df.iloc[: , :-1]
y = df.iloc[: , -1]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=160)
model = LogisticRegression()
model.fit(X_train, y_train)

fpr0, tpr0, thresholds0 = roc_curve(y_test, model.decision_function(X_test)[:, 0], pos_label=0)
fpr1, tpr1, thresholds1 = roc_curve(y_test, model.decision_function(X_test)[:, 1], pos_label=1)
fpr2, tpr2, thresholds2 = roc_curve(y_test, model.decision_function(X_test)[:, 2], pos_label=2)

print(fpr0, tpr0, thresholds0)

plt.plot(fpr0, tpr0, "r-", label="class 0 ")
plt.plot(fpr1, tpr1, "g-", label="class 1")
plt.plot(fpr2, tpr2, "b-", label="class 2")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
plt.xlim(-0.05, 1.0)
plt.ylim(0, 1.05)
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()
plt.savefig("./static/curv.jpg")


import pickle
pickle.dump(model, open('mymodel_iris.pkl', 'wb'))
# model.save('mymodel.h5')