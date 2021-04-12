import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

dataset = load_iris()
col_name = ['sepal_length', 'sepal_width', 'petal_lenth', 'petal_width']
df=pd.DataFrame(data=dataset.data, columns=col_name)
df['target']=dataset.target

문제 = df.iloc[:,:-1] #=df.drop('target', axis=1)
답안 = df.iloc[:,-1]  #=df['target']

문제_학습80, 문제_테스트20, 답안_학습80, 답안_테스트20 \
    = train_test_split(문제, 답안, test_size=0.2, random_state=121, shuffle=True)

dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()

models = [dt_model, knn_model, rf_model]

for model in models:
    # fit : 학습하다
    model.fit(문제_학습80, 답안_학습80)

    # predict : 시험
    예측답안20 = model.predict(문제_테스트20)

    # score : 예측 정확도 확인
    score = accuracy_score(답안_테스트20, 예측답안20)

    print(model.__class__)
    print(model.__str__(), ":", score)  # 0.9666666666666667