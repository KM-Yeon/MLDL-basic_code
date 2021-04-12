import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = load_iris()
print(dataset.keys())
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.data[:5])
print(dataset.target[:5])
print(dataset.target_names[:5]) #dataset['target_names']
print(dataset.feature_names[:5])

col_name = ['sepal_length', 'sepal_width', 'petal_lenth', 'petal_width']
df=pd.DataFrame(data=dataset.data, columns=col_name)
print(df.shape)
print(df.info())
print(df.head())

df['target']=dataset.target
print(df.tail())

#iloc[행, 열]
문제 = df.iloc[:,:-1] #=df.drop('target', axis=1)
답안 = df.iloc[:,-1]  #=df['target']
print(문제[:2]) #행 2개만 나옴
print(답안[:2])

#                      *arrays(데이터 프레임),
#                      test_size=None,
#                      train_size=None,
#                      random_state=None, --> 값을 넣으면 매번 랜덤으로 섞일 때마다 해당 난수표의 n번 인덱스가 나열 된다.
#                      shuffle=True,
#                      stratify=None

# 데이터 프레임 안에는 문제지(붓꽃 특징)와 답안지(붓꽃 종류)가 함께 있으므로 X와 y로 나눠야 한다.
# 사진 = x / 개 or 고양이 = y     수학 익힘책 train / 중간고사 test

# X_train        X_test           y_train       y_test
# 학습_문제지 80, 테스트_문제지 20, 답안_학습 80, 답안_테스트 20

#train_test_split(df, test_size=0.2, train_size=0.8, random_state=3, shuffle) -> shuffle이 true이기 때문에 자동으로 섞임

문제_학습80, 문제_테스트20, 답안_학습80, 답안_테스트20 \
    = train_test_split(문제, 답안, test_size=0.2, random_state=121, shuffle=True) #리턴값이 여러개

# 분석 모델 / 알고리즘(x)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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