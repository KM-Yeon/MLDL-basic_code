# Pregnancies: 임신 횟수
# Glucose: 포도당 부하 검사 수치
# BloodPressure: 혈압(mm Hg)
# SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
# Insulin: 혈청 인슐린(mu U/ml)
# BMI: 체질량지수 (체중(kg) / 키(m)^2)
# DiabetesPedigreeFunction: 당뇨 내력 가중치 값
# Age: 나이
# Outcome: 클래스 결정 값 (0 또는 1) --> target

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score,precision_score,recall_score, roc_curve, classification_report,precision_recall_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Binarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings(action="ignore")

np.random.seed(121)
tf.random.set_seed(121)

df = pd.read_csv("diabetes.csv")

#----------------------DataFrame 확인
print(df.shape)  #(768, 9)
print(df.info()) #결측 X, Object X
print(df.head())
print(df["Outcome"].value_counts())

#df.hist()
#plt.show()

# 1. 이상치 발견(0값)
# 2. 나이 --> 구간화
# 3. 편중된 데이터 정규화(스케일링/아웃라이어 삭제)
# 4. 타겟: Outcome(0/1)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
print(X.shape, y.shape)

#-- 분석하기 좋은 데이터: // 정규화 하는 것은 자료가 좀 더 좋게 나오게 하는 것이다.
#-- 결측(X): isnull(), dropna(), fillna()
#-- Object(X): oh.Encoding()->(010...), pd.getDummy()--> 결측처리+인코딩(글자->수치)

#sns.heatmap(df.corr(), annot=True, fmt=".2g", cmap="OrRd") #대략적인 관계 확인
#plt.show()

model = Sequential()
model.add(Dense(units=16, input_dim=8, activation="relu"))

model.add(Dense(units=8, activation="relu")) #input_dim=16
model.add(Dense(units=4, activation="relu")) #input_dim=8
model.add(Dense(units=1, activation="sigmoid"))

# l o m 설정
model.compile(optimizer='adam', #비용을 최소로 만들게 해줌 ex) GD(경사하강, U), SGD, momentum, adam
              loss='binary_cross_entropy',
              #손실함수 cost loss, loss 점수가 제일 작은게 좋음(0에 가까울 수록 좋음)
              metrics=["accuracy"])

#----------- 최초점수
model.fit(x=X, y=y, epochs = 200, validation_split = 0.2)
# res.history
loss = model.history.history['loss'] #학습
vloss = model.history.history['val_loss'] #학습 한 거 검증
#loss가 계속 줄고 있는데 vloss가 어느순간 올라간 경우 -> 과적합

plt.plot(np.arange(1, 201), loss, label="loss")
plt.plot(np.arange(1, 201), vloss, label="val_loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()



# # 데이터 전처리(Data Preprocessing)
# # 스케일링/정규화, Object-->수치변환, 아웃라이어, 피쳐병합/삭제, 구간화(범주화)
#
# #------------------------------------------
# # 아웃라이어/특이값 : 0값 처리
# # 1.row 삭제  2.채우기(평균, 최빈도, 중위값)(V) 3.예측해서 채우기
# #------------------------------------------
# #gn = X["Glucose"].nonzero()
# for col in X.columns:
#     gcnt = X[X[col] ==0][col].count()
#     print(col, gcnt, np.round(gcnt/X.shape[0]*100, 2))
#
# # Pregnancies 111 --> 임신은 0인게 이상하지 않음
# # Glucose 5 --> 이상
# # BloodPressure 35 --> 이상
# # SkinThickness 227 --> 이상
# # Insulin 374 --> 이상
# # BMI 11 --> 이상
# # DiabetesPedigreeFunction 0 0.0 --> 문제 없음
# # Age 0 0.0 --> 문제 없음
#
# zero_column = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
# zero_column_mean = X[zero_column].median().round(1)
# X[zero_column] = X[zero_column].replace(0, zero_column_mean)
#
# print(df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].describe())
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)
# rf_model.fit(X_train, y_train)
# pred = rf_model.predict(X_test)
# get_score(y_test, pred, "아웃라이어/특이값(0) 처리 후 점수")
#
# #------------------------------------------
# # 나이 구간화(범주화) / 인코딩 25~75세
# #------------------------------------------
# X['Age_cate'] = X['Age'].apply(lambda x : int(x//10))
# X_encoding = pd.get_dummies(data=X, columns=["Age_cate"], prefix = "OH_Age_cate")  #, drop_first = True
# print(X.info())
# # print(X[["Age_cate","Age"]].head()) #인코딩 후 Age_cate는 자동 삭제 -- drop_first = False 동작X
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121)
# rf_model.fit(X_train, y_train)
# pred = rf_model.predict(X_test)
# get_score(y_test, pred, "Age 인코딩/범주화 후 점수")
#
# #------------------------------------------
# # 스케일링/정규화 : RobustScaler, MinMaxScaler, StandardScaler
# # 아웃라이어 처리하고 사용해라 --> (대부분은)효과를 본다
# #------------------------------------------
# scaler = StandardScaler()
# X_scaler = scaler.fit_transform(X)
#
# X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=121)
# rf_model.fit(X_train, y_train)
# pred = rf_model.predict(X_test)
# get_score(y_test, pred, "스케일링/정규화 후 점수")
# # F1이 낮아진다면?: 그 전이 너무 과적합 됐었기 때문 -> 정규화 함으로서 다른 데이터를 받았을 때 그것도 잘 맞추게 되는 것
#
# #------------------------------------------
# # precision_recall_curv : 임계치 확인
# #------------------------------------------
# proba = rf_model.predict_proba(X_test)
# print(pred[:10])  # 임계값 0.5 기준
# print(proba[:10]) #negative와 positive중에 어떤걸 밀어줘서(왼쪽: 0/오른쪽: 1) pred 값이 나오는지 알려줌
#
# precision, recall, th = precision_recall_curve(y_test, proba[:,1])
# #proba[:,1] -> 양의 값만 찾아와라/어차피 임계값하고 비교라 음의 값까지 필요 없음
# print(len(precision), len(recall), len(th)) #67 67 66 --> 개수가 다름
# plt.plot(th, precision[:len(th)], label="precision")
# plt.plot(th, recall[:len(th)], label="recall")
# plt.xlabel("threadshold")
# plt.ylabel("precision & recall value")
# plt.legend() #plt.legend(["precision", "recall"])
# plt.grid()
# plt.show()
#
# #------------------------------------------
# # roc_auc_curve : FPR / TPR 비율
# #------------------------------------------
# fpr, tpr, th = roc_curve(y_test, proba[:,1])
# auc = roc_auc_score(y_test, proba[:, 1].reshape(-1, 1))
# plt.plot(fpr, tpr, label='ROC')
# plt.plot([0,1], [0,1], label='th:0.5')
# plt.title(auc)
# plt.xlabel("FPR")
# plt.ylabel("TPR(recall)")
# plt.grid()
# plt.show()
#
# #------------------------------------------
# # precision_recall_curve : 임계치 튜닝을 통한 점수 보정
# #------------------------------------------
# my_th = [.4, .43, .45, .47, .49, .51, .53]
# #my_th = [0., 0.2, 0.4, 0.8, 1.0]
#
# for th in my_th:
#     print("N : P =", th, 1 - th)
#     rf_model.fit(X_train, y_train)
#     pred = rf_model.predict(X_test)
#     proba = rf_model.predict_proba(X_test)
#     get_score(y_test, pred)
#
#     bn = Binarizer(threshold=th)
#     # threshold = 임계치 / Binarizer는 0보다 크면 1 작으면 0 --> 임계치 조정해주는 함수
#     fit_trans = bn.fit_transform(proba[:, 1].reshape(-1, 1))
#     # proba[:,1]-> array 값으로 나오니까 다시 matrix로 바꾸기 위해 reshape 한다.
#     # 임계치 올라가면 정밀도가 올라간다.
#     auc = roc_auc_score(y_test, proba[:, 1].reshape(-1, 1))
#     print(auc)







