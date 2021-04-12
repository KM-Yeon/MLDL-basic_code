# 1. EDA 차트 그리기 (생략)
#   0) Dataframe 살펴보기
#   1) 차트그리기
# 2. 결측치/이상치 --> 처리
# 3. 피쳐 전처리/가공
# 4. 학습/평가   -> 불균형 데이터기 때문에 accuracy는 의미X / f1&auc
# 5. 검증(GridSearchCV, confusion_matrix)

import pandas as pd
from pkg.common.my_common import *

df = pd.read_csv("creditcard22.csv")
# df1 = df[df['Class']==1].iloc[:200,:]
# df2 = df[df['Class']==0].iloc[:20000,:]
# df = df1.append(df2)
# df.to_csv("creditcard22.csv", index=None)

X, y = CHECK_DATAFRAME(df, "Class")

CHECK_NULL(X)
CHECK_ZERO(X) #Time, Amount --> zero값 존재

#CHART_HEATMAP(df, "Class", 10)
#Time 삭제 -->Time만 유독 도드라지게 class와 관련이 있기때문에 time만 보고 결정할까봐 빼버림
#양의 상관관계: v11 v4 v2
#음의 상관관계: v12, v14, v10, v17, v3
df.drop("Time", axis=1, inplace=True)

model = RandomForestClassifier();
print(X.shape, y.shape)
SPLIT_FIT_PREDICT(model, X, y, str="최초 점수", rate=0.2)

# #-----IQR
# cols = ["V11", "V4", "V2", "V12", "V14", "V10", "V17", "V3"]
# for i, col in enumerate(cols):
#     oulier_idx = CHECK_OUTLIER(df=df, column=col)
#     print(col, oulier_idx)
#    #df.drop(outlier_idx, axis=0, inplace=True)
#
# #'V11', 'V14' --> 주요 영향도 있는 피쳐만 아웃라이어 드러내자
# outlier_idx = CHECK_OUTLIER(df=df, column="V14")
# df.drop(outlier_idx, axis=0, inplace=True)
# y = df["Class"]
# X = df.drop("Class", axis=1)
# SPLIT_FIT_PREDICT(model, X, y, str="V14 outlier 제거 후 점수", rate=0.2)
#
# #-----box plot
# # fig, axs = plt.subplots(figsize=(20, 8), ncols=5, nrows=2)
# # for i, col in enumerate(cols):
# #     r = int(i/5)
# #     c = i % 4
# #     sns.boxplot(x="Class", y=col, data=df, ax=axs[r][c])
# # plt.show()
#
# scaler = StandardScaler() #--> 이미 scaler된 자료이므로 PCA 안 한 amount만 스케일링 한다.
# X['Amount'] = scaler.fit_transform(X["Amount"].values.reshape(-1,1))  #V... PCA : null, 정규, 스케일
# X_train, X_test, y_train, y_test = SPLIT_FIT_PREDICT(model, X, y, str="스케일링 적용 후 점수", rate=0.2, prc=True, roc=True)
# 스케일링 하니까 점수가 떨어짐 ^_ㅜ.....

# my_hyper_param = {
#     "n_estimators"      :[100], #n_estimators: 랜덤 포레스트 안의 결정 트리 갯수
#     "max_depth"         :[3,5,7,9],
#     "min_samples_leaf"  :[1,3,5],
#     "random_state"      :[121,]
# }
# GRIDSEARCHCV(my_hyper_param, model, X_train, y_train)


#--------------------------------
#UnderSampling : iloc
#--------------------------------
print(df.shape)
print(df["Class"].value_counts())

P_df = df[df["Class"] ==1]
N_df = df[df["Class"] ==0][:len(P_df)]
under_df = P_df.append(N_df)
print(P_df.shape, N_df.shape, under_df.shape)

y = under_df["Class"]
X = under_df.drop("Class", axis=1)

model = RandomForestClassifier();
print(X.shape, y.shape)
SPLIT_FIT_PREDICT(model, X, y, str="언더샘플링 점수", rate=0.2)

#--------------------------------
#OverSampling  : SMOTE  NearMiss
#--------------------------------
from imblearn.over_sampling import SMOTE

print(df.shape)
print(df["Class"].value_counts())

y = df["Class"]
X = df.drop("Class", axis=1)
smote = SMOTE()  #Over-sampling
X_over, y_over = smote.fit_resample(X, y)

print(X_over.shape, y_over.shape)
print(pd.Series(y_over).value_counts())

model = RandomForestClassifier();
print(X.shape, y.shape)
SPLIT_FIT_PREDICT(model, X_over, y_over, str="오버샘플링 점수", rate=0.2)

#우리 이 데이터로는 오버샘프링이 잘 맞음...





