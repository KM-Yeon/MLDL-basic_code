import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

#-------------------------------------------
# Target 피쳐 선정
#  1   Survived     891 non-null    int64
#-------------------------------------------
df = pd.read_csv("Titanic.csv")
print(df.info())
print(df.shape)
print(df.head())

X = df.drop("Survived", axis=1) #df.iloc[:,:-1]
y = df["Survived"] #df.iloc[:,-1]
print(X[:2])
print(y[:2])
#-------------------------------------------
# Object 처리
#-------------------------------------------
#  3   Name         891 non-null    object -> Sex로 성별 구분 정도로 활용
#  4   Sex          891 non-null    object -> lambda 이용해 female:0, male:1으로 변경
#  8   Ticket       891 non-null    object -> 의미 있는 데이터라 보기 어렵다고 판단
#  10  Cabin        204 non-null    object -> NaN이 아닌 애들만 뽑아오기 -> 결측이 너무 많음(687건) => 버림 / 앞 글자만 빼오기
#  11  Embarked     889 non-null    object

#4
X['Sex'] = X['Sex'].apply(lambda x : 1 if x =='male' else 0)
print(X['Sex'].head())

#글자 1개만 추출(문법공부)
cp = X[X['Cabin'].isnull() == False].copy() #강제 복사 -> 메모리에 따로 복사함..
print(cp['Cabin'].isnull().sum())
X['Cabin2'] = X['Cabin'].str[0:1] #맨 앞 글자만 뺴오겠다. / #.str[0]을 하면 cabin2와 cabin과 모양이 달라서 경고 메시지 뜸
print(X['Cabin2'])

print(pd.crosstab(X['Cabin2'], y).T)
print(pd.crosstab(X['Cabin2'], X['Pclass']).T)
print(pd.crosstab(X['Pclass'], y).T)
print(pd.crosstab([X['Pclass'], X['Sex']], y).T)

#-------------------------------------------
# 결측처리  1.삭제 2.대체 3.예측
#-------------------------------------------
#   5  Age          714 non-null    float64
#  10  Cabin        204 non-null    object
#-------------------------------------------

#Age의 null값을 대체수로 채우기
#이름을 이용하여 Miss, Mrs, Mr 떼어내서 각 존칭 당 평균 나이 구하여 null값 채우기 --> 정규표현식 사용
#X["Age"].fillna()
# cp = X[X["Age"].isnull() == True].copy() #나이가 null 인것들만 가졍괴
# cp['Age2'] = cp['Age'].fillna(55) #.mean()
# print(cp[['Age','Age2']])

#-------------------------------------------
#나이를 예측하기 위해 이름의 호칭 추출  SibSp	Parch
#호칭 별 평균 나이로 Age 결측 데이터 정리
#-------------------------------------------
X["Name2"] = X["Name"].str.extract("([A-Za-z]+)\.")
# fill_mean_func = lambda g: g["Age"].fillna(g.mean())
# X = X.groupby(by=["Name2"]).apply(fill_mean_func)
dict = X.groupby(by=["Name2"])[["Name2","Age"]].mean().astype(np.int32).to_dict()
print(dict['Age'])
print(X[["Name2","Name","Age"]].head(10))
fill_mean_func = lambda gname: gname.fillna(dict['Age'][gname.name])
X = X.groupby('Name2').apply(fill_mean_func)
print(X[["Name2","Name","Age"]].head(10))
# X["Age"] = X["Age"].fillna(30)

# 11 12 13 --> 10
# 22 23 24 --> 20
#나이 구간화    /(나누기) %(나머지) //(몫)
X['Age_cate'] = X['Age'].apply(lambda x : int(x//10))
print(X[['Age_cate', 'Age']])
print(pd.crosstab([X['Pclass'], X['Sex'], X['Age_cate']], y).T)

#-------------------------------------------
#  11  Embarked     889 non-null    object -> 선착장 위치
#-------------------------------------------
print(pd.crosstab(X['Embarked'], y).T) #3등급은 S이용, 2등급은 Q와 S 둘 다 이용
print(pd.crosstab([X['Embarked'], X['Pclass']], y).T) #-> S가 문제(?)라기 보단 3등급 칸이 많이 죽음

X['Embarked'] = X['Embarked'].apply(lambda x : 1 if x == 'C' else (2 if x == 'Q' else 3 ))
print(X['Embarked'])
#-> = 선착장 위치는 중요 변수X = 생존과 무관

# 병합 피쳐 : 중복된특징,
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
X["SP"] = X["SibSp"] + X["Parch"]
#print(X[["SP", "SibSp", "Parch"]])

# 삭제 피쳐 : 일련번호
#  0   PassengerId  891 non-null    int64
print(X.shape)
X.drop("PassengerId", axis=1, inplace=True)
print(X.shape)
print(X.info())

del_col = ["SibSp", "Parch", "Name", "Name2", "Age"] #SP=SibSp+Parch / Age_cate <- Name, Name2, Age
replace_col = ["Ticket", "Fare", "Cabin", "Cabin2", "Embarked"] #ticket 일련번호라 지움, Cabin은 Null이 많음
#Fare <--Pclass,SP로 유추 가능  /  "Embarked"     => 지워도 되고 안 지워도 되고
#기존 삭제 목록들 중 데이터 분석을 통해 "Ticket", "Fare", "Cabin", "Embarked" 추가

replace_col = replace_col + del_col
X.drop(replace_col, axis=1, inplace=True)
print(X.info())

#상관분석 하는 이유
# 1.종속변수에 지대한 영향을 미치는 변수들 찾으려고
# 2.모델 복잡도를 줄이기 위해(독립변수 간 상관도를 보고 비슷한거 줄이기) => 학습 성능 높이려고

# heatmap(
#     data, *,
#     cmap(컬러표)=None, center=None, robust=False,
#     annot=None, fmt=".2g"
heat_df = X.copy()
heat_df["Servvv"] =y
plt.figure(figsize=(10,10))
#sns.heatmap(data = df.corr(), annot=True, fmt=".2f") #상관분석은 object는 자동으로 빼고 수치형만 자동으로 그려줌
sns.heatmap(data = heat_df.corr(), annot=True, fmt=".2f") #전처리 한 값으로 상관 -> 나이는 상관 없어지고 Pclass와 성별(*)이 주요 변수
#plt.show()

#-------------------------------------------
# 분석 1.모델 선정 2.평가척도 3.척도
#-------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()

X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.2, random_state=121, shuffle=True) #리턴값이 여러개

models = [dt_model, knn_model, rf_model]

for model in models:
    # fit : 학습하다
    model.fit(X_train, y_train)

    # predict : 시험
    y_pred = model.predict(X_test)

    # score : 예측 정확도 확인
    score = accuracy_score(y_test, y_pred)

    print(model.__class__)
    print(model.__str__(), ":", score)

#-------------------------------------------
## 분석(예측)력을 저해하는 원인
# 1. 피쳐가 많은 경우        --> 모델의 복잡도가 증가하는 경우(오버피팅 <-> 언더피팅)
# 2. 수치가 큰 경우         --> log, scalling
# 3. 결측데이터(Null)       --> isnull(), fillna() ,print(X.isnull().sum())
# 4. 이상치(Outlier)       --> 협의 후 삭제/대체
# 5. 데이터가 편중          --> 정규분포화
# 6. 피쳐가공 (Object-->변환,  유니크한 일련번호X, 구간(범주)화,  원핫인코딩)
# 7. 데이터 적은 경우        --> 데이터 증강
# 8. 모델이 적절하지 않는 경우 --> 다른 모델 사용, 튜닝(Hyper Parameter)
#-------------------------------------------

# 3. 결측데이터(Null값)           --> isnull(), fillna()
print(X.isnull().sum())

# 4. 이상치(Outlier)
# -------------------------------------
# 4-1. box plot , scatter plot
# -------------------------------------
# fig, axes = plt.subplots(nrows=3, ncols=5)
# columns = df.columns  #[....]
# for i, col in enumerate(columns) :
#     r = int(i / 5)
#     c = i % 5
#     sns.boxplot(x=col, y='Survived', data=df, ax=axes[r][c])
# plt.show()

# -------------------------------------
# 4-2. IQR : 25%~75% 범위 값
# -------------------------------------
def get_outlier(df=None, column=None):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    Q1 = np.percentile(df[column].values, 25)
    Q3 = np.percentile(df[column].values, 75)
    IQR = Q3 - Q1
    IQR_weight = IQR * 1.5
    minimum = Q1 - IQR_weight
    maximum = Q3 + IQR_weight
    outlier_idx = df[column][  (df[column]<minimum) | (df[column]>maximum)  ].index
    return outlier_idx

# 함수 사용해서 이상치 값 삭제
numeric_columns = df.dtypes[df.dtypes != 'object'].index #df['dtypes'][df['dtypes != 'object]]
for i, col in enumerate(numeric_columns):
    outlier_idx = get_outlier(df=df, column=col)
    print(col, outlier_idx)
    #df.drop(outlier_idx, axis=0, inplace=True)

# 5. 데이터가 편중되거나 적은 경우   --> 정규분포화
df.hist(figsize=(20, 5))
#plt.show()

#아래 세개 스케일링 다 해보고 선택함
from sklearn.preprocessing import StandardScaler # 평균 0, 분산 1
from sklearn.preprocessing import MinMaxScaler   # 최대/최소값이 각각 1, 0이 되도록 스케일링. = y값이 0~1 사이
from sklearn.preprocessing import RobustScaler   # 중위값을 사용하기 때문에 outlier 영향을 덜 받는다. #min~median~max

scaler = StandardScaler()
#scaler.fit()
#scaler.transform()
X_scaler = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=121, shuffle=True)

models = [dt_model, knn_model, rf_model]
for model in models :
    # fit : 학습하다
    model.fit(X_train, y_train)
    # predict : 시험
    y_pred = model.predict(X_test)
    # score : 예측 정확도 확인
    score = accuracy_score(y_test, y_pred)
    #-- f1, auc, accuracy, 교차검증
    #-- mse mae rmse
    #-- shilluet
    print(model.__class__)
    print(model.__str__(), ":" , score) #0.9666666666666667


# 6. 피쳐가공 (Object-->변환,  유니크한 일련번호X, 구간(범주)화,  원핫인코딩)
# Object --> 숫자로된 범주형 변수로 바꾼 후 원핫인코딩 가능 (숫자가 크면 무조건 좋다고 생각하는 컴퓨터 때문에 함) -> 단 이산형으로 착각 X

print(X["Age_cate"].head())

print(X.info())
X_encoding = pd.get_dummies(data=X, columns=["Age_cate"], prefix = "OH_Age_cate")  #, drop_first = True
print(X_encoding.info())
print(X_encoding.head())
#     (drop_first)
#      Age_cate   OH_0   OH_1  (891, 12)  +9개추가  --> (891,21)
# 0    1           1      0
# 1    1           0      1

#** 평가 매트릭스           --> conf_matrix, f1_score(), accuracy_score()
#accuracy_score
# f1_score -- precision + recall
#     (scoring = f1_micro / f1_macro)
# roc_auc -- precision + recall --> FNR/TPR


# 7. 데이터 적은 경우        --> 데이터 증강 ==> K-Fold, St.K-Fold, St.K-Fold, GridSearchCV(증강+튜닝)
# ==> 검증(신뢰), 대량의 학습으로 예측이 좋아진다.

#-------------------------------------------------------------------------------
# Cross-Validation ex)KFold, StratifiedKFold, cross_val_score
#-------------------------------------------------------------------------------

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold #문법은 동일

skf = StratifiedKFold(n_splits=5, shuffle=True,  random_state=121)
kf = KFold(n_splits=5, shuffle=True, random_state=121) #n_splits 값을 너무 크게 하면 오버피팅 될 수 있다.
accuracy_score_list = []
f1_score_list = []
#for (idx_train, idx_test) in kf.split(X):
#i = 0
for i, (idx_train, idx_test) in enumerate(kf.split(X)):
    #X_train = df.iloc[idx_train]
    #X_text  = df.iloc[idx_test]
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test] #어차피 인덱스 값이라 X로만 돌려도 상관X
    #-------이하 학습 동일--------------------
    # fit : 학습하다
    rf_model.fit(X_train, y_train)
    # predict : 시험
    y_pred = model.predict(X_test)
    # score : 예측 정확도 확인
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_score_list.append(accuracy)
    f1_score_list.append(f1)
    print(i, ":", accuracy, f1)
    #i = i+1
print("KFold 평균 정확도:", np.mean(accuracy_score_list))
print("KFold 평균 F1:", np.mean(f1_score_list))

#Scoring matrix
from sklearn.model_selection import cross_val_score
#내부적으로 알아서 루프를 돌아서 점수만 받아올 때 편하게 쓰임 = 중간에 개입을 할 수 없음
#다만 원래는 Kfold와 같은 기능이라 값이 같게 나와야 하는데 kfold가  shuffle 기능이 생겨서 kfold보다 정확도가 떨어짐 ㅠ

score_list = cross_val_score(rf_model, X, y, scoring="f1", cv=5, verbose=0)
#score_list = cross_val_score(rf_model, X, y, scoring="accuracy", cv=5, verbose=0)
#타이타닉은 종속변수가 0,1(죽는다 사는다)만 있어서 f1사용
#cv는 몇번 돌리는지를 넣어줌
print("cross_val_score 평균 F1:", np.mean(score_list))
#print("cross_val_score 평균 정확도:", np.mean(score_list))

#cross_val_score은 한 개의 값만 나오기 때문에 여러개를 나오게 하고 싶다면 cross_validate을 사용해야 한다.
from sklearn.model_selection import cross_validate
my_score={"acc":"accuracy", "f1":"f1"}
score_list = cross_validate(rf_model, X, y, scoring=my_score, cv=5, verbose=0)
print("score_list------->", score_list)
score_df = pd.DataFrame(score_list)
print(score_df.head(10))
print("cross_validation 평균 정확도 : " , score_df["test_acc"].mean())
print("cross_validation 평균 f1 : " , score_df["test_f1"].mean())

#GridSearchCV(param  ) --> crss_val_score + 튜닝

#나올 수 있는 case by case를 다 검사한다
#총 loop 횟수: 12 = 1*4*3   #모델 2개 되면 추가한 개수만큼 또 loop 돔 = 24회 / 모델도 추가되면 어떤 모델이 좋은지도 선택해 줌
#ex) 100-3-1 / 100-3-3 / 100-3-5 ...
# bootstrap = True => 복원추출을 하기 때문에 랜덤 포레스트 개수만큼 값도 다 다름

my_hyper_param = {
    "n_estimators"      :[100], #n_estimators: 랜덤 포레스트 안의 결정 트리 갯수
    "max_depth"         :[3,5,7,9],
    "min_samples_leaf"  :[1,3,5],
}

from sklearn.model_selection import GridSearchCV
gcv_model = GridSearchCV(rf_model, param_grid=my_hyper_param, scoring="f1", refit=True, cv=5, verbose=0)
#refit --> cv만큼 돌린 후 가장 좋은 값을 내는 모델을 찾아서 자동으로 반영해주는 역할을 함

# fit : 학습하다
gcv_model.fit(X_train, y_train)

#내부적으로 내가 파라미터 개수를 준 만큼 가장 좋았던 모델을 아래에 던져주므로 학습만 시켜주면 됨
print("best_estimator_", gcv_model.best_estimator_)
print("best_params_",    gcv_model.best_params_)
print("best_score_" ,    gcv_model.best_score_)

#GridSearchCV로 scoring을 여러개 줄 경우
my_score={"acc":"accuracy", "f1":"f1"}
gcv_model = GridSearchCV(rf_model, param_grid=my_hyper_param, scoring=my_score, refit="f1", cv=5, verbose=0)
# fit : 학습하다
gcv_model.fit(X_train, y_train)

print("best_estimator_", gcv_model.best_estimator_)
print("best_params_",    gcv_model.best_params_)
print("best_score_" ,    gcv_model.best_score_)

#다이렉트로 빼는ver
print("GridSearchCV 평균 정확도:" , gcv_model.cv_results_["mean_test_acc"].mean()) #mean_test_(본인의 score 키값)
print("GridSearchCV 평균 F1:"    , gcv_model.cv_results_["mean_test_f1"].mean())

#데이터 프레임으로 만드는ver
gcv_df = pd.DataFrame(gcv_model.cv_results_)
print(gcv_df.info())
print("GridSearchCV 평균 정확도:" , gcv_df["mean_test_acc"].mean())
print("GridSearchCV 평균 F1:"    , gcv_df["mean_test_f1"].mean())

# 8. 모델이 적절하지 않는 경우 --> 다른 모델 사용 ==> XGBoost LightGBM (데이터량 大, 튜닝 어려움)







