import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(121)
tf.random.set_seed(121) #난수값

dataset = load_iris()
col_name = ['sepal_length', 'sepal_width', 'petal_lenth', 'petal_width']
df=pd.DataFrame(data=dataset.data, columns=col_name)
df['target']=dataset.target

# sns.pairplot(df, hue="target")
# plt.show()

문제 = df.drop('target', axis=1)
답안 = df['target']

# train_test_split
# rf_model = RandomForestClassifier()
# rf_model.fit
# rf_model.predict
# accuracy_score

model = Sequential()
model.add(Dense(units=16, input_dim=4, activation="relu"))
#units는 동그라미를 뜻한다.
#input dimension은 4개가 들어옴 = target 뺀 나머지 피쳐
#시그모이드는 백프로파게이션 하다보면 값이 사라질 수 있음 따라서 히든에선 렐루를 사용한다.
model.add(Dense(units=8, activation="relu")) #input_dim=16
model.add(Dense(units=4, activation="relu")) #input_dim=8
model.add(Dense(units=3, activation="softmax"))
# 결과값이 0,1,2 중 한개니까 3 - 종의 개수를 세는게 맞음//sigmoid는 0또는 1이므로 1로 주면 된다

# l o m 설정
model.compile(optimizer='adam', #비용을 최소로 만들게 해줌 ex) GD(경사하강, U), SGD, momentum, adam
              loss='sparse_categorical_crossentropy', #원핫인코딩을 안 했기 때문에
              #손실함수 cost loss, loss 점수가 제일 작은게 좋음(0에 가까울 수록 좋음)
              metrics=["accuracy"])

# x=None,
# y=None,
# batch_size=None, -> 총 데이터를 몇번으로 나눌 것인가.
# epochs=1, -> 처음부터 끝까지 읽은 횟수
# verbose=1,
# callbacks=None, -> 학습하다가 나아질 기미가 안 보이면 멈추게 하는 기능능# validation_split=0.,
# validation_split=0.2 혹은
# validation_data=(문제_테스트20, 답안_테스트20),

model.fit(x=문제, y=답안, epochs = 200, validation_split = 0.2)
# res.history
loss = model.history.history['loss'] #학습
vloss = model.history.history['val_loss'] #학습 한 거 검증
#loss가 계속 줄고 있는데 vloss가 어느순간 올라간 경우 -> 과적합

# plt.plot(np.arange(1, 201), loss, label="loss")
# plt.plot(np.arange(1, 201), vloss, label="val_loss")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.show()

#print(model.evaluate())

acc = model.history.history['accuracy']
vacc = model.history.history['val_accuracy']

plt.plot(np.arange(1, 201), acc, label="acc")
plt.plot(np.arange(1, 201), vacc, label="val_acc")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

accuracy = model.evaluate(문제, 답안)
print("epoch 200회 평균 acc 점수:", acc)
