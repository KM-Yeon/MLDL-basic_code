from sklearn.preprocessing import Binarizer

X = [
    [ 1., -1.,  2.],
    [ 2.,  0.,  0.],
    [ 0.,  1., -1.]
]

bn = Binarizer(threshold=0.0) #threshold = 임계치 / Binarizer는 0보다 크면 1 작으면 0 --> 임계치 조정해주는 함수
fit = bn.fit(X)  # 0.0보다 큰지 작은지 비교
print(fit)
trans = fit.transform(X)
fit_trans = bn.fit_transform(X)
print(fit_trans)