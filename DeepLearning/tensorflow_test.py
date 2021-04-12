import tensorflow as tf

x = [1,2,3]
y = [1,2,3]

# wx + b
w = tf.Variable(0.3, name='w') # w와 b는 값을 담고 있는게 아니라 주소를 담고 있는 변수이다.
b = tf.Variable(0.2, name='b') # 값은 텐서에 들어있기 때문에 일일이 텐서에 들어가서 초기화하고 실행해줘야 한다.
print(w, b)

# H(x) = wx + b
h = x*w + b

# min(cost) = mse --> mean(sum(y-y^)^2)
cost = tf.reduce_min(tf.square(y - h)) #결과의 최소값 : reduce_min --> 최소제곱을 위함..

# optimizer : train의 cost 비용 최소화
optimizer = tf.train.GradientDecesentOptimizer(0.01)  #비용을 최소화 하게 함
train = optimizer(cost)

sess = tf.Session() # 큰 네모가 세션, 그 안에 노드들이 텐서플로
sess.run(tf.global_variables_initizlizer()) # 텐서플로 초기화

for i in range(100) :
    sess.run(train) # 실행
    print(i, sess.run(train), sess.run(w), sess.run(b))

    0   0.024       0.3        0.3
    1   0.021       0.274      0.26
    ...
    80  0.00000023  1.1004142  -0.000242
    ...
    99  0.000014    1.2000142  0.00000174

    => 최소비용은 80번째꺼고 회귀식은 y = 1.1004142x -0.000242