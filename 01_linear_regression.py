import tensorflow as tf
import numpy as np

# 创建-1~1之间的等差数列
trX = np.linspace(-1, 1, 101)

# 创建一个线性函数,然后算出y的数组
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

# 创建两个占位的变量
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 创建了一个变量 w为0
w = tf.Variable(0.0, name='weights')

# 值为X×w,X变量在这个时候是未知的，那下面这个就是方程咯
y_model = tf.mul(X, w)

# 这里也是个方程，Y是个变量，y_model是个函数，然后用来求了个平方
cost = tf.square(Y - y_model)

# 定义了学习率和怎么减少损失，GradientDescentOptimizer是一个梯度下降优化器。
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))
    writer = tf.train.SummaryWriter('./log/', sess.graph)
