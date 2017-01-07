import tensorflow as tf

trX = [1, 2, 3]
trY = [4, 5, 6]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(0.0, name='W')
b = tf.Variable(tf.zeros(1))

Y_model = W * X + b
cost = tf.square(Y_model - Y)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(W), sess.run(b))
