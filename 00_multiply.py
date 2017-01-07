import tensorflow as tf

# 创建变量a,b
a = tf.placeholder('float')
b = tf.placeholder('float')

# y=a*b
y = tf.mul(a, b)

with tf.Session() as sess:
    print(sess.run(y, feed_dict={a: 1, b: 2}))
    print(sess.run(y, feed_dict={a: 10, b: 100}))
