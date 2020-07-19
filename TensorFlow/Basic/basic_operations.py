import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

#加载会话
with tf.Session() as sess:
    print('a:%i'%sess.run(a),'b:%i' % sess.run(b))
    print('Add with constants:%i' % sess.run(a+b))
    print('Multi with constants:%i' % sess.run(a*b))

#placeholder是占位符节点，一种常量，由用户在调用run方法时传递常数值
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

#申明函数
add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
    print('Add with variables:%i' % sess.run(add,feed_dict={a:2,b:3}))#这里在传参数
    print('Multi with variables:%i' % sess.run(mul,feed_dict={a:2,b:3}))
 
#申明矩阵
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

#矩阵点乘
product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)