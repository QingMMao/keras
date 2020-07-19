import tensorflow as tf

#Constan被当作默认graph的一个常量节点
hello = tf.constant('Hello,world!')

#开始会话（Session）
sess = tf.Session()

#跑程序
res = sess.run(hello)#这个没输出,但是可以代表输出
print(sess.run(hello))#这个输出是b'Hello,world!'

a = tf.constant(2)#这个的值不是2
c = sess.run(a)#这个就是正常的值2