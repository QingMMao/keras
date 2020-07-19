from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#设置学习率，迭代次数，每次显示包含的数量
lr,num_epochs,batch_size = 0.01,1000,50

#初始化训练数据
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
#训练数据的数量
n_samples = train_X.shape[0]

#这里的X,Y要与下文feed_dict的X,Y一致
X = tf.placeholder('float')
Y = tf.placeholder('float')

#通过创建Variable类的实例向graph中添加变量。Variable()需要初始值，一旦初始值确定，那么该变量的类型和形状都确定了。
W = tf.Variable(np.random.randn(),name='weight')
b = tf.Variable(np.random.randn(),name='bias')

#定义两个函数
pred = tf.add(tf.multiply(X,W),b)

cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

#梯度下降作为优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

#训练自己的神经网络时，加上这一行
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #初始化
    sess.run(init)
    
    #训练
    for epoch in range(num_epochs):
        for (x,y) in zip(train_X,train_Y):
            #feed_dict里面的X,Y要和自己定义的x,y分开
            sess.run(optimizer,feed_dict={X:x,Y:y})#因为其他的W,b已经定好的，其余未定的就只有X,Y，因此只用传入X,Y
            
        #显示
        if (epoch+1)%batch_size == 0:
            c = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print('Epoch:' ,(epoch+1),'cost=','{:.9f}'.format(c),
                  'W=',sess.run(W),'b=',sess.run(b))
            
    print('Optimization Finished!')
    #ssess.run()过的，就可以直接print，不用再次sess.run()
    train_cost = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
    print('Training cost=',train_cost,'W=',sess.run(W),'b=',sess.run(b))
    
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    #把多个axes放在一起
    plt.legend()
    plt.show()
    
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    
    print('Testing...(Mean square loss Comparison)')
    
    #必须得传参数，不能直接代替，test_X.shape[0]没有出现在定义的函数中，不能用形参代替
    test_cost = sess.run(tf.reduce_sum(tf.pow(pred-Y,2))/(2*test_X.shape[0]),
                         feed_dict={X:test_X,Y:test_Y})
    print('Testing cost=',test_cost)
    print('Absolute mean square loss difference:',
          abs(train_cost-test_cost))
    
    plt.plot(test_X,test_Y,'bo',label='Testing data')
    plt.plot(test_X,sess.run(W)*test_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
    
    
    