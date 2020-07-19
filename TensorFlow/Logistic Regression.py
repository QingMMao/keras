import tensorflow as tf

from keras.datasets import fashion_mnist
from keras.utils import to_categorical

(X_train,y_train),(X_test,y_test) = fashion_mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#y_label = np.zeros([len(y_train),10])
#for i in range(len(y_train)):
#    y_label[i][y_train[i]] = 1
#y_train = y_label
#
#y_label = np.zeros([len(y_test),10])
#for i in range(len(y_test)):
#    y_label[i][y_test[i]] = 1
#y_test = y_label

X_train = X_train.reshape((60000,28*28))
X_train = X_train.astype('float32')/255

X_test = X_test.reshape((10000,28*28))
X_test = X_test.astype('float32')/255

lr,num_epochs,batch_size,display_step = 0.01,25,100,1

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b  =tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(num_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        
        for i in range(100):
            batch_xs,batch_ys = X_train[i*total_batch:i*total_batch+batch_size],y_train[i*total_batch:i*total_batch+batch_size]
            print(batch_xs.shape,batch_ys.shape)
            _,c = sess.run([optimizer,cost],feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost += c/total_batch
            
        if (epoch+1)%display_step == 0:
            print("Epoch:",(epoch+1), "cost=", "{:.9f}".format(avg_cost))
            
    print('Optimization Finished')
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({X: X_test[:3000], Y:y_test[:3000]}))