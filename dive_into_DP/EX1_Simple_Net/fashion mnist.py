from matplotlib import pyplot as plt
import Tools
from mxnet import autograd,nd

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1,keepdims=True)
    return X_exp/partition

def net(X):
    return softmax(nd.dot(X.reshape((-1,num_inputs)),w)+b)#-1是将一维转为二维的

#X,y = mnist_train[0:9]
#Tools.show_fashion_mnist(X,Tools.get_fashion_mnist_labels(y))
    
batch_size = 256
train_iter,test_iter = Tools.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

w = nd.random.normal(scale=0.01,shape=(num_inputs,num_outputs))
b = nd.zeros(num_outputs)

w.attach_grad()
b.attach_grad()

def cross_entropy(y_hat,y):
    return -nd.pick(y_hat,y).log()

def accuracy(y_hat,y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()#y_hat最大值的Index。y的值转化为float32格式。

num_epochs,lr = 5,0.1

def train(net,train_iter,test_iter,loss,num_epochs,batch_size,
          params=None,lr=None,trainer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            if trainer is None:
                Tools.sgd(params,lr,batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = Tools.evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train accuracy %.3f,test acc %.3f'%
              (epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))
    
train(net,train_iter,test_iter,cross_entropy,num_epochs,
      batch_size,[w,b],lr)

def show_fashion_mnist(images, labels):
    """Plot Fashion-MNIST images with labels."""
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    #如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        
for X,y in test_iter:
    break
true_labels = Tools.get_fashion_mnist_labels(y.asnumpy())
pres_labels = Tools.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [truelabel+'\n'+prelabel 
          for truelabel,prelabel in zip (true_labels,pres_labels)]
Tools.show_fashion_mnist(X[0:9],titles[0:9])











