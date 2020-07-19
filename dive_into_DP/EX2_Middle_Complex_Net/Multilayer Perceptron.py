import Tools
from mxnet import nd
from mxnet.gluon import loss as gloss

batch_size = 256
train_iter,test_iter = Tools.load_data_fashion_mnist(batch_size)

num_inputs,num_output,num_hiddens = 784,10,256

W1 = nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01,shape=(num_hiddens,num_output))
b2 = nd.zeros(num_output)

params = [W1,b1,W2,b2]

for param in params:
    param.attach_grad()
    
def relu(X):
    return nd.maximum(X,0)

def net(X):
    #我们不知道X的具体规模，但是想改成num_inputs规模的形式
    #比如从（256,1,28，28）改成了（256,784），784就是num_inputs的值
    X = X.reshape((-1,num_inputs))
    #点乘就是矩阵相乘
    H = relu(nd.dot(X,W1)+b1)
    return nd.dot(H,W2)+b2

loss = gloss.SoftmaxCrossEntropyLoss()

num_epochs,lr = 10,0.5

Tools.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)

for X,y in test_iter:
    break

true_labels = Tools.get_fashion_mnist_labels(y.asnumpy())
pred_labels = Tools.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
title = [truelabel+'\n'+predlabel
         for truelabel,predlabel in zip(true_labels,pred_labels)]
Tools.show_fashion_mnist(X[90:100],title[90:100])