from mxnet import autograd,nd
from matplotlib import pyplot as plt
import random

num_inputs = 2
num_examples = 1000
true_w = nd.array([2,-3.4])
true_b = 4.2
#scale表示方差，均值为0
features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
labels = nd.dot(features,true_w)+true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

#def set_figsize(figsize=(3.5,2.5)):
#    #设置图片像素
#    plt.rcParams['figure.figsize'] = figsize
#set_figsize()
#plt.figure(figsize=(10,6))
#plt.scatter(features[:,1].asnumpy(),labels.asnumpy(),1)

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    #0-num_examples的一维数组
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = nd.array(indices[i:min(i+batch_size,num_examples)])
        #yield是有return功能的生成器，再次进入程序的时候，从断上次断掉的地方开始执行掉
        yield features.take(j),labels.take(j)

batch_size = 10
#for X,y in data_iter(batch_size,features,labels):
#    print(X,y)
#    break

w = nd.random.normal(scale=0.01,shape=(num_inputs,1))
b = nd.zeros(shape=(1,))

w.attach_grad()
b.attach_grad()

#模型
def linreg(X,w,b):
    return nd.dot(X,w)+b

#损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

#优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param[:] = param-lr*param.grad/batch_size
        
#开始训练
lr = 0.03
num_epochs = 3#迭代次数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        with autograd.record():#与前面的attach_grad()
            l = loss(net(X,w,b),y)
        l.backward()#反向计算，梯度
        sgd([w,b],lr,batch_size)
    train_l = loss(net(features,w,b),labels)
    print('epoch %d, loss %f'%(epoch+1,train_l.mean().asnumpy()))
    
print('estimating w',true_w-w.reshape(true_w.shape))
print('estimating b',true_b-b)