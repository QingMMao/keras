from mxnet import autograd,nd

#复制Linear Regression的初始化
num_inputs = 2
num_examples = 1000
true_w = nd.array([2,-3.4])
true_b = 4.2
features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
labels = nd.dot(features,true_w)+true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

from mxnet.gluon import data as gdata

batch_size = 10
#获得属性和标签
dataset = gdata.ArrayDataset(features,labels)
#下载打乱的数据
data_iter = gdata.DataLoader(dataset,batch_size,shuffle=True)

from mxnet.gluon import nn

#是序贯模型
net = nn.Sequential()
#输出值得个数为1,不需要input_shape
net.add(nn.Dense(1))

from mxnet import init

#初始化权重和偏置等，均值为0，方差为0.01
net.initialize(init.Normal(sigma=0.01))

from mxnet.gluon import loss as gloss

#差的平方和的一半
loss = gloss.L2Loss()

from mxnet import gluon

#梯度下降法
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        with autograd.record():
            l = loss(net(X),y)
            print(l)
        l.backward()
        trainer.step(batch_size)
    train_l = loss(net(features),labels)
    print('epoch %d, loss %f'%(epoch,train_l.mean().asnumpy()))
    
w = net[0].weight.data()
b = net[0].bias.data()
print('Error in estimating w',true_w-w.reshape(true_w.shape))
print('Error in estimating b',true_b-b)