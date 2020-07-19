#from mxnet import gluon,init
#from mxnet.gluon import loss as gloss,nn
#import Tools
#
##每次提取256个数据进行训练
#batch_size = 256
#train_iter, test_iter = Tools.load_data_fashion_mnist(batch_size)
#
#net = nn.Sequential()
#net.add(nn.Dense(10))
#net.initialize(init.Normal(sigma=0.01))
#
##交叉验证
#loss = gloss.SoftmaxCrossEntropyLoss()
##梯度下降法
#trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})
##迭代次数
#num_epochs = 5
#
#Tools.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,
#                None,trainer)

#This is Multilayer Perception
from mxnet import gluon,init
from mxnet.gluon import loss as gloss,nn
import Tools

#每次提取256个数据进行训练
batch_size = 256
train_iter, test_iter = Tools.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(256,activation='relu'))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

#交叉验证
loss = gloss.SoftmaxCrossEntropyLoss()
#梯度下降法
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})
#迭代次数
num_epochs = 10

Tools.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,
                None,trainer)