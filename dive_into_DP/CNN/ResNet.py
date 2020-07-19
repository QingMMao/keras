import Tools
from mxnet import gluon, init, nd
from mxnet.gluon import nn

class Residual(nn.Block):
    def __init__(self,num_channels,use_1x1conv=False,strides=1,**kwargs):
        super(Residual,self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels,kernel_size=3,padding=1,strides=strides)
        self.conv2 = nn.Conv2D(num_channels,kernel_size=3,padding=1)
        
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        
    def forward(self,X):
        
        
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y+X)
    

def resnet_block(num_channels,num_residuals,first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.add(Residual(num_channels,use_1x1conv=True,strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


net = nn.Sequential()
net.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
        nn.BatchNorm(),nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1),
        
        resnet_block(64,2,first_block=True),
        resnet_block(128,2),
        resnet_block(256,2),
        resnet_block(512,2),
        
        nn.GlobalAvgPool2D(),
        nn.Dense(10))

lr, num_epochs, batch_size, ctx = 0.05, 5, 256, Tools.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = Tools.load_data_fashion_mnist(batch_size, resize=96)
Tools.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
num_epochs)

#epoch 1, loss 0.4822, train acc 0.831, test acc 0.895, time 2102.0 sec
#epoch 2, loss 0.2538, train acc 0.908, test acc 0.904, time 2099.0 sec
#epoch 3, loss 0.1897, train acc 0.931, test acc 0.884, time 2117.8 sec
#epoch 4, loss 0.1434, train acc 0.948, test acc 0.906, time 2086.2 sec
#epoch 5, loss 0.1040, train acc 0.964, test acc 0.918, time 2094.6 sec










