import Tools
from mxnet import gluon,init,nd
from mxnet.gluon import nn

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels,kernel_size=3,padding=1))
    return blk

class DenseBlock(nn.Block):
    def __init__(self,num_convs,num_channels,**kwargs):
        super(DenseBlock,self).__init__(**kwargs)
        
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))
            
    def forward(self,X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X,Y,dim=1)
        return X
        
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),nn.Activation('relu'),
            nn.Conv2D(num_channels,kernel_size=1),
            nn.AvgPool2D(pool_size=2,strides=2))
    return blk

    
#net = DenseBlock(2,10)
#net.initialize()
#print(net.params)
#X = nd.random.uniform(shape=(4,3,8,8))
#Y = net(X)
#print(Y.shape)
#
#blk = transition_block(10)
#blk.initialize()
#print(blk(Y).shape)

net = nn.Sequential()
net.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
       nn.BatchNorm(),nn.Activation('relu'),
       nn.MaxPool2D(pool_size=3,strides=2,padding=1)) 

num_channels,growth_rate = 64,32
num_convs_in_dense_block = [4,4,4,4]

for i, num_convs in enumerate(num_convs_in_dense_block):
    net.add(DenseBlock(num_convs,growth_rate))
    num_channels += num_convs*growth_rate
    
    if i != len(num_convs_in_dense_block)-1:
        net.add(transition_block(num_channels//2))

net.add(nn.BatchNorm(),nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
    
lr, num_epochs, batch_size, ctx = 0.1, 5, 256, Tools.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = Tools.load_data_fashion_mnist(batch_size, resize=96)
Tools.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
num_epochs) 
    
    
#epoch 1, loss 0.5301, train acc 0.811, test acc 0.869, time 1446.8 sec
#epoch 2, loss 0.3139, train acc 0.886, test acc 0.890, time 1486.9 sec
#epoch 3, loss 0.2638, train acc 0.904, test acc 0.849, time 1498.0 sec
#epoch 4, loss 0.2327, train acc 0.915, test acc 0.907, time 1547.1 sec
#epoch 5, loss 0.2096, train acc 0.924, test acc 0.889, time 1524.3 sec
    
    
    