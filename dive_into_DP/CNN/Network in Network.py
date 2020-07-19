import Tools
from mxnet import gluon,init,nd
from mxnet.gluon import nn

def nin_block(num_channels,kernel_size,strides,padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels,kernel_size,strides,padding,activation='relu'),
            nn.Conv2D(num_channels,kernel_size=1,activation='relu'),
            nn.Conv2D(num_channels,kernel_size=1,activation='relu'),)
    return blk

net = nn.Sequential()
net.add(nin_block(96,kernel_size=11,strides=4,padding=0),
        nn.MaxPool2D(pool_size=3,strides=2),
        
        nin_block(256,kernel_size=5,strides=1,padding=2),
        nn.MaxPool2D(pool_size=3,strides=2),
        
        nin_block(384,kernel_size=3,strides=1,padding=1),
        nn.MaxPool2D(pool_size=3,strides=2),
        
        nn.Dropout(0.5),
        
        nin_block(10,kernel_size=3,strides=1,padding=1),
        
        nn.GlobalAvgPool2D(),
        
        nn.Flatten())

#X = nd.random.uniform(shape=(1, 1, 224, 224))
#net.initialize()
#for layer in net:
#    X = layer(X)
#    print(layer.name, 'output shape:\t', X.shape)

print('NiN begin training...')
lr, num_epochs, batch_size, ctx = 0.1, 5, 128, Tools.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = Tools.load_data_fashion_mnist(batch_size, resize=224)
Tools.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
num_epochs)
print('NiN end!')

#training on cpu(0)
#epoch 1, loss 2.0399, train acc 0.249, test acc 0.498, time 1475.0 sec
#epoch 2, loss 1.1589, train acc 0.603, test acc 0.693, time 2036.1 sec
#epoch 3, loss 1.1399, train acc 0.578, test acc 0.576, time 1943.9 sec
#epoch 4, loss 0.8011, train acc 0.701, test acc 0.732, time 1308.9 sec
#epoch 5, loss 0.5560, train acc 0.796, test acc 0.838, time 1307.6 sec