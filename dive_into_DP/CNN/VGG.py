import Tools
from mxnet import gluon,init,nd
from mxnet.gluon import nn

def vgg_block(num_convs,num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels,kernel_size=3,
                          padding=1,activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2,strides=2))
    return blk

conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

def vgg(conv_arch):
    net = nn.Sequential()
    for num_convs,num_channels in conv_arch:
        net.add(vgg_block(num_convs,num_channels))
    net.add(nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
            nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
            nn.Dense(10))
    return net

#net = vgg(conv_arch)
#net.initialize()
#X = nd.random.uniform(shape=(1,1,224,224))
#for sequential in net:
#    X = sequential(X)
#    print(sequential.name,'shape:',X.shape)
    
ratio = 4
small_conv_arch = [(pair[0],pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size, ctx = 0.05, 5, 128, Tools.try_gpu()
net.initialize(ctx=ctx,init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_iter,test_iter = Tools.load_data_fashion_mnist(batch_size,resize=224)
Tools.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)

print('End')