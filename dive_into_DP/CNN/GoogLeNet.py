import Tools
from mxnet import gluon,init,nd
from mxnet.gluon import nn

class Inception(nn.Block):
    def __init__(self,c1,c2,c3,c4,**kwargs):
        super(Inception,self).__init__(**kwargs)
        self.p1_1 = nn.Conv2D(c1,kernel_size=1,activation='relu')
        
        self.p2_1 = nn.Conv2D(c2[0],kernel_size=1,activation='relu')
        self.p2_2 = nn.Conv2D(c2[1],kernel_size=3,padding=1,activation='relu')
        
        self.p3_1 = nn.Conv2D(c3[0],kernel_size=1,activation='relu')
        self.p3_2 = nn.Conv2D(c3[1],kernel_size=5,padding=2,activation='relu')
        
        self.p4_1 = nn.MaxPool2D(pool_size=3,strides=1,padding=1)
        #下面这个不行
        #self.p4_1 = nn.MaxPool2D(pool_size=3,padding=1)
        self.p4_2 = nn.Conv2D(c4,kernel_size=1,activation='relu')
        
    def forward(self, X):
        p1 = self.p1_1(X)
        p2 = self.p2_2(self.p2_1(X))
        p3 = self.p3_2(self.p3_1(X))
        p4 = self.p4_2(self.p4_1(X))
        
        return nd.concat(p1,p2,p3,p4)
    
net = nn.Sequential()
net.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1),
        nn.Conv2D(64,kernel_size=1),
        nn.Conv2D(192,kernel_size=3,padding=1),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1),
        
        Inception(64,(92,128),(16,32),32),
        Inception(128,(128,192),(32,96),64),
        
        nn.MaxPool2D(pool_size=3,strides=2,padding=1),
        
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        nn.MaxPool2D(pool_size=3,strides=2,padding=1),
        
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        nn.GlobalAvgPool2D(),
        
        nn.Dense(10))

#X = nd.random.uniform(shape=(1, 1, 96, 96))
#net.initialize()
#for layer in net:
#    X = layer(X)
#    print(layer.name, 'output shape:\t', X.shape)

lr, num_epochs, batch_size, ctx = 0.1, 5, 128, Tools.try_gpu()
net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_iter,test_iter = Tools.load_data_fashion_mnist(batch_size,resize=96)
Tools.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)


#[01:17:32] src/operator/nn/mkldnn/mkldnn_base.cc:68: Allocate 56623104 bytes with malloc directly
#epoch 1, loss 2.1330, train acc 0.218, test acc 0.495, time 800.9 sec
#[01:30:53] src/operator/nn/mkldnn/mkldnn_base.cc:68: Allocate 56623104 bytes with malloc directly
#epoch 2, loss 0.7604, train acc 0.710, test acc 0.802, time 897.1 sec
#[01:45:51] src/operator/nn/mkldnn/mkldnn_base.cc:68: Allocate 56623104 bytes with malloc directly
#epoch 3, loss 0.4740, train acc 0.824, test acc 0.844, time 1207.0 sec
#[02:05:58] src/operator/nn/mkldnn/mkldnn_base.cc:68: Allocate 56623104 bytes with malloc directly
#epoch 4, loss 0.3863, train acc 0.855, test acc 0.864, time 1100.5 sec
#[02:24:18] src/operator/nn/mkldnn/mkldnn_base.cc:68: Allocate 56623104 bytes with malloc directly
#epoch 5, loss 1.1960, train acc 0.543, test acc 0.678, time 1162.3 sec
