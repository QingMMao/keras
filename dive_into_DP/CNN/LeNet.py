import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time
import Tools

net = nn.Sequential()
#Stack of LeNet
net.add(nn.Conv2D(channels=6,kernel_size=5,padding=2,activation='sigmoid'),
        nn.AvgPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        nn.AvgPool2D(pool_size=2,strides=2),
        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        nn.Dense(10))

#Better 
#net.add(nn.Conv2D(channels=6,kernel_size=5,padding=2,activation='relu'),
#        nn.MaxPool2D(pool_size=2,strides=2),
#        nn.Conv2D(channels=16,kernel_size=5,activation=='relu'),
#        nn.MaxPool2D(pool_size=2,strides=2),
#        nn.Dense(120,activation='relu'),
#        nn.Dense(84,activation=='relu'),
#        nn.Dense(10))

X = nd.random.uniform(shape=(1,1,28,28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name,' output shape:\t',X.shape)


batch_size = 256
train_iter,test_iter = Tools.load_data_fashion_mnist(batch_size=batch_size)

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def evaluate_accuracy(data_iter,net,ctx):
    acc_sum,n = nd.array([0],ctx=ctx),0
    for X,y in data_iter:
        X,y = X.as_in_context(ctx),y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar()/n

def train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,
              num_epochs):
    loss = gloss.SoftmaxCrossEntropyLoss()
    
    i = 0
    
    for epoch in range(num_epochs):
        
        
        print('i=',i)
        i += 1
        
        train_l_sum,train_acc_sum,n,start = 0.0,0.0,0,time.time()
        
        
        for X,y in train_iter:
            
            j = 0
            print('j=',j)
            j += 1
            
            X,y = X.as_in_context(ctx), y.as_in_context(ctx)
            
            print('j=',j)
            j += 1
            
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
                
            print('j=',j)
            j += 1
            
            l.backward()
            
            print('j=',j)
            j += 1
            
            trainer.step(batch_size)
            
            print('j=',j)
            j += 1
            
            y = y.astype('float32')
            
            print('j=',j)
            j += 1
            
            train_l_sum += l.asscalar()
            
            print('j=',j)
            j += 1 
            
            train_acc_sum += (y_hat.argmax(axis=1)==y).sum().asscalar()
            
            print('j=',j)
            j += 1 
            
            n += y.size
            
            print('j=',j)
            j += 1
            
        test_acc = evaluate_accuracy(test_iter,net,ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))
        
lr, num_epochs,ctx = 0.9, 5, Tools.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

    
