import Tools
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import nn

def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not autograd.is_training():
        X_hat = (X-moving_mean)/nd.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape) == 2:
            mean = X.mean(axis=0)
            var = ((X-mean)**2).mean(axis=0)
        else:
            mean = X.mean(axis=(0,2,3),keepdims=True)
            var = ((X-mean)**2).mean(axis=(0,2,3),keepdims=True)
        X_hat = (X-mean)/nd.sqrt(var+eps)
        moving_mean = momentum*moving_mean+(1.0-momentum)*mean
        moving_var = momentum*moving_var+(1.0-momentum)*var
    Y = gamma*X_hat+beta
    return Y,moving_mean,moving_var

class BatchNorm(nn.Block):
    def __init__(self,num_features,num_dims,**kwargs):
        super(BatchNorm,self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1,num_features)
        else:
            shape = (1,num_features,1,1)
        
        self.gamma = self.params.get('gamma',shape=shape,init=init.One())
        self.beta = self.params.get('beta',shape=shape,init=init.Zero())
        
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)
        
    def forward(self,X):
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        
        Y,self.moving_mean,self.moving_var = batch_norm(X,self.gamma.data(),
                                                        self.beta.data(),
                                                        self.moving_mean,
                                                        self.moving_var,
                                                        eps=1e-5,momentum=0.9)
        return Y
    
    


net = nn.Sequential()
#Stack of LeNet
net.add(nn.Conv2D(channels=6,kernel_size=5,padding=2,activation='sigmoid'),
        BatchNorm(6,4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        BatchNorm(16,4),
        nn.Activation('sigmoid'),
        
        
        nn.MaxPool2D(pool_size=2,strides=2),
        
        nn.Dense(120,activation='sigmoid'),
        BatchNorm(120,2),
        nn.Activation('sigmoid'),
        
        nn.Dense(84,activation='sigmoid'),
        BatchNorm(84,2),
        nn.Activation('sigmoid'),
        
        nn.Dense(10))

lr, num_epochs, batch_size, ctx = 1.0, 5, 256, Tools.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = Tools.load_data_fashion_mnist(batch_size)
Tools.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
num_epochs)


