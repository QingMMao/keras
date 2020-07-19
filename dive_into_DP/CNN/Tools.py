import mxnet as mx
from mxnet import nd
from mxnet.gluon import data as gdata,loss as gloss
from mxnet import autograd,nd
import sys
import os
import time

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
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start = 0.0,0.0,0,time.time()
        for X,y in train_iter:
            X,y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat,y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1)==y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter,net,ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def corr2d(x,k):
    h,w = k.shape
    y = nd.zeros((x.shape[0]-h+1,x.shape[1]-w+1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = (x[i:i+h,j:j+w]*k).sum()
    return y

def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    """Download the fashion mnist dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]#ToTensor从uint8格式变为32-bit格式，并将最后一维放到第一维
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4#读数据的时候无需提速/需要提速

    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers=num_workers)
    
    
    return train_iter, test_iter

