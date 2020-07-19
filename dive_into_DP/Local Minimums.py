
import math
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import nn,data as gdata,loss as gloss
import d2l
from mpl_toolkits import mplot3d
import numpy as np
import time

#first
def f(x):
    return x*np.cos(np.pi*x)

#d2l.set_figsize((4.5,2.5))
#x = np.arange(-1.0,2.0,0.1)
#fig, = d2l.plt.plot(x,f(x))
#fig.axes.annotate('local minimum',xy=(-0.3,-0.25),xytext=(-0.77,-1.0),
#                  arrowprops=dict(arrowstyle='->'))
#fig.axes.annotate('global minimum',xy=(1.1,-0.95),xytext=(0.6,0.8),
#                  arrowprops=dict(arrowstyle='->'))
#d2l.plt.xlabel('x')
#d2l.plt.ylabel('f(x)')
    
#x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]
#z = x ** 2 - y ** 2
#ax = d2l.plt.figure().add_subplot(111, projection='3d')
#ax.plot_wireframe(x, y, z, ** {'rstride': 2, 'cstride': 2})
#ax.plot([0], [0], [0], 'rx')
#ticks = [-1, 0, 1]
#d2l.plt.xticks(ticks)
#d2l.plt.yticks(ticks)
#ax.set_zticks(ticks)
#d2l.plt.xlabel('x')
#d2l.plt.ylabel('y')

#second 
#def gd(eta):
#    x = 10
#    results = [x]
#    for i in range(10):
#        x = -eta*2*x
#        results.append(x)
#    print('epoch 10,x:',x)
#    return results
#
#res = gd(1.2)
#
#def show_trace(res):
#    n = max(abs(min(res)), abs(max(res)), 10)
#    f_line = np.arange(-n, n, 0.1)
#    d2l.set_figsize()
#    d2l.plt.plot(f_line, [x * x for x in f_line])
#    d2l.plt.plot(res, [x * x for x in res], '-o')
#    d2l.plt.xlabel('x')
#    d2l.plt.ylabel('f(x)')
#show_trace(res)

#third
#def train_2d(trainer):
#    x1,x2,s1,s2 = -5,-2,0,0
#    results = [(x1,x2)]
#    for i in range(20):
#        x1,x2,s1,s2 = trainer(x1,x2,s1,s2)
#        results.append((x1,x2))
#    print('epoch %d,x1 %f, x2 %f' % (i + 1, x1, x2))
#    return results
#
#def show_trace_2d(f, results):
#    d2l.plt.plot( * zip( * results), '-o', color='#ff7f0e')
#    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
#    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
#    d2l.plt.xlabel('x1')
#    d2l.plt.ylabel('x2')
#    
#eta = 0.1
#
#def f_2d(x1, x2): # Objective function
#    return x1 ** 2 + 2 * x2 ** 2
#def gd_2d(x1, x2, s1, s2):
#    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)
#
#show_trace_2d(f_2d,train_2d(gd_2d))
#
##forth:SGD
#def sgd_2d(x1, x2, s1, s2):
#    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
#            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)
#
#show_trace_2d(f_2d,train_2d(sgd_2d))

#sixth:Mini-batch SGD
def get_data_ch7():
    data = np.genfromtxt('./airfoil_self_noise.dat',delimiter='\t')
    data = (data-data.mean(axis=0))/data.std(axis=0)
    return nd.array(data[:1500,:-1]),nd.array(data[:1500,-1])

features, labels = get_data_ch7()

def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
        
def train_ch7(trainer_fn,states,hyperparams,features,labels,
              batch_size=10,num_epochs=2):
    net,loss = d2l.linreg,d2l.squared_loss
    w = nd.random.normal(scale=0.01,shape=(features.shape[1],1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()
    
    def eva_loss():
        return loss(net(features,w,b),labels).mean().asscalar()
    
    ls = [eva_loss()]
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features,labels),batch_size,shuffle=True)
    
    for i in range(num_epochs):
        start = time.time()
        for batch_i,(X,y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X,w,b),y).mean()
            l.backward()
            trainer_fn([w,b],states,hyperparams)
            if(batch_i+1)*batch_size%100 == 0:
                ls.append(eva_loss())
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)


def train_gluon_ch7(trainer_name,trainer_hyperparams,features,labels,
                    batch_size=10,num_epochs=2):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    
    def eval_loss():
        return loss(net(features),labels).mean().asscalar()
    
    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
            gdata.ArrayDataset(features,labels),batch_size,shuffle=True)
    trainer = gluon.Trainer(
            net.collect_params(),trainer_name,trainer_hyperparams)
    for i in range(num_epochs):
        start = time.time()
        for batch_i,(X,y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X),y)
            l.backward()
            trainer.step(batch_size)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss:',ls[-1],',',time.time() - start,'sec per epoch')
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')

train_gluon_ch7('sgd', {'learning_rate': 0.05}, features, labels, 10)
    
    
    
    
    
    
    
    
    
    
    
    












