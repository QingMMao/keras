from mxnet import autograd,gluon,nd
from mxnet.gluon import data as gdata,loss as gloss,nn
import Tools
from matplotlib import pyplot as plt

maxdgree = 20 #多项式最多能有多少项
n_train,n_test = 100,1000 #size of
true_w = nd.zeros(maxdgree)
true_w[:4] = nd.array([5,1.2,-3.4,5.6])

features = nd.random.normal(shape=(n_train+n_test,1)) #size是（1100,1）
features = nd.random.shuffle(features)
poly_features = nd.power(features,nd.arange(maxdgree).reshape((1,-1)))#reshape成一行，任意列
#arange(10)是从0-9
poly_features = poly_features/(
        nd.gamma(nd.arange(maxdgree)+1).reshape((1,-1)))
labels = nd.dot(poly_features,true_w)
labels += nd.random.normal(scale=0.1,shape=labels.shape)

#使用y的对数刻度来画图
def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,
             legend=None,figsize=(3.5,2.5)):
    Tools.set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals,y2_vals,linestyle=':')
        plt.legend(legend)

num_epochs, loss = 200, gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1,use_bias=False))
    net.initialize()
    batch_size = min(10,train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
            train_features,train_labels),batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(),'sgd',
                            {'learning_rate':0.01})
    train_loss,test_loss = [],[]
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X),y)
            l.backward()
            trainer.step(batch_size)
        train_loss.append(loss(net(train_features),
                          train_labels).mean().asscalar())
        test_loss.append(loss(net(test_features),
                         test_labels).mean().asscalar())
    print('train loss:',train_loss,'test loss:',test_loss)
    semilogy(range(1,num_epochs+1),train_loss,'epochs','loss',
             range(1,num_epochs+1),test_loss,['train','test'])
    print('weight',net[0].weight.data().asnumpy())

fit_and_plot(poly_features[:n_train, 0:4], poly_features[n_train:, 0:4],
labels[:n_train], labels[n_train:])
        