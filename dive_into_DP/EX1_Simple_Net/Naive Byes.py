#from matplotlib import pyplot as plt
#from Ipython import display
##矢量图显示
#display.set_matplotlib_formats('svg')
import mxnet as mx
from mxnet import nd
import numpy as np

def transform(data,label):
    #floor返回下舍整数，astype实现类型转化
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)

#transform标准化数据
mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

xcount = nd.ones((784,10))
ycount = nd.ones((10))

for data,label in mnist_train:
    y = int(label)
    ycount[y] += 1
    xcount[:,y] += data.reshape((784))
    
#计算先验概率
py = ycount/ycount.sum()
#计算条件概率
px = (xcount/ycount.reshape(1,10))

##显示图像
##figsize设置图像的宽和高
#fig, figarr = plt.subplots(1,10, figsize = (10,10))
#for i in range(10):
#    figarr[i].imshow(xcount[:,i].reshape((28,28)).asnumpy(), cmap='hot')
#    figarr[i].axes.get_xaxis().set_visible(False)
#    figarr[i].axes.get_yaxis().set_visible(False)
#plt.show()

logpx = nd.log(px)
logpxneg = nd.log(1-px)
logpy = nd.log(py)

def bayespost(data):
    logpost = logpy.copy()
    #结合先验概率
    logpost += (logpx*data+logpxneg*(1-data)).sum(0)
    #避免上溢或者下溢
    logpost -= nd.max(logpost)
    post = nd.exp(logpost).asnumpy()
    post /= np.sum(post)
    return post

#fig, figarr = plt.subplots(2, 10, figsize=(10, 3))
## Show 10 images
#ctr = 0
#for data, label in mnist_test:
#    x = data.reshape((784,1))
#    y = int(label)
#    post = bayespost(x)
#    # Bar chart and image of digit
#    figarr[1, ctr].bar(range(10), post)
#    figarr[1, ctr].axes.get_yaxis().set_visible(False)
#    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
#    figarr[0, ctr].axes.get_xaxis().set_visible(False)
#    figarr[0, ctr].axes.get_yaxis().set_visible(False)
#    ctr += 1
#    if ctr == 10:
#        break
#plt.show()

ctr = 0
err = 0

for data,label in mnist_test:
    ctr += 1
    x = data.reshape((784,1))
    y = int(label)
    
    post = bayespost(x)
    if post[y] < post.max():
        err += 1
#在所有的test中，出错概率
print('Naive Bayes has an error rate of', err/ctr)
