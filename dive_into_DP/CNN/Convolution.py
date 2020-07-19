from mxnet import autograd,nd
from mxnet.gluon import nn

def corr2d(x,k):
    h,w = k.shape
    y = nd.zeros((x.shape[0]-h+1,x.shape[1]-w+1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = (x[i:i+h,j:j+w]*k).sum()
    return y

#和x=[[0,1,2],[3,4,5],[6,7,8]]有一个存储的功能
x = nd.array([[0,1,2],[3,4,5],[6,7,8]])
k = nd.array([[0,1],[2,3]])
#print(corr2d(x,k))

class Conv2D(nn.Block):
    def __init__(self,kernel_size,**kwargs):
        super(Conv2D,self).__init__(**kwargs)
        self.weight = self.params.get('weight',shape=kernel_size)
        self.bias = self.params.get('bias',shape=(1,))
        
    def forward(self,x):
        return corr2d(x,self.weight.data())+self.bias.data()
    
x = nd.ones((6,8))
x[:,2:6] = 0
k = nd.array([[1,-1]])

y = corr2d(x,k)
#转置矩阵：x.T

#conv2d = nn.Conv2D(1,kernel_size=(1,2))
#conv2d.initialize()
#
#x = x.reshape((1,1,6,8))
#y = y.reshape((1,1,6,7))
#
#for i in range(10):
#    with autograd.record():
#        y_hat = conv2d(x)
#        l = (y_hat-y)**2
#    l.backward()
#    #自己的理解：加上[:]是挨个回溯计算参数的
#    conv2d.weight.data()[:] -= 3e-2*conv2d.weight.grad()
#    if (i+1)%2 == 0:
#        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

def comp_conv2d(conv2d,x):
    conv2d.initialize()
    x = x.reshape((1,1)+x.shape)
    print(x.shape)
    y = conv2d(x)
    return y.reshape(y.shape[2:])

conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape