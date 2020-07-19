from mxnet import nd
from mxnet.gluon import nn

x = nd.arange(4)

#写入文件
nd.save('x-file',x)

#从文件中读出
x2 = nd.load('x-file')

y = nd.zeros(4)
nd.save('x-files',[x,y])

x2,y2 = nd.load('x-files')

#存储list
mydict = {'x':x,'y':x}
nd.save('mydict',mydict)
mydict2 = nd.load('mydict')

class MLP(nn.Block):
    def __init__(self,**kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Dense(256,activation='relu')
        self.output = nn.Dense(10)
    
    def forward(self, x):
        return self.output(self.hidden(x))
    
net = MLP()
net.initialize()
x = nd.random.uniform(shape=(2,20))
y = net(x)

net.save_parameters('mlp.params')



