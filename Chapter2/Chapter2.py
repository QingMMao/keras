#从keras自带的数据集中导入mnist
from keras.datasets import mnist
#画图工具
import matplotlib.pyplot as plt

#数据格式已经做好了，下载分到训练集和测试集
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

#keras的模型和层
from keras import models
from keras import layers

#序贯模型是函数式模型的简略版，为最简单的线性结构，从头到尾，不分叉。
network = models.Sequential()
#增加全连接层，神经元个数为512，激活函数为relu，输入的大小为28*28
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
#增加全连接层，分为10类，输出神经元为10个，激活函数为softmax
network.add(layers.Dense(10,activation='softmax'))
#训练模式，优化器为rmsprop，损失函数为categorical_crossentropy，metrics是列表，包含评估模型在训练和测试时的网络性能的指标
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss,test_acc = network.evaluate(test_images,test_labels)

print('test_acc:',test_acc)

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
