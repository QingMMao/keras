from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np

#读取数据
DATA_DIR = '../data/fashion'

fashion_mnist = input_data.read_data_sets(DATA_DIR, one_hot=False, validation_size=0)
#读取训练图片
train_images = fashion_mnist.train.images  
#训练图片reshape为28*28
train_labels = train_images.reshape((60000,28,28))
#读取训练标签
train_labels = fashion_mnist.train.labels
#打乱数据
train_images,train_labels = shuffle(train_images,train_labels)

#读取训练图片
test_images = fashion_mnist.test.images  
#训练图片reshape为28*28
test_labels = test_images.reshape((10000,28,28))
#读取训练标签
test_labels = fashion_mnist.test.labels
#打乱数据
test_images,test_labels = shuffle(test_images,test_labels)




