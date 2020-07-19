from tensorflow.examples.tutorials.mnist import input_data

#读取数据
DATA_DIR = '../data/fashion'

fashion_mnist = input_data.read_data_sets(DATA_DIR, one_hot=False, validation_size=0)

train_images = fashion_mnist.train.images
#train_images = train_images.reshape((60000,28,28))
train_labels = fashion_mnist.train.labels

test_images = fashion_mnist.test.images
#test_images = test_images.reshape((10000,28,28))
test_labels = fashion_mnist.test.labels

#from keras import models
#from keras import layers
#
#model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu',
#                        input_shape=(28*28,)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dense(512, activation='relu'))
#model.add(layers.Dense(10, activation='softmax'))
#
#from keras import optimizers
#
#model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
#              loss='binary_crossentropy',
#              metrics=['acc'])
#
##
#train_images = train_images.reshape((60000,28*28))
#train_images = train_images.astype('float32')/255
#
#test_images = test_images.reshape((10000,28*28))
#test_images = test_images.astype('float32')/255
#
#from keras.utils import to_categorical
#
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)
#
#model.fit(train_images, train_labels, epochs=5, batch_size=128)
#
#test_loss,test_acc = model.evaluate(test_images,test_labels)
#
#print('test_acc:',test_acc)







