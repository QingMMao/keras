#GAN

#Generator
import keras
from keras import layers
import numpy as np

latend_dim = 32
height = 32
width = 32
channels = 3
generator_input = keras.Input(shape=(latend_dim,))

#Transform the input into a 16x16 128-channels feature map
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

#Add a  convolution layer
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

#Upsample to 32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

#Produce a 32x32 1-channel feature map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()



#Discriminator
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

#Adversarial
discriminator.trainable = False

gan_input = keras.Input(shape=(latend_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

#GAN training
import os
from keras.preprocessing import image

#Load data
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
#Choose frog
x_train = x_train[y_train.flatten() == 6]
#Normalize data
x_train = x_train.reshape(
        (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20
save_dir = 'Qing_dir'

start = 0

for step in range(iterations):
    #Sample random points in the latent sapce
    random_latent_vectors = np.random.normal(size=(batch_size, latend_dim))
    #Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)
    #Combine them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combine_images = np.concatenate([generated_images, real_images])
    #Assemble labels discriminating real from fake images
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    #Add random noise to the labels - important trik!!!!
    labels += 0.05 * np.random.random(labels.shape)
    #Train discriminator
    d_loss = discriminator.train_on_batch(combine_images, labels)
    #Sample random points inthe latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latend_dim))
    #Assemble labels that say "all real images"
    misleading_targets = np.zeros((batch_size, 1))

#Train generator via gan(discriminator weights are frozen)
a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
start += batch_size
if start > len(x_train) - batch_size:
    start = 0
if step % 100 == 0:
    #Save model weights
    gan.save_weights('gan.h5')
    #Print metrics
    print('Discriminatior loss:', d_loss)
    print('Adversarial loss:', a_loss)
    #Save one generated image
    img = image.array_to_img(generated_images[0] * 255., scale=False)
    img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
    #Save one real image for comparison
    img = image.array_to_img(real_images[0] * 255., scale=False)
    img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))



















