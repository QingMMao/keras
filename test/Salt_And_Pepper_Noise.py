import skimage.io as io
import numpy as np


str = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/test/cats' + '/*.jpg'
coll = io.ImageCollection(str)

#加噪声
def Salt_And_Pepper_Noise(img):
    rows, cols, dims= img.shape  
    for i in range(5000):  
        x = np.random.randint(0, rows)  
        y = np.random.randint(0, cols)  
        img[x, y, :] = 255
    return img

for i in range(len(coll)):
    fname = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/noise/cats/'+'cats.%d.jpg'%i
    io.imsave(fname,Salt_And_Pepper_Noise(coll[i]))
    
    
str = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/test/dogs' + '/*.jpg'
coll = io.ImageCollection(str)

for i in range(len(coll)):
    fname = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/noise/dogs/'+'dogs.%d.jpg'%i
    io.imsave(fname,Salt_And_Pepper_Noise(coll[i]))


