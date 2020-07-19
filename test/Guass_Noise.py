import skimage.io as io
import numpy as np

def Guass_Noise(img):
    coutn = 100000
    for k in range(0,coutn):
        #get the random point
        xi = int(np.random.uniform(0,img.shape[1]))
        xj = int(np.random.uniform(0,img.shape[0]))
        #add noise
        if img.ndim == 2:
            img[xj,xi] = 255
        elif img.ndim == 3:
            img[xj,xi,0] = 25
            img[xj,xi,1] = 20
            img[xj,xi,2] = 20
    return img


str = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/test/cats' + '/*.jpg'
coll = io.ImageCollection(str)

for i in range(len(coll)):
    fname = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/noise1/cats/'+'cats.%d.jpg'%(i+1000)
    io.imsave(fname,Guass_Noise(coll[i]))
'''
str = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/test/dogs' + '/*.jpg'
coll = io.ImageCollection(str)

for i in range(len(coll)):
    fname = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/noise1/dogs/'+'dogs.%d.jpg'%(i+1000)
    io.imsave(fname,Guass_Noise(coll[i]))
 '''   