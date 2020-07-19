import skimage.io as io
from skimage import exposure

def Bright(img):
    image = exposure.adjust_gamma(img,0.7)
    return image

str = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/test/cats' + '/*.jpg'
coll = io.ImageCollection(str)

for i in range(10):
    fname = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/noise2/cats/'+'cats.%d.jpg'%(i+1000)
    io.imsave(fname,Bright(coll[i]))

str = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/test/dogs' + '/*.jpg'
coll = io.ImageCollection(str)

for i in range(len(coll)):
    fname = 'E:/Dog_vs_Cats_data/cats_and_dogs_small/noise2/dogs/'+'dogs.%d.jpg'%(i+1000)
    io.imsave(fname,Bright(coll[i]))
    