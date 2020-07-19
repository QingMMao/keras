import random
import skimage.io as io
import numpy as np
from skimage import exposure

def Salt_And_Pepper_Noise(img):
    rows, cols, dims= img.shape  
    for i in range(5000):  
        x = np.random.randint(0, rows)  
        y = np.random.randint(0, cols)  
        img[x, y, :] = 255
    return img

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


def Bright(img):
    image = exposure.adjust_gamma(img,0.6)
    return image

def Somber(img):
    image = exposure.adjust_gamma(img,4.0)
    return image

#Cats with noise
str_cat = 'E:/test' + '/*.jpg'
coll_cat = io.ImageCollection(str_cat)

for i in range(len(coll_cat)):
    a = random.randint(0, 3)
    if a == 0:
        fname = 'E:/KITTI/Noise_Images/'+'%s.jpg'%str(i).zfill(6)
        io.imsave(fname,Salt_And_Pepper_Noise(coll_cat[i]))
    if a == 1:
        fname = 'E:/KITTI/Noise_Images/'+'%s.jpg'%str(i).zfill(6)
        io.imsave(fname,Somber(coll_cat[i]))
    if a == 2:
        fname = 'E:/KITTI/Noise_Images/'+'%s.jpg'%str(i).zfill(6)
        io.imsave(fname,Bright(coll_cat[i]))
    if a == 3:
        fname = 'E:/KITTI/Noise_Images/'+'%s.jpg'%str(i).zfill(6)
        io.imsave(fname,Guass_Noise(coll_cat[i]))


        
        
        
        
        
        
        
        
        
        
        