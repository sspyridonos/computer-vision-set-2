# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:53:48 2018

@author: sspyr
"""
from PIL import Image
import numpy as np
import Sobel


def hough(img):
    im = np.array(Image.open(img))
    length = len(im)
    width = len(im[0])
    l = np.int((length/2)-10)#cuts the upper part of the image
    for i in range(0,l):
        for j in range(0,width):
            im[i,j] = 0
    Image.fromarray(im).save('Hough.png')
    im = Sobel.edge_finder(0.7,'Hough.png')
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    diag = np.int(np.sqrt(length * length + width * width))   # max dist
    rhos = np.linspace(-diag, diag, diag * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    numThetas = len(thetas)
       
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag, numThetas), dtype=np.uint64)
    yidys, xidxs= np.nonzero(im) 
    
    
      # Vote in the hough accumulator
    for i in range(len(xidxs)):
       x = xidxs[i]
       y = yidys[i]
        
       for t in range(numThetas):
         rho = round(x * cos_t[t] + y * sin_t[t])
         t=np.int(t)
         rho=np.int(rho)
         accumulator[rho,t] += 1
    
    
    idx = np.argmax(accumulator)
    sh = np.int(idx / accumulator.shape[1])
    rho = rhos[sh]
    theta = thetas[idx % accumulator.shape[1]]
    im = np.array(Image.open(img))
    im2 = np.array(Image.open(img))
    for i in range(l,length):
        for j in range(0,width):
            im2[i,j] = -theta*im2[i,j]+rho
            if(im2[i,j,0]<50 and im2[i,j,1]<25):
                print(i,j,im[i,j])
                im[i,j,0] = 0
                im[i,j,1] = 255
                im[i,j,2] = 0
    Image.fromarray(im).save('Hough.png')
    

hough('roadsidecop.png')