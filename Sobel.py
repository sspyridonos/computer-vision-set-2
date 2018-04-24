# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:53:18 2018

@author: sspyr
"""

import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt


def edge_finder(T,img):
    im = np.array(Image.open(img))

    ymask = [[1,2,1],[0,0,0],[-1,-2,-1]]

    xmask = [[-1,0,1],[-2, 0,2],[-1, 0, -1]]
    gray = im.sum(-1)

    gy = signal.convolve2d(gray, ymask,
                              mode='same', boundary='fill', fillvalue=0)

    gx = signal.convolve2d(gray, xmask,
                              mode='same', boundary='fill', fillvalue=0)

    
    G = np.sqrt (gx*gx+gy*gy)
    G2 = np.copy(G)
    G = np.uint8(G)
    

    length = len(G)
    width = len(G[0])
    T = np.max(G)*T
    for i in range(0,length):
        for j in range(0,width):
            if(G[i,j]>T):
                G[i,j] = 255
            else:
                G[i,j] = 0
                
    G = np.float64(G)
    G *= 255.0 / np.max(G2)
    G = np.uint8(G)
    plt.imshow(G)
    plt.show()
    Image.fromarray(G).save('123Sobel.jpg')
    return G



edge_finder(0.68,'123.jpg')