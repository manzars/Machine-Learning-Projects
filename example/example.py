#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:55:57 2019

@author: manzars
"""


import numpy as np
import cv2


file = open('example.csv', 'w')
header = ''
for i in range(784):
    if(i != 783):
        header += 'pixel ' + str(i) + ','  
    else:
        header += 'pixel ' + str(i)
        
file.write(header)
file.write('\n')

img = ['suneo.jpg', 'suneo2.jpg', 'suneo3.jpg']
no_of_image = len(img)

for j in range(no_of_image):
    image = cv2.imread(img[j], 0)
    res = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    x = res.ravel()
    count = 0
    for i in x:
        if(count != 783):
            file.write(str(i) + ', ')
        else:
            file.write(str(i))
        count += 1
    file.write('\n')
file.close()