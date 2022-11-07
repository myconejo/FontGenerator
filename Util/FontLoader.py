import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
import os

def font_single_loader(ch, font_path, cur_font, n):
    cur_filename = font_path+'/'+cur_font+'/'+str(n)+ch+'.npy'
    img = np.load(cur_filename)
    #plt.figure(figsize=(1,2))
    #plt.imshow(img,cmap='gray')
    #plt.show()
    return img
    
def font_loader(cur_font):
    letter_file = open("./LetterType")
    font_path = "./GeneratedFontImage"
    
    letter_list = letter_file.read().split()
    font_images = np.array([])
    n = 0
    for ch in letter_list:
        np.append(font_images,np.array(font_single_loader(ch, font_path, cur_font, n)))
        n+=1
    
    print(font_images)
        
cur_font = "arial"
font_loader(cur_font)