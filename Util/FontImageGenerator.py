import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
import os


def single_font_image_saver(ch, size, save_path, font_path,n):
    adding=4
    font = ImageFont.truetype(font_path, size = size+adding)
    x, y = font.getsize(ch)
    font_size = max(x,y)


    img = Image.new('RGB', (size, size), (255, 255, 255)).convert('L')
    draw = ImageDraw.Draw(img)
    draw.text(((size-x)/2,(size-y)/2), ch, font=font)
    # draw.text((20,20),"A",font=font)

    img = np.array(img)
    tmp = np.where(img!=255)
    ws,we,hs,he = max(min(tmp[0]-5),0), min(max(tmp[0]+5),size), max(min(tmp[1])-5,0), min(max(tmp[1]+5),size)
    img = img[ws:we,hs:he]

    w_left = int((size-img.shape[0])/2)
    w_right = size-img.shape[0]-w_left
    h_left = int((size-img.shape[1])/2)
    h_right = size-img.shape[1]-h_left

    img = np.pad(img,((w_left,w_right),(h_left,h_right)), 'constant', constant_values=255)

    
    np.save(save_path+'/'+str(n)+ch,img, allow_pickle=True)

    # plt.figure(figsize=(1,2))
    # plt.imshow(img,cmap='gray')
    # plt.show()
    
def font_saver(size):
    font_file = open("./FontType")
    letter_file = open("./LetterType")
    font_list = font_file.read().split('\n')
    letter_list = letter_file.read().split()
    
    for cur_font in font_list:
        font_path = "./Font/" + cur_font+'.ttf'
        save_path = "./GeneratedFontImage/"+cur_font
        os.makedirs(save_path, exist_ok = True)
        n = 0
        for ch in letter_list:       
            single_font_image_saver(ch, size, save_path, font_path, n)
            n+=1
    
font_saver(100)