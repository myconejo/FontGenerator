import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
import os


def single_font_image_saver(ch, size, save_path, font_path,n):
    adding=4
    font = ImageFont.truetype(font_path, size = 90)
    x, y = font.getsize(ch)
    font_size = max(x,y)


    img = Image.new('RGB', (size, size), (0, 0, 0)).convert('L')
    draw = ImageDraw.Draw(img)
    draw.text(((size-x)/2,(size-y)/2), ch,255, font=font)
    #draw.text((10,10),ch,255,font=font)
    
    
    
    tmp = img.getbbox()
    if(tmp !=None):
        hs,he,ws,we = max(tmp[0]-5,0), min(tmp[2]+5,size), max(tmp[1]-5,0), min(tmp[3]+5,size)
    else:
        hs,he,ws,we=0,size, 0,size

    img = np.array(img)
    img = img[ws:we,hs:he]
    img = 255 -img
    """
    img = np.array(img)
    tmp = np.where(img!=255)
    print(min(tmp[0]), max(tmp[0]))
    print(min(tmp[1]), max(tmp[1]))
    ws,we,hs,he = max(min(tmp[0]-5),0), min(max(tmp[0]+5),size), max(min(tmp[1])-5,0), min(max(tmp[1]+5),size)
    img = img[ws:we,hs:he]
    """
    
    w_left = int((size-img.shape[0])/2)
    w_right = size-img.shape[0]-w_left
    h_left = int((size-img.shape[1])/2)
    h_right = size-img.shape[1]-h_left

    img = np.pad(img,((w_left,w_right),(h_left,h_right)), 'constant', constant_values=255)
    #print(font_path)
    
    np.save(save_path+'/'+str(n)+ch,img, allow_pickle=True)

    #plt.figure(figsize=(1,2))
    #plt.imshow(img,cmap='gray')
    #plt.show()
    
def font_saver(size):
    font_file = open("./Util/FontType")
    new_file = open("./Util/newfont")
    letter_file = open("./Util/LetterType")
    new_list = new_file.read().split('\n')
    font_list = font_file.read().split('\n')
    letter_list = letter_file.read().split()
    
    for cur_font in new_list+font_list:
        font_path = "./Util/Font/" + cur_font+'.ttf'
        save_path = "./Util/GeneratedFontImage/"+cur_font
        os.makedirs(save_path, exist_ok = True)
        n = 0
        print(cur_font)
        for ch in letter_list:       
            single_font_image_saver(ch, size, save_path, font_path, n)
            n+=1
    
font_saver(128)