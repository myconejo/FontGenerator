import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
import os
from fontdataprovider import FontDataProvider

def font_single_loader(ch, font_path, cur_font, n):
    cur_filename = font_path+'/'+cur_font+'/'+str(n)+ch+'.npy'
    print(cur_filename)
    img = np.load(cur_filename)
    #plt.figure(figsize=(1,2))
    #plt.imshow(img,cmap='gray')
    #plt.show()
    return img
    
def font_loader(cur_font):
    letter_file = open("./LetterType")
    font_path = "./GeneratedFontImage"
    
    letter_list = letter_file.read().split()
    font_images = []
    n = 0
    for ch in letter_list:
        letter_img = [cur_font,ch,font_single_loader(ch, font_path, cur_font, n)]
        font_images.append(letter_img)
        #print(letter_img)
        n+=1
    
    print(font_images)
        



batch_num = 64
font_data_provider = FontDataProvider(".")

def train_loader(batch_size, font_data_provider):
    #train_dataloader = np.array([[]])
    train_imgs = font_data_provider.train_list
    source_imgs = font_data_provider.source_list
    print(len(train_imgs), len(source_imgs))
    x = []
    y = []
    c_nums = []
    for idx in range(len(train_imgs)):
        letter_n = train_imgs[idx,1]
        category_n = train_imgs[idx,0]
        x.append(source_imgs[letter_n, 2])
        y.append(train_imgs[idx, 2])
        #np.concatenate([x, source_imgs[letter_n, 2]], axis=0)
        #np.concatenate([y, train_imgs[idx, 2]], axis=0)
        c_nums.append(category_n)

        #train_dataloader.append([source_imgs[letter_n, 2],train_imgs[idx, 2],category_n])
        if(len(x)>=batch_size):
            yield torch.tensor(x), torch.tensor(y), c_nums
            x = []
            y = []
            c_nums = []
    if(len(x) !=0):
        yield torch.tensor(x), torch.tensor(y), c_nums

def val_loader(batch_size, font_data_provider):
    #train_dataloader = np.array([[]])
    val_imgs = font_data_provider.val_list
    source_imgs = font_data_provider.source_list
    print(len(val_imgs), len(source_imgs))
    x = []
    y = []
    c_nums = []
    for idx in range(len(val_imgs)):
        letter_n = val_imgs[idx,1]
        category_n = val_imgs[idx,0]
        x.append(source_imgs[letter_n, 2])
        y.append(val_imgs[idx, 2])
        #np.concatenate([x, source_imgs[letter_n, 2]], axis=0)
        #np.concatenate([y, train_imgs[idx, 2]], axis=0)
        c_nums.append(category_n)

        #train_dataloader.append([source_imgs[letter_n, 2],train_imgs[idx, 2],category_n])
        if(len(x)>=batch_size):
            #yield np.array(x), np.array(y), c_nums
            yield torch.tensor(x), torch.tensor(y), c_nums
            x = []
            y = []
            c_nums = []
    if(len(x) !=0):
        yield torch.tensor(x), torch.tensor(y), c_nums

"""
for idx, batch in enumerate(train_loader(batch_num = batch_num,font_data_provider=font_data_provider)):
    x,y,c_nums = batch
    print(idx)
    print(torch.from_numpy(x).shape)
    
for idx, batch in enumerate(val_loader(batch_num = batch_num,font_data_provider=font_data_provider)):
    x,y,c_nums = batch
    print(idx)
    print(torch.from_numpy(x).shape)   
"""