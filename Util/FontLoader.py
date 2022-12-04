import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0,os.getcwd())
from Util.fontdataprovider import FontDataProvider

def font_single_loader(ch, font_path, cur_font, n, verbose = False):
    cur_filename = font_path+'/'+cur_font+'/'+str(n)+ch+'.npy'
    img = np.load(cur_filename)
    if(verbose):
        print(cur_filename)
        plt.figure(figsize=(1,2))
        plt.imshow(img,cmap='gray')
        plt.show()
    return img
    
def font_loader(cur_font,verbose = False):
    letter_file = open("Util/LetterType")
    font_path = "Util/GeneratedFontImage"
    letter_list = letter_file.read().split()
    font_images = []
    n = 0
    for ch in letter_list:
        letter_img = [cur_font,ch,font_single_loader(ch, font_path, cur_font, n, verbose = verbose)]
        font_images.append(letter_img)
        print(cur_font, ch)
        n+=1
        
def new_train_loader(batch_size, font_data_provider):
    train_imgs = font_data_provider.new_train_list
    source_imgs = font_data_provider.source_list
    x = []
    y = []
    c_nums = []
    for idx in range(len(train_imgs)):
        letter_n = train_imgs[idx,1]
        category_n = train_imgs[idx,0]
        x.append(source_imgs[letter_n, 2])
        y.append(train_imgs[idx, 2])
        c_nums.append(category_n)

        if(len(x)>=batch_size):
            yield torch.FloatTensor(x), torch.FloatTensor(y), c_nums
            x = []
            y = []
            c_nums = []
    if(len(x) !=0):
        yield torch.FloatTensor(x), torch.FloatTensor(y), c_nums
        
def new_val_loader(batch_size, font_data_provider):
    val_imgs = font_data_provider.new_val_list
    source_imgs = font_data_provider.source_list

    x = []
    y = []
    c_nums = []
    for idx in range(len(val_imgs)):
        letter_n = val_imgs[idx,1]
        category_n = val_imgs[idx,0]
        x.append(source_imgs[letter_n, 2])
        y.append(val_imgs[idx, 2])
        c_nums.append(category_n)

        if(len(x)>=batch_size):
            yield torch.FloatTensor(x), torch.FloatTensor(y), c_nums

            x = []
            y = []
            c_nums = []
    if(len(x) !=0):
        yield torch.FloatTensor(x), torch.FloatTensor(y), c_nums

def train_loader(batch_size, font_data_provider):
    train_imgs = font_data_provider.train_list
    source_imgs = font_data_provider.source_list
    x = []
    y = []
    c_nums = []
    for idx in range(len(train_imgs)):
        letter_n = train_imgs[idx,1]
        category_n = train_imgs[idx,0]
        x.append(source_imgs[letter_n, 2])
        y.append(train_imgs[idx, 2])
        c_nums.append(category_n)

        if(len(x)>=batch_size):
            yield torch.FloatTensor(x), torch.FloatTensor(y), c_nums
            x = []
            y = []
            c_nums = []
    if(len(x) !=0):
        yield torch.FloatTensor(x), torch.FloatTensor(y), c_nums

def val_loader(batch_size, font_data_provider):
    val_imgs = font_data_provider.val_list
    source_imgs = font_data_provider.source_list

    x = []
    y = []
    c_nums = []
    for idx in range(len(val_imgs)):
        letter_n = val_imgs[idx,1]
        category_n = val_imgs[idx,0]
        x.append(source_imgs[letter_n, 2])
        y.append(val_imgs[idx, 2])
        c_nums.append(category_n)

        if(len(x)>=batch_size):
            yield torch.FloatTensor(x), torch.FloatTensor(y), c_nums
            x = []
            y = []
            c_nums = []
    if(len(x) !=0):
        yield torch.FloatTensor(x), torch.FloatTensor(y), c_nums


def afont_loader(verbose = False):
    font_list = open("./Util/FontType").read().split()
    font_path = "Util/GeneratedFontImage"
    n = 0
    ch = 'a'
    for tp in font_list:
        print(tp)
        font_single_loader(ch, font_path, tp, n, verbose = verbose)
