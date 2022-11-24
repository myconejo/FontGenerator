#from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import time
import os, random, sys
sys.path.insert(0,os.getcwd())


class FontDataProvider():
    def __init__(self,root_dir, val_ratio=0.2):
        self.root_dir = root_dir
        self.val_ratio = val_ratio
        self.fonttype_list = open("./Util/FontType").read().split()
        self.letter_list = open("./Util/LetterType").read().split()
        self.all_list = self.all_loader()
        self.train_val_split()
        #self.source_list = self.all_list[0,:]


    def font_single_loader(self,ch, font_path, cur_font, n):
        cur_filename = font_path+'/'+cur_font+'/'+str(n)+ch+'.npy'
        img = np.load(cur_filename)
        #plt.figure(figsize=(1,2))
        #plt.imshow(img,cmap='gray')
        #plt.show()
        return img
    
    def font_loader(self,cur_font, font_id):
        letter_file = open("./Util/LetterType")
        font_path = "./Util/GeneratedFontImage"
        #print(os.path.basename)
        
        letter_list = letter_file.read().split()
        font_images = np.array([[0,0,0]])
        letter_id = 0
        for ch in letter_list:
            letter_img = np.array([[font_id,letter_id,self.font_single_loader(ch, font_path, cur_font, letter_id).tolist()]],dtype=object)
            font_images = np.concatenate((font_images,letter_img), axis = 0)
            #print(font_id, letter_id)
            #plt.figure(figsize=(1,2))
            #plt.imshow(letter_img[0][2],cmap='gray')
            #plt.show()
            letter_id+=1

        return font_images[1:]

    def all_loader(self):
        all_list = np.array([[0,0,0]])
        font_id = 0
        print(f"{len(self.fonttype_list)} fonts loaded!")
        for tp in self.fonttype_list:
            all_list=np.concatenate((all_list, self.font_loader(tp, font_id=font_id)), axis = 0)
            font_id +=1
        return all_list[1:]

    def train_val_split(self):
        val_list = np.array([[0,0,0]])
        train_list = np.array([[0,0,0]])
        source_list = np.array([[0,0,0]])
        for idx in range(len(self.all_list)):
            cur = self.all_list[idx:idx+1]
            #random between 0 and 1
            r = random.random()
            #print(cur[0][0])
            if(cur[0][0] == 0):
                source_list = np.concatenate((source_list, cur), axis = 0)
            #random value bigger then val_ratio
            if(r>self.val_ratio):
                train_list = np.concatenate((train_list, cur), axis = 0)
            else:
                val_list = np.concatenate((val_list, cur), axis = 0)
        self.val_list = val_list[1:]
        np.random.shuffle(self.val_list)
        self.train_list = train_list[1:]
        np.random.shuffle(self.train_list)
        self.source_list = source_list[1:]