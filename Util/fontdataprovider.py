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
        self.new_font = open("./Util/newfont").read().split()
        self.all_list = self.all_loader()
        self.new_list = self.new_list_loader()
        self.new_val_split()
        self.train_val_split()
        self.list_size = len(self.all_list)
        print(f"There are {self.list_size} data!")


    def font_single_loader(self,ch, font_path, cur_font, n):
        cur_filename = font_path+'/'+cur_font+'/'+str(n)+ch+'.npy'
        img = np.load(cur_filename)
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
            letter_id+=1

        return font_images[1:]
    
    def new_list_loader(self):
        return self.font_loader(self.new_font[0],0)
        
    def all_loader(self):
        all_list = np.array([[0,0,0]])
        font_id = 0
        
        for tp in self.fonttype_list:
            all_list=np.concatenate((all_list, self.font_loader(tp, font_id=font_id)), axis = 0)
            font_id +=1
        
        print(f"{len(self.fonttype_list)} fonts loaded!")
        return all_list[1:]
    
    def shuffle_train_data(self):
        np.random.shuffle(self.train_list)
    
    def save_data_provider(self):
        save_path = "./train/dataset"
        print("started saving font file")
        np.save(save_path+'/'+"train_list",self.train_list, allow_pickle=True)
        np.save(save_path+'/'+"all_list",self.all_list, allow_pickle=True)
        np.save(save_path+'/'+"val_list",self.val_list, allow_pickle=True)
        np.save(save_path+'/'+"source_list",self.source_list, allow_pickle=True)
        print(f"font saved! in {save_path}")
        
    def load_data_provider(self, load_path):
        self.train_list = np.load(load_path+'/train_list.npy', allow_pickle=True)
        self.val_list = np.load(load_path+'/val_list.npy', allow_pickle=True)
        self.all_list = np.load(load_path+'/all_list.npy', allow_pickle=True)
        self.source_list = np.load(load_path+'/source_list.npy', allow_pickle=True)
        self.list_size = len(self.all_list)
        print("font_loaded!")
        
    def new_val_split(self):
        new_train_list = np.array([[0,0,0]])
        new_val_list = np.array([[0,0,0]])
        for idx in range(len(self.new_list)):
            cur = self.new_list[idx:idx+1]
            #random between 0 and 1
            r = random.random()
            #random value bigger then val_ratio
            if(r>0.5):
                new_train_list = np.concatenate((new_train_list, cur), axis = 0)
            else:
                new_val_list = np.concatenate((new_val_list, cur), axis = 0)
        self.new_val_list = new_val_list[1:]
        self.new_train_list = new_train_list[1:]

    def train_val_split(self):
        val_list = np.array([[0,0,0]])
        train_list = np.array([[0,0,0]])
        source_list = np.array([[0,0,0]])
        for idx in range(len(self.all_list)):
            cur = self.all_list[idx:idx+1]
            #random between 0 and 1
            r = random.random()
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