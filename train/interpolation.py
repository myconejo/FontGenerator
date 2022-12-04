####
#Experiment for interpolation 
#
#####
import numpy as np
import os
import sys
cur_path = os.getcwd()
sys.path.insert(0,cur_path)

from Model.Backbone import BaseGenerator, Discriminator, BaseEncoder, SimpleDecoder, SimpleEncoder, BaseDecoder
from Util.fontdataprovider import FontDataProvider
from Model.Embedding import embedding_tensor_generate, get_batch_embedding


from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

batch_size = 64
fine_tune = False


def interpolation(category_num=96):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Define the models
    Enc = BaseEncoder().to(device) 
    Dec = BaseDecoder().to(device)
    Gen = BaseGenerator(Enc, Dec,category_num = category_num, learnembed = False).to(device) 
    Dis = Discriminator(category_num=category_num).to(device) 
    
    model_path = os.path.join(cur_path,"results\\ckpt\\pre_train4")
    image_path = os.path.join(cur_path,"results\\fake-image\\pre_train4")
    
    
    print(f"Model Path:{model_path}, Image Path:{image_path}")
    
    if(device == torch.device('cpu')):
        Enc.load_state_dict(torch.load(os.path.join(model_path, f"Enctest.pt"), map_location=torch.device('cpu')))
        Dec.load_state_dict(torch.load(os.path.join(model_path, f"Dectest.pt"), map_location=torch.device('cpu')))
        Gen.load_state_dict(torch.load(os.path.join(model_path, f"Gentest.pt"), map_location=torch.device('cpu') ))
        Dis.load_state_dict(torch.load(os.path.join(model_path, f"Distest.pt"), map_location=torch.device('cpu') ))
        embedding = torch.load(os.path.join(model_path, "embeddings.pt"), map_location=torch.device('cpu'))
    else:
        Enc.load_state_dict(torch.load(os.path.join(model_path, f"Enctest.pt")))
        Dec.load_state_dict(torch.load(os.path.join(model_path, f"Dectest.pt")))
        Gen.load_state_dict(torch.load(os.path.join(model_path, f"Gentest.pt") ))
        Dis.load_state_dict(torch.load(os.path.join(model_path, f"Distest.pt") ))
        embedding = torch.load(os.path.join(model_path, "embeddings.pt"))
        
    
    image_size = 128
    font_data_provider = FontDataProvider(".")
    source_list = font_data_provider.source_list.T[2][:8]
    x=[]
    for i in range(len(source_list)):
        x.append(source_list[i])
    x=torch.FloatTensor(x)
    print(f"shape:{x.shape}")
    for i in range(5):
        ratio = i/4.
        x = (2*x/255-1).detach().reshape(-1,1,image_size,image_size).to(device)
        
        #Generate fake image from the generator
        encoded_source, encoder_dict = Gen.encoder_model(x)
        font_embed1 = get_batch_embedding(8, [0]*8, embedding, Gen.embedding_dim).to(device)
        font_embed2 = get_batch_embedding(8, [1]*8, embedding, Gen.embedding_dim).to(device)
        font_embed = font_embed1*ratio + font_embed2*(1-ratio)
        font_embed = font_embed.reshape(-1, Gen.embedding_dim, 1, 1)
        print(encoded_source.shape, font_embed.shape)
        embedded=torch.concat((encoded_source, font_embed),dim=1)
        fake_image=Gen.decoder_model(embedded, encoder_dict)
        val_image_path = os.path.join(cur_path,"results\\interpolation")
        save_path = os.path.join(val_image_path, "ratio-%d-%d.png" % (i,4))
        save_image(fake_image.data,save_path,pad_value=255)
         
interpolation(category_num=79)