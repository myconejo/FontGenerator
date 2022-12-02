import numpy as np
import os, datetime, time, glob
import sys
cur_path = os.getcwd()
sys.path.insert(0,cur_path)

from Model.Backbone import BaseGenerator, Discriminator, BaseEncoder, SimpleDecoder, SimpleEncoder, BaseDecoder
from Util.fontdataprovider import FontDataProvider
from Model.Embedding import embedding_tensor_generate
from Util.FontLoader import train_loader, val_loader

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchsummary import summary
import random


import numpy as np
import os, datetime, time, glob


batch_size = 64
fine_tune = False

def valid(epoch,category_num = 96):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Define the models
    
    Enc = BaseEncoder().to(device) 
    #embed_layer = nn.Embedding(category_num, 128)
    Dec = BaseDecoder().to(device)
    Gen = BaseGenerator(Enc, Dec,category_num = category_num, learnembed = True).to(device) 
    Dis = Discriminator(category_num=category_num).to(device) 

    
    model_path = os.path.join(cur_path,"results\\ckpt\\pre_train1")
    image_path = os.path.join(cur_path,"results\\fake-image\\pre_train1")    
    
    print(f"Model Path:{model_path}, Image Path:{image_path}")    

    Enc.load_state_dict(torch.load(os.path.join(model_path, f"Enccont.pt")))
    Dec.load_state_dict(torch.load(os.path.join(model_path, f"Deccont.pt")))
    Gen = BaseGenerator(Enc, Dec,category_num = category_num, learnembed = True).to(device) 
    Gen.load_state_dict(torch.load(os.path.join(model_path, f"Gencont.pt") ))
    Dis.load_state_dict(torch.load(os.path.join(model_path, f"Discont.pt") ))
    
    # Embedding
    embeddings = embedding_tensor_generate(category_num=category_num, embedding_dim=128)

    l1_criterion = nn.L1Loss(size_average=True).to(device)
    const_criterion = nn.MSELoss(size_average=True).to(device)
    bce_criterion = nn.BCEWithLogitsLoss(size_average=True).to(device)
    
    l1_losses, const_losses, cat_losses, d_losses, g_losses = [],[],[],[],[]
    
    font_data_provider = FontDataProvider(".")
    
    font_data_provider.load_data_provider("./train/dataset")
    print(f"font lists:{font_data_provider.fonttype_list}")
    
    valid_dataloader = val_loader(batch_size=batch_size, font_data_provider=font_data_provider)
    cur_iter = 0
    image_size = 128
    Lconst_penalty = 500
    L1_penalty = 1000
    
    
    # Forward
    for id, batch in enumerate(valid_dataloader):
        cur_iter+=1
        ## x: input image
        ## real_image: real image 
        
        x, real_image, font_nums = batch
        x = (2*x/255-1).detach().reshape(-1,1,image_size,image_size).to(device)
        real_image = (2*real_image/255-1).reshape(-1,1,image_size,image_size).to(device)
        # Init Dis params to zeros
            
        # Generate fake image from the generator
        fake_image, encoded_source = Gen(x, font_nums=font_nums, embedding=embeddings)
            
        #fake_image, encoded_source,_ = Generator(x,Enc,Dec,font_nums=font_nums,embed_layer=embed_layer, category_num=category_num)
            
                        
        # Concatenate the input and the generated fake image, and get the discrimimator losses
        fake_patch = torch.cat([x, fake_image], dim=1)
        f_loss, f_loss_logit, fcat_logit = Dis(fake_patch)

        # Concatenate the input and the real image, and get the discrimimator losses
        real_patch = torch.cat([x, real_image], dim=1)
        r_loss, r_loss_logit, rcat_logit = Dis(real_patch)
            
        # Const Loss for generator backward process
        encoded_fake = Enc(fake_image)[0]
        const_loss = Lconst_penalty * const_criterion(encoded_source, encoded_fake)

            
        # real_loss = nn.BCELoss(real_result, 1)
            
        # Category Loss using the BCE Losss
        real_category = F.one_hot(torch.tensor(font_nums), category_num).to(device).to(torch.float)
        #print(real_category.shape,len(font_nums) ,rcat_logit.shape)
        real_cat_loss = bce_criterion(rcat_logit, real_category)
        fake_cat_loss = bce_criterion(fcat_logit, real_category)
        cat_loss = 0.5 * (real_cat_loss + fake_cat_loss)
            
        # Labels
        one_labels = torch.ones([x.shape[0],1]).to(device)
        zero_labels = torch.zeros([x.shape[0],1]).to(device)
            
        # Binary Loss (T/F Loss from the Discriminator)
        real_bin_loss = bce_criterion(r_loss_logit, one_labels)
        fake_bin_loss = bce_criterion(f_loss_logit, zero_labels)
        bin_loss = fake_bin_loss + real_bin_loss
            
        # L1 Loss of Fake and Real Madrid
        l1_loss = L1_penalty * l1_criterion(real_image, fake_image)
        # Cheat Loss for fool model's discriminator
        cheat_loss = bce_criterion(f_loss_logit, one_labels)
        
            
            
        G_loss = fake_cat_loss + const_loss + l1_loss + cheat_loss
        D_loss = cat_loss + bin_loss
            
        
        l1_losses.append(int(l1_loss.data))
        const_losses.append(int(const_loss.data)) 
        cat_losses.append(int(cat_loss.data))
        d_losses.append(int(D_loss.data))
        g_losses.append(int(G_loss.data))
        
        # Log
        
        #if (id + 1) % log_step == 0:
        
        #log_string = f"step[{id}], l1_loss: {l1_loss.item()}, d_loss: {D_loss.item()}, g_loss: {G_loss.item()}"
        #print(log_string)

        # Save Images
        #if (id + 1) % 350 == 0:
        if(id) %5 == 0:
            id_save_path = os.path.join(image_path, "valid-%d.png" % (id))
            id_save_path_t = os.path.join(image_path, "valid-true-%d.png" % (id))
            save_image(fake_image.data, id_save_path, nrow=8, pad_value=255)
            save_image(real_image.data, id_save_path_t, nrow=8, pad_value=255)
    print(f"iter:{cur_iter}, avgl1loss:{sum(l1_losses)/cur_iter}, avgconst_losses:{sum(const_losses)/cur_iter}, \
        cat_losses:{sum(d_losses)/cur_iter}, g_losses:{sum(g_losses)/cur_iter}")    
#valid(category_num=96,)