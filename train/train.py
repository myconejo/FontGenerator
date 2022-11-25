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

def train(cont_learn=False, epoch=400, category_num = 96):
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
    pre_epoch = 0
    
    model_path = os.path.join(cur_path,"results\\ckpt\\pre_train1")
    image_path = os.path.join(cur_path,"results\\fake-image\\pre_train1")
    
    
    print(f"Model Path:{model_path}, Image Path:{image_path}")
    
    if(cont_learn):
        Enc.load_state_dict(torch.load(os.path.join(model_path, f"Enctest.pt")))
        Dec.load_state_dict(torch.load(os.path.join(model_path, f"Dectest.pt")))
        Gen = BaseGenerator(Enc, Dec,category_num = category_num, learnembed = True).to(device) 
        Gen.load_state_dict(torch.load(os.path.join(model_path, f"Gentest.pt") ))
        Dis.load_state_dict(torch.load(os.path.join(model_path, f"Distest.pt") ))
        pre_epoch = 1000
    
    #summary(Gen,(1,128,128))
    summary(Enc, (1,128,128))
    summary(Dis,(2,128,128))

    input_fonts = []
    source_dataset = []

    # Log step
    log_step = 100
    if(cont_learn):
        learning_rate = 2e-4
    else:
        learning_rate = 1e-3
    milestones = [300]
    
    
    # Optimization for generator and discriminators
    G_optimizer = optim.Adam(Gen.parameters(), lr=learning_rate,betas=(0.5,0.999))
    #G_optimizer = optim.Adam(list(Enc.parameters())+list(Dec.parameters()), lr=learning_rate,betas=(0.5,0.999))
    D_optimizer = optim.Adam(Dis.parameters(), lr=learning_rate,betas=(0.5,0.999))
    #Scheduling optimizers with given milestones
    print(f"Current Learning Rate is {G_optimizer.param_groups[0]['lr']}")
    
    G_scheduler = optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=milestones, gamma=0.2)
    D_scheduler = optim.lr_scheduler.MultiStepLR(D_optimizer, milestones=milestones, gamma=0.2)
    # L1 Loss, Constant Loss
    l1_criterion = nn.L1Loss(size_average=True).to(device)
    const_criterion = nn.MSELoss(size_average=True).to(device)
    bce_criterion = nn.BCEWithLogitsLoss(size_average=True).to(device)
    
    l1_losses, const_losses, cat_losses, d_losses, g_losses = [],[],[],[],[]
    
    # Embedding
    embeddings = embedding_tensor_generate(category_num=category_num, embedding_dim=128)

    
    cur_iter = 0
    image_size = 128

    new_data = True
    font_data_provider = FontDataProvider(".")
    
    
    #train_dataloader = train_loader(batch_size=batch_size, font_data_provider=font_data_provider)
    if(new_data and not cont_learn):
        font_data_provider.save_data_provider()
    else:
        font_data_provider.load_data_provider("./train/dataset")
    print(f"font lists:{font_data_provider.fonttype_list}")
    
    for cur_epoch in (pbar:=tqdm(range(epoch))):
    #for cur_epoch in range(epoch):
        # Penalties for generator's losses
        Lconst_penalty = 15
        L1_penalty = 100
        Lcategory_penalty = 1
        if(cur_epoch+pre_epoch >500):
            Lconst_penalty=1000
            L1_penalty = 500

        # Iterable dataset
        #random.shuffle(font_data_provider.train_list)
        font_data_provider.shuffle_train_data()
        train_dataloader = train_loader(batch_size=batch_size, font_data_provider=font_data_provider)

        # Forward
        for id, batch in enumerate(train_dataloader):
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
            cat_loss = Lcategory_penalty * 0.5 * (real_cat_loss + fake_cat_loss)
            
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
            
            
            # Gradient Update
            # Discriminator Backward Process
            Dis.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optimizer.step()
            
            
            # Generator Backward Process
            
            Gen.zero_grad()
            #Enc.zero_grad()
            #Dec.zero_grad()
            G_loss.backward(retain_graph=True)
            G_optimizer.step()
            
            l1_losses.append(int(l1_loss.data))
            const_losses.append(int(const_loss.data)) 
            cat_losses.append(int(cat_loss.data))
            d_losses.append(int(D_loss.data))
            g_losses.append(int(G_loss.data))
            
            # Log
            
            #if (id + 1) % log_step == 0:
            if (id) == 0:
                log_string = f"Epoch[{cur_epoch+pre_epoch+1}/{epoch+pre_epoch}], step[{id}], l1_loss: {l1_loss.item()}, d_loss: {D_loss.item()}, g_loss: {G_loss.item()}"
                pbar.set_description(log_string)
                #print(log_string)

            # Save Images
            #if (id + 1) % 350 == 0:
            if (cur_epoch+1) % 5==0 and (id) == 0:
                id_save_path = os.path.join(image_path, "fake-img-%d-%d.png" % (cur_epoch+pre_epoch+1, id))
                id_save_path_t = os.path.join(image_path, "true-img-%d-%d.png" % (cur_epoch+pre_epoch+1, id))
                save_image(fake_image.data, id_save_path, nrow=8, pad_value=255)
                save_image(real_image.data, id_save_path_t, nrow=8, pad_value=255)
                #print(font_nums)
        if(not cont_learn):   
            #update scheduler
            G_scheduler.step()
            D_scheduler.step()
        if(cur_epoch in milestones):
            print(f"Current Learning Rate is {G_optimizer.param_groups[0]['lr']}")
        if (cur_epoch+1)%50==0:
            # Save Checkpoints
            now = datetime.datetime.now()
            now_date = now.strftime("%m-%d")
            now_time = now.strftime('%H:%M')
            #torch.save(Enc.state_dict(), os.path.join(model_path, f"Enc{epoch}-{now_date}-{now_time}-test.pt"))
            torch.save(Enc.state_dict(), os.path.join(model_path, f"Enctestmid.pt"))
            #torch.save(Dec.state_dict(), os.path.join(model_path, f"Dec{epoch}-{now_date}-{now_time}-test.pt"))
            torch.save(Dec.state_dict(), os.path.join(model_path, f"Dectestmid.pt"))
            #torch.save(Gen.state_dict(), os.path.join(model_path, f"Gen{epoch}-{now_date}-{now_time}-test.pt"))
            torch.save(Gen.state_dict(), os.path.join(model_path, f"Gentestmid.pt"))
            #torch.save(Dis.state_dict(), os.path.join(model_path, f"Dis{epoch}-{now_date}-{now_time}-test.pt"))
            torch.save(Dis.state_dict(), os.path.join(model_path, f"Distestmid.pt"))
            torch.save([l1_losses, const_losses, cat_losses, d_losses, g_losses], os.path.join(model_path, f"losses.pt"))
        
    # Save Checkpoints
    now = datetime.datetime.now()
    now_date = now.strftime("%m-%d")
    now_time = now.strftime('%H:%M')
    if(cont_learn):
        torch.save(Enc.state_dict(), os.path.join(model_path, f"Enccont.pt"))
        torch.save(Dec.state_dict(), os.path.join(model_path, f"Deccont.pt"))
        torch.save(Gen.state_dict(), os.path.join(model_path, f"Gencont.pt"))
        torch.save(Dis.state_dict(), os.path.join(model_path, f"Discont.pt"))
        torch.save([l1_losses, const_losses, cat_losses, d_losses, g_losses], os.path.join(model_path, f"lossescont.pt"))
    else:
        torch.save(Enc.state_dict(), os.path.join(model_path, f"Enctest.pt"))
        torch.save(Dec.state_dict(), os.path.join(model_path, f"Dectest.pt"))
        torch.save(Gen.state_dict(), os.path.join(model_path, f"Gentest.pt"))
        torch.save(Dis.state_dict(), os.path.join(model_path, f"Distest.pt"))
        torch.save([l1_losses, const_losses, cat_losses, d_losses, g_losses], os.path.join(model_path, f"losses.pt"))
            
train(cont_learn=True, epoch=1000, category_num = 96)