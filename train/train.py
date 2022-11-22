import numpy as np
import os, datetime, time, glob
print(os.getcwd())
import sys
print(sys.path)

from Model.Backbone import BaseGenerator, Discriminator, BaseEncoder, SimpleDecoder, SimpleEncoder, BaseDecoder
from Util.fontdataprovider import FontDataProvider
from Util.Embedding import embedding_tensor_generate
from Util.FontLoader import train_loader, val_loader

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np
import os, datetime, time, glob


batch_size = 64

def dataset():

    return []

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    epoch = 20

    # Define the models
    Gen = BaseGenerator() 
    Enc = SimpleEncoder() 
    Dec = BaseDecoder() 
    Dis = Discriminator(category_num=50) 

    input_fonts = []
    source_dataset = []

    # Log step
    log_step = 100

    # Optimization for generator and discriminators
    G_optimizer = optim.Adam(Gen.parameters(), lr=2e-4,betas=(0.5,0.999))
    D_optimizer = optim.Adam(Dis.parameters(), lr=2e-4,betas=(0.5,0.999))

    # L1 Loss, Constant Loss
    l1_criterion = nn.L1Loss(size_average=True)
    const_criterion = nn.MSELoss(size_average=True)
    bce_criterion = nn.BCELoss(size_average=True)
    
    category_num = 2
    # Embedding
    embeddings = embedding_tensor_generate(category_num=category_num, embedding_dim=128)
    
    model_path = "/ckpt/"
    image_path = "/fake-image/"
    cur_iter = 0

    font_data_provider = FontDataProvider(".")
    train_dataloader = train_loader(batch_size=batch_size, font_data_provider=font_data_provider)
    for cur_epoch in tqdm(range(epoch)):

        # Penalties for generator's losses
        Lconst_penalty = 15
        L1_penalty = 100
        Lcategory_penalty = 1

        # Iterable dataset
        
        
        # Forward
        for id, batch in enumerate(train_dataloader):
            cur_iter+=1
            ## x: input image
            ## real_image: real image 
            x, real_image, font_nums = batch

            # Init Dis params to zeros
            Dis.zero_grad()

            # Generate fake image from the generator
            fake_image, encoded_source = Gen.forward(x, font_nums=font_nums, embedding=embeddings)
            
            # Concatenate the input and the generated fake image, and get the discrimimator losses
            fake_patch = torch.cat([x, fake_image], dim=1)
            f_loss, f_loss_logit, fcat_logit = Dis(fake_patch)

            # Concatenate the input and the real image, and get the discrimimator losses
            real_patch = torch.cat([x, real_image], dim=1)
            r_loss, r_loss_logit, rcat_logit = Dis(real_patch)
            
            # real_loss = nn.BCELoss(real_result, 1)
            
            # Category Loss using the BCE Loss
            real_category = "Something"

            real_cat_loss = bce_criterion(rcat_logit, real_category)
            fake_cat_loss = bce_criterion(fcat_logit, real_category)
            cat_loss = Lcategory_penalty * 0.5 * (real_cat_loss + fake_cat_loss)
            
            # Labels
            one_labels = torch.ones([batch_size,1]).to(device)
            zero_labels = torch.zeros([batch_size,1]).to(device)
            
            # Binary Loss (T/F Loss from the Discriminator)
            real_bin_loss = bce_criterion(r_loss_logit, one_labels)
            fake_bin_loss = bce_criterion(f_loss_logit, zero_labels)
            bin_loss = fake_bin_loss + real_bin_loss
            
            # L1 Loss of Fake and Real Madrid
            l1_loss = L1_penalty * l1_criterion(real_image, fake_image)

            # Const Loss for generator backward process
            encoded_fake = Enc(fake_image)[0]
            const_loss = Lconst_penalty * const_criterion(encoded_source, encoded_fake)
            
            # Gradient Update
            # Discriminator Backward Process
            D_loss = cat_loss + bin_loss
            Dis.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            
            # Generator Backward Process
            G_loss = fake_bin_loss + const_loss + l1_loss
            Gen.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            
            # Log
            if (id + 1) % log_step == 0:
                log_string = f"Epoch[{cur_epoch}/{epoch}], step[{id}/{batch_size}], \
                    l1_loss: {l1_loss.item()}, d_loss: {D_loss.item()}, g_loss: {G_loss.item()}"
                print(log_string)

            # Save Images
            if (id + 1) % 350 == 0:
                id_save_path = os.path.join(image_path, "fake-img-%d-%d.png" % (epoch, id+1))
                save_image(fake_image.data, id_save_path, nrow=8, pad_value=255)

        # Save Checkpoints
        now = datetime.datetime.now()
        now_date = now.strftime("%m%d")
        now_time = now.strftime('%H:%M')
        torch.save(Gen.state_dict(), os.path.join(model_path, f"{epoch}/{now_date}/{now_time}Gen.ckpt"))
        torch.save(Dis.state_dict(), os.path.join(model_path, f"{epoch}/{now_date}/{now_time}Dis.ckpt"))
            
train()