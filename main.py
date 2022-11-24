import numpy as np
import os, datetime, time, glob
import sys
import torch
from tqdm import tqdm
cur_path = os.getcwd()
sys.path.insert(0,cur_path)

import torch
import torch.nn as nn
from Model.Backbone import BaseGenerator, Discriminator, BaseEncoder, SimpleDecoder, SimpleEncoder, BaseDecoder
from Util.fontdataprovider import FontDataProvider
from Model.Embedding import embedding_tensor_generate
from Util.FontLoader import train_loader, val_loader
from torchvision.utils import save_image

model_path = os.path.join(cur_path,"results/ckpt")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


category_num=25

Enc = BaseEncoder().to(device) 
#embed_layer = nn.Embedding(category_num, 128)
Dec = BaseDecoder().to(device)
Enc.load_state_dict(torch.load(os.path.join(model_path, f"Enctest.pt"), map_location=torch.device('cpu') ))
Dec.load_state_dict(torch.load(os.path.join(model_path, f"Dectest.pt"), map_location=torch.device('cpu') ))
    
Gen = BaseGenerator(Enc, Dec,category_num = category_num, learnembed = True).to(device) 
Gen.load_state_dict(torch.load(os.path.join(model_path, f"Gentest.pt"), map_location=torch.device('cpu') ))
Dis = Discriminator(category_num=category_num).to(device)
Dis.load_state_dict(torch.load(os.path.join(model_path, f"Distest.pt"), map_location=torch.device('cpu') ))

batch_size = 64
image_size = 128
font_data_provider= FontDataProvider('.')
# Embedding
embeddings = embedding_tensor_generate(category_num=category_num, embedding_dim=128)
image_path = os.path.join(cur_path,"results/fake-image")
l1_criterion = nn.L1Loss(size_average=True).to(device)

valid_dataloader = train_loader(batch_size=batch_size, font_data_provider=font_data_provider)
# Forward
cur_iter = 0
for id, batch in tqdm(enumerate(valid_dataloader)):
    cur_iter+=1
    ## x: input image
    ## real_image: real image 

    x, real_image, font_nums = batch
    x = (x/255).detach().reshape(-1,1,image_size,image_size).to(device)
    real_image = (real_image/255).reshape(-1,1,image_size,image_size).to(device)
    # Init Dis params to zeros
        
    # Generate fake image from the generator
    fake_image, _ = Gen(x, font_nums=font_nums, embedding=embeddings)

    l1_loss =l1_criterion(real_image, fake_image)

    print(f"l1_loss:{l1_loss}")
    # Save Images
    id_save_path = os.path.join(image_path, "valid/fake-img-%d-%d.png" % (500, cur_iter))
    id_save_path_t = os.path.join(image_path, "valid/true-img-%d-%d.png" % (500, cur_iter))
    save_image(fake_image.data, id_save_path, nrow=8, pad_value=255)
    save_image(real_image.data, id_save_path_t, nrow=8, pad_value=255)

    
