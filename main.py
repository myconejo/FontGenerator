import numpy as np
import os, datetime, time, glob
import sys
cur_path = os.getcwd()
sys.path.insert(0,cur_path)

import torch
from Model.Backbone import BaseGenerator, Discriminator, BaseEncoder, SimpleDecoder, SimpleEncoder, BaseDecoder
from Util.fontdataprovider import FontDataProvider
from Model.Embedding import embedding_tensor_generate
from Util.FontLoader import train_loader, val_loader


model_path = os.path.join(cur_path,"results\\ckpt\\")
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


category_num=25

Enc = BaseEncoder().to(device) 
#embed_layer = nn.Embedding(category_num, 128)
Dec = BaseDecoder().to(device)
Enc.load_state_dict(torch.load(os.path.join(model_path, f"Enctest.pt")))
Dec.load_state_dict(torch.load(os.path.join(model_path, f"Dectest.pt")))
    
Gen = BaseGenerator(Enc, Dec,category_num = category_num, learnembed = True).to(device) 
Gen.load_state_dict(torch.load(os.path.join(model_path, f"Gentest.pt")))
Dis = Discriminator(category_num=category_num).to(device)
Dis.load_state_dict(torch.load(os.path.join(model_path, f"Dixtest.pt")))

