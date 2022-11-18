from Model.Backbone import Generator, Discriminator, Encoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os, datetime

batch_size = 64

def dataset():

    return []

def train():
    if torch.cuda.is_available():
        device = torch.cuda.device(0)
    else:
        device = torch.device('cpu')

    epoch = 20
    Enc = Encoder()
    Gen = Generator().to(device)
    Dis = Discriminator().to(device)

    input_fonts = []
    source_dataset = []
    
    log_step = 100

    G_optimizer = optim.Adam(Gen.parameters(), lr=2e-4,betas=(0.5,0.999))
    D_optimizer = optim.Adam(Dis.parameters(), lr=2e-4,betas=(0.5,0.999))
    l1_criterion = nn.L1Loss(size_average=True)
    const_criterion = nn.MSELoss(size_average=True)
    
    model_path = "/ckpt/"
    cur_iter = 0
    
    for cur_epoch in range(epoch):
        train_dataset = []
        Lconst_penalty = 15
        L1_penalty = 100
        Lcategory_penalty = 1
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #forward process
        for id, batch in enumerate(train_dataloader):
            cur_iter+=1
            x,real_image = batch

            Dis.zero_grad()
            fake_image, encoded_source = Gen(x, embeddings)
            fake_patch = torch.cat([x, real_image])
            f_loss, f_loss_logit, fcat_logit = Dis(fake_patch)
            

            real_patch = torch.cat([x,fake_image])
            t_loss, t_loss_logit, tcat_logit = Dis(real_patch)
            
            #real_loss = nn.BCELoss(real_result, 1)
            
            #category loss
            fake_cat_loss = nn.BCELoss(fcat_logit, real_category)
            true_cat_loss = nn.BCELoss(tcat_logit, real_category)
            cat_loss = Lcategory_penalty* 0.5*(true_cat_loss+fake_cat_loss)
            
            one_labels = torch.ones([batch_size,1]).to(device)
            zero_labels = torch.zeros([batch_size,1]).to(device)
            
            #binary loss
            fake_bin_loss = nn.BCELoss(fcat_logit, zero_labels)
            true_bin_loss = nn.BCELoss(tcat_logit, one_labels)
            bin_loss = fake_bin_loss + true_bin_loss
            
            #l1_loss, const_loss for generator backward process
            l1_loss = L1_penalty * l1_criterion(real_image, fake_image)
            encoded_fake = Enc(fake_image)[0]
            const_loss = Lconst_penalty * const_criterion(encoded_source, encoded_fake)
            
            #gradient update
            #Discriminator Backward process
            D_loss = cat_loss + bin_loss
            Dis.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            
            #Generator Backward process
            G_loss = fake_bin_loss + const_loss + l1_loss
            Gen.zero_grad()
            G_loss.bnackward()
            G_optimizer.step()
            
            #log
            if id % log_step == 0:
                log_string = f"Epoch[{cur_epoch}/{epoch}], step[{id}/{batch_size}], \
                    l1_loss: {l1_loss.item()}, d_loss: {D_loss.item()}, g_loss: {G_loss.item()}"
                print(log_string)
        #save ckpt
        now = datetime.datetime.now()
        now_date = now.strftime("%m%d")
        now_time = now.strftime('%H:%M')
        torch.save(Gen.state_dict(), os.path.join(model_path, f"{epoch}/{now_date}/{now_time}Gen.ckpt"))
        torch.save(Dis.state_dict(), os.path.join(model_path, f"{epoch}/{now_date}/{now_time}Dis.ckpt"))
            
