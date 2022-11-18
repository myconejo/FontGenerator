from Model.Backbone import Generator, Discriminator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

batch_size = 64

def dataset():

    return []

def train():
    if torch.cuda.is_available():
        device = torch.cuda.device(0)
    else:
        device = torch.device('cpu')

    epoch = 20
    
    Gen = Generator().to(device)
    Dis = Discriminator().to(device)

    input_fonts = []
    source_dataset = []

    G_optimizer = optim.Adam(G.parameters(), lr=2e-4,betas=(0.5,0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=2e-4,betas=(0.5,0.999))
    
    for i in range(epoch):
        train_dataset = []
        Lconst_penalty = 15
        L1_penalty = 100
        Lcategory_penalty = 1
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #forward process
        for id, batch in enumerate(train_dataloader):
            x,y = batch

            Dis.zero_grad()
            fake_image = Gen(x)
            fake_patch = torch.cat(x, y)
            fake_result = Dis(fake_patch)
            fake_loss = nn.BCELoss(fake_result, 0)

            real_patch = torch.cat(x,fake_image)
            real_result = Dis(real_patch)
            real_loss = nn.BCELoss(real_result, 1)

            D_loss = fake_loss + real_loss
            D_loss.backward()
            D_optimizer.step()

           
            #loss
            const_loss = Lconst_penalty * nn.MSELoss(real_result, fake_result)
            l1_loss = L1_penalty * nn.
            #gradient update


        