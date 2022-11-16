import torch
import torch.nn as nn

###################################### HELPER FUNCTIONS ######################################
# conv2d: nn.Conv2d with Leaky ReLU and Batch Normalization
def conv2d(c_in, c_out, k_size=3, stride=2, pad=1, dilation=1, bn=True, lrelu=True, leak=0.2):
    layers = []
    if lrelu:
        layers.append(nn.LeakyReLU(leak))
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

# deconv2d: nn.Conv2d with Batch Normalization and Dropout
def deconv2d(c_in, c_out, k_size=3, stride=1, pad=1, dilation=1, bn=True, dropout=False, p=0.5):
    layers = []
    layers.append(nn.LeakyReLU(0.2))     # set leaky param as 0.2
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    if dropout:
        layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)

###################################### HELPER FUNCTIONS ######################################

# GENERATOR
class Generator(nn.Module):
    def __init__(self, input_dim=1, conv_dim=64):
        super(Generator, self).__init__()
        # Encoder and Decoder
        self.encoder_model = Encoder()
        self.decoder_model = Decoder()

    def forward(self, input, font_embed):
        encoder_result, encoder_dict = self.encoder_model(input)
        embedded = torch.cat((encoder_result, font_embed), dim=1)
        decoder_result = self.decoder_model(embedded, encoder_dict)
        return decoder_result
        
# ENCODER
class Encoder(nn.Module):
    # input_dim:    number of images
    # conv_dim:     
    def __init__(self, input_dim=1, conv_dim=64):
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_dim, conv_dim, kernel_size=5, stride=2, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=5, stride=2, padding=2, dilation=2)

        # Helper Function
        self.bn2 = nn.BatchNorm2d(conv_dim)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, input):
        encoder_dict = dict()
        output1 = self.conv1(input)
        encoder_dict['enc1']=output1
        output = self.bn2(self.conv2(self.lrelu(output1)))
        return output, encoder_dict
    
# DECODER
class Decoder(nn.Module):
    # input_dim:    number of images
    # conv_dim:     
    def __init__(self, input_dim=1, conv_dim=64):
        # Deconvolution Layers
        self.deconv1 = nn.ConvTranspose2d(conv_dim*10, conv_dim*8, kernel_size=3)
        self.deconv2 = nn.Conv2d(conv_dim*16, conv_dim*8, kernel_size=5, stride=2, padding=2, dilation=2)

        # Helper Methods
        self.bn2 = nn.BatchNorm2d(conv_dim*8)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, input, encoder_dict):
        output1 = self.deconv1(input)
        output2 = self.dropout(self.bn2(self.deconv2(self.lrelu(output1))))
        output2 = torch.cat((output2, encoder_dict['enc7']), dim=1)
        output = torch.tanh(output2)
        return output   

# DISCRIMINATOR
class Discriminator(nn.Module):
    # category_num: number of categories for the fonts
    # img_dim:      dimension of the image (generated image + groud-truth image)
    # disc_dim:     dimension of the discriminator (64 from the decoder)
    def __init__(self, category_num, img_dim=2, disc_dim=64):
        super(Discriminator, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(img_dim, disc_dim, kernel_size=3, stride=2, pad=1, dilation=1)
        self.conv2 = nn.Conv2d(disc_dim, 2*disc_dim, kernel_size=3, stride=2, pad=1, dilation=1)
        self.conv3 = nn.Conv2d(2*disc_dim, 4*disc_dim, kernel_size=3, stride=2, pad=1, dilation=1)
        self.conv4 = nn.Conv2d(4*disc_dim, 8*disc_dim, kernel_size=3, stride=2, pad=1, dilation=1)
        
        # Fully Connected (Linear) Layers
        self.fc1 = nn.Linear(512*disc_dim, 1)
        self.fc2 = nn.Linear(512*disc_dim, category_num)
        
        # Helper Methods
        self.lrelu = nn.LeakyReLU(0.2)
        self.bnorm = nn.BatchNorm2d()
        self.dropout = nn.Dropout()
        
    def forward(self, images):
        # image: generated or sample data images for the discriminator
        batch_size = images.shape[0]
        output = self.conv1(images)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        
        # resize the output to match the fc layer
        output = output.reshape(batch_size, -1)
        
        # compute the losses:
        ## tf_loss:     loss from the image
        ## cat_loss:    loss from the category of the fonts
        tf_loss_logit = self.fc1(output)
        tf_loss = torch.sigmoid(tf_loss_logit)
        cat_loss = self.fc2(output)
        
        return (tf_loss, tf_loss_logit, cat_loss)
        
        
