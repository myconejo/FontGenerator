import torch
import torch.nn as nn
from Model.Embedding import get_batch_embedding


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')
# Helper Functions
## conv2d: nn.Conv2d with Leaky ReLU and Batch Normalization
def conv2d(c_in, c_out, k_size=3, stride=2, pad=1, dilation=1, bn=True, lrelu=True, leak=0.2):
    layers = []
    if lrelu:
        layers.append(nn.LeakyReLU(leak))
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

## deconv2d: nn.Conv2d with Batch Normalization and Dropout
def deconv2d(c_in, c_out, k_size=3, stride=1, pad=1, dilation=1, bn=True, dropout=False, p=0.5):
    layers = []
    layers.append(nn.LeakyReLU(0.2))     # set leaky param as 0.2
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    if dropout:
        layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)


def Generator(images, En, De,font_nums, embed_layer, encode_layers=False, category_num = 50, embedding_dim=128):
    encoded_source, encode_layers = En(images)
    font_embed = embed_layer(torch.IntTensor(font_nums).to(device))
    font_embed = font_embed.reshape(len(font_nums), embedding_dim, 1, 1)
    embedded = torch.cat((encoded_source, font_embed), 1)
    fake_target = De(embedded, encode_layers)
    if encode_layers:
        return fake_target, encoded_source, encode_layers
    else:
        return fake_target, encoded_source

class CategoryLayer(nn.Module):
    def __init__(self, input_dim=1, category_dim=96):
        super(CategoryLayer, self).__init__()
        self.embedgenerate = nn.Linear(1,category_dim)
    def forward(self, input):
        output = self.embedgenerate(input)
        return output

# GENERATOR
class BaseGenerator(nn.Module):
    def __init__(self,En, De, input_dim=1, conv_dim=64, learnembed=False, category_num = 50, embedding_dim = 128):
        super(BaseGenerator, self).__init__()
        # Encoder and Decoder
        self.encoder_model = En
        self.category_num = category_num
        self.learnembed = learnembed
        self.embedding_dim = conv_dim*2
        self.embed_layer = nn.Embedding(category_num, self.embedding_dim)
        self.decoder_model = De

    def forward(self, input, font_nums, embedding):
        encoder_result, encoder_dict = self.encoder_model(input)
        if(self.learnembed):
            font_embed = self.embed_layer(torch.IntTensor(font_nums).to(device))
            font_embed = font_embed.reshape(len(font_nums), self.embedding_dim, 1, 1)
        else:
            font_embed = get_batch_embedding(len(font_nums), font_nums, embedding, self.embedding_dim).to(device)
        embedded=torch.concat((encoder_result, font_embed),dim=1)
        decoder_result = self.decoder_model(embedded, encoder_dict)
        return decoder_result, encoder_result
        
# ENCODER
class BaseEncoder(nn.Module):
    # input_dim:    number of images
    # conv_dim:     
    def __init__(self, input_dim=1, conv_dim=64):
        super(BaseEncoder, self).__init__()
        # Convolutional Layers
        self.conv1 = conv2d(input_dim, conv_dim, k_size=5, stride=2, pad=2, dilation=2, lrelu=False, bn=False)
        self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=2, pad=2, dilation=2)
        self.conv3 = conv2d(conv_dim*2, conv_dim*4, k_size=4, stride=2, pad=1, dilation=1)
        self.conv4 = conv2d(conv_dim*4, conv_dim*8)
        self.conv5 = conv2d(conv_dim*8, conv_dim*8)
        self.conv6 = conv2d(conv_dim*8, conv_dim*8)
        self.conv7 = conv2d(conv_dim*8, conv_dim*8)
        self.conv8 = conv2d(conv_dim*8, conv_dim*8)
        
    def forward(self, input):
        encode_dicts = dict()
        e1 = self.conv1(input)
        encode_dicts['e1'] = e1
        e2 = self.conv2(e1)
        encode_dicts['e2'] = e2
        e3 = self.conv3(e2)
        encode_dicts['e3'] = e3
        e4 = self.conv4(e3)
        encode_dicts['e4'] = e4
        e5 = self.conv5(e4)
        encode_dicts['e5'] = e5
        e6 = self.conv6(e5)
        encode_dicts['e6'] = e6
        e7 = self.conv7(e6)
        encode_dicts['e7'] = e7
        encoded_source = self.conv8(e7)
        encode_dicts['e8'] = encoded_source

        return encoded_source, encode_dicts

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=1, conv_dim=64):
        super(SimpleEncoder, self).__init__()
        # Convolutional Layers
        self.conv1 = conv2d(input_dim, conv_dim, k_size=5, stride=2, pad=2, dilation=2, lrelu=False, bn=False)
        self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=4, pad=2, dilation=2)
        self.conv3 = conv2d(conv_dim*2, conv_dim*4)
        self.conv4 = conv2d(conv_dim*4, conv_dim*8)
        self.conv5 = conv2d(conv_dim*8, conv_dim*8)
        self.conv6 = conv2d(conv_dim*8, conv_dim*8)
        self.conv7 = conv2d(conv_dim*8, conv_dim*8)
        
    def forward(self, input):
        encode_dicts = dict()
        e1 = self.conv1(input)
        encode_dicts['e1'] = e1
        e2 = self.conv2(e1)
        encode_dicts['e2'] = e2
        e3 = self.conv3(e2)
        encode_dicts['e3'] = e3
        e4 = self.conv4(e3)
        encode_dicts['e4'] = e4
        e5 = self.conv5(e4)
        encode_dicts['e5'] = e5
        e6 = self.conv6(e5)
        encode_dicts['e6'] = e6
        encoded_source = self.conv7(e6)
        encode_dicts['e7'] = encoded_source

        return encoded_source, encode_dicts
    
# DECODER
class BaseDecoder(nn.Module):

    def __init__(self, img_dim=1, embedded_dim=640, conv_dim=64):
        super(BaseDecoder, self).__init__()
        self.deconv1 = deconv2d(embedded_dim, conv_dim*8, dropout=True)
        self.deconv2 = deconv2d(conv_dim*16, conv_dim*8, dropout=True, k_size=4)
        self.deconv3 = deconv2d(conv_dim*16, conv_dim*8, k_size=5, dilation=2, dropout=True)
        self.deconv4 = deconv2d(conv_dim*16, conv_dim*8, k_size=4, dilation=2, stride=2)
        self.deconv5 = deconv2d(conv_dim*16, conv_dim*4, k_size=4, dilation=2, stride=2)
        self.deconv6 = deconv2d(conv_dim*8, conv_dim*2, k_size=4, dilation=2, stride=2)
        self.deconv7 = deconv2d(conv_dim*4, conv_dim*1, k_size=4, dilation=2, stride=2)
        self.deconv8 = deconv2d(conv_dim*2, img_dim, k_size=4, dilation=2, stride=2, bn=False)
    
    
    def forward(self, embedded, encode_dicts):
        d1 = self.deconv1(embedded)
        d1 = torch.cat((d1, encode_dicts['e7']), dim=1)
        d2 = self.deconv2(d1)
        d2 = torch.cat((d2, encode_dicts['e6']), dim=1)
        d3 = self.deconv3(d2)
        d3 = torch.cat((d3, encode_dicts['e5']), dim=1)
        d4 = self.deconv4(d3)
        d4 = torch.cat((d4, encode_dicts['e4']), dim=1)
        d5 = self.deconv5(d4)
        d5 = torch.cat((d5, encode_dicts['e3']), dim=1)
        d6 = self.deconv6(d5)
        d6 = torch.cat((d6, encode_dicts['e2']), dim=1)
        d7 = self.deconv7(d6)
        d7 = torch.cat((d7, encode_dicts['e1']), dim=1)
        d8 = self.deconv8(d7)        
        fake_target = torch.tanh(d8)
        
        return fake_target

# DECODER
class SimpleDecoder(nn.Module):
    def __init__(self, img_dim=1, embedded_dim=640, conv_dim=64):
        super(SimpleDecoder, self).__init__()
        self.deconv1 = deconv2d(embedded_dim, conv_dim*8, dropout=True)
        self.deconv2 = deconv2d(conv_dim*16, conv_dim*8, dropout=True, k_size=4)
        self.deconv3 = deconv2d(conv_dim*16, conv_dim*8, k_size=5, dilation=2, dropout=True)
        self.deconv4 = deconv2d(conv_dim*16, conv_dim*8, k_size=4, dilation=2, stride=2)
        self.deconv5 = deconv2d(conv_dim*16, conv_dim*4, k_size=4, dilation=2, stride=2)
        self.deconv6 = deconv2d(conv_dim*8, conv_dim*2, k_size=4, dilation=2, stride=2)
        self.deconv7 = deconv2d(conv_dim*4, conv_dim*1, k_size=4, dilation=2, stride=2)
        self.deconv8 = deconv2d(conv_dim*2, img_dim, k_size=4, dilation=2, stride=2, bn=False)
    
    def forward(self, embedded, encode_dicts):
        d1 = self.deconv1(embedded)
        d1 = torch.cat((d1, encode_dicts['e7']), dim=1)
        d2 = self.deconv2(d1)
        d2 = torch.cat((d2, encode_dicts['e6']), dim=1)
        d3 = self.deconv3(d2)
        d3 = torch.cat((d3, encode_dicts['e5']), dim=1)
        d4 = self.deconv4(d3)
        d4 = torch.cat((d4, encode_dicts['e4']), dim=1)
        d5 = self.deconv5(d4)
        d5 = torch.cat((d5, encode_dicts['e3']), dim=1)
        d6 = self.deconv6(d5)
        d6 = torch.cat((d6, encode_dicts['e2']), dim=1)
        d7 = self.deconv7(d6)
        d7 = torch.cat((d7, encode_dicts['e1']), dim=1)
        d8 = self.deconv8(d7)        
        fake_target = torch.tanh(d8)
        
        return fake_target

# DISCRIMINATOR
class Discriminator(nn.Module):
    # category_num: number of categories for the fonts
    # img_dim:      dimension of the image (generated image + groud-truth image)
    # disc_dim:     dimension of the discriminator (64 from the decoder)
    def __init__(self, category_num, img_dim=2, disc_dim=64):
        super(Discriminator, self).__init__()
        
        # Convolutional Layers
        self.conv1 = conv2d(img_dim, disc_dim, bn=False)
        self.conv2 = conv2d(disc_dim, disc_dim*2)
        self.conv3 = conv2d(disc_dim*2, disc_dim*4)
        self.conv4 = conv2d(disc_dim*4, disc_dim*8)
        
        # Fully Connected (Linear) Layers
        self.fc1 = nn.Linear(disc_dim*8*8*8, 1)
        self.fc2 = nn.Linear(disc_dim*8*8*8, category_num)
        
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
        tf_loss_logit = self.fc1(output.detach())
        tf_loss = torch.sigmoid(tf_loss_logit)
        cat_loss = self.fc2(output.detach())
        
        return (tf_loss, tf_loss_logit, cat_loss)
        
        
