import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim=1, conv_dim=64):
        super(Generator, self).__init__()
        self.encoder_model = Encoder()
        self.decoder_mode = Decoder()
    def forward(self, input, font_embed):
        encoder_result, encoder_dict = self.encoder_model(input)
        embedded = torch.cat((encoder_result, font_embed), dim=1)
        decoder_result = self.decoder_model(embedded, encoder_dict)
        return decoder_result
        
        

class Encoder(nn.Module):
    def __init__(self, input_dim=1, conv_dim=64):
        self.conv1 = nn.Conv2d(input_dim, conv_dim, kernel_size=5, stride=2, padding=2,dilation=2)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=5, stride=2, padding=2,dilation=2)
        self.bn2 = nn.BatchNorm2d(conv_dim)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, input):
        encoder_dict = dict()
        output1 = self.conv1(input)
        encoder_dict['enc1']=output1
        output = self.bn2(self.conv2(self.lrelu(output1)))
        return output, encoder_dict
    
class Decoder(nn.Module):
    def __init__(self, input_dim=1, conv_dim=64):
        self.deconv1 = nn.ConvTranspose2d(conv_dim*10, conv_dim*8, kernel_size=3)
        self.deconv2 = nn.Conv2d(conv_dim*16, conv_dim*8, kernel_size=5, stride=2, padding=2,dilation=2)
        self.bn2 = nn.BatchNorm2d(conv_dim*8)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, input, encoder_dict):
        output1 = self.deconv1(input)
        
        output2 = self.dropout(self.bn2(self.deconv2(self.lrelu(output1))))
        output2 = torch.cat((output2, encoder_dict['enc7']), dim=1)
        
        output = torch.tanh(output2)
        return output       