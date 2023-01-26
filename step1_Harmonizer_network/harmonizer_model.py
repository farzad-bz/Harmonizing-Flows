import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)


def double_conv(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
            ConcatELU(),
            nn.Conv2d(2*in_channels, out_channels, kernel_size, padding='same'),
            ConcatELU(),
            nn.Conv2d(2*out_channels, out_channels, kernel_size, padding='same'))

class Harmonizer(nn.Module):

    def __init__(self, c_in):
        super().__init__()
                
        self.dconv_down1 = nn.Conv2d(c_in, 16, 3, padding='same')
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 48)
        self.dconv_down4 = double_conv(48, 64)
        self.dconv_down5 = double_conv(64, 64)     

        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(14)
        self.alpha_last = nn.Conv2d(64, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up4 = double_conv(64 + 64, 64)
        self.dconv_up3 = double_conv(48 + 64, 48)
        self.dconv_up2 = double_conv(32 + 48, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)
        
        self.bias_last = nn.Conv2d(16, c_in, 1)
        
    def forward(self, input_x):
        conv1 = self.dconv_down1(input_x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)  
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)  
        
        x = self.dconv_down5(x)
        alpha_ =  self.avgpool(x)
        alpha = self.alpha_last(alpha_)
        x = self.upsample(x)     
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)  
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        bias = self.bias_last(x)
        return alpha*input_x + bias