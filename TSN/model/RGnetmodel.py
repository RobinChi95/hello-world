import torch
import torch.nn as nn

##############################################################################################################################################
##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class conv_block1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class block_cv2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(block_cv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
############################################################################################################################################

def double_conv_0(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),
    )
    
def double_conv_1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),

        
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),
    )


def double_conv_2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=2,stride=1, bias=False, dilation=2),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),

        nn.Conv2d(out_channels, out_channels, 3, padding=2,stride=1, bias=False, dilation=2),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),
    )

def double_conv_3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=3,stride=1, bias=False, dilation=3),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),

        nn.Conv2d(out_channels, out_channels, 3, padding=3,stride=1, bias=False, dilation=3),
        nn.ReLU(inplace=False),
        nn.BatchNorm2d(out_channels),
    )

def decon(in_channels, out_channels):
    return nn.Sequential(
       nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1),
       nn.ReLU(inplace=False),
       nn.BatchNorm2d(out_channels),
            )

##############################
#        Generator
##############################

class UNetPP(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv_0(1, 32)
        self.dconv_down2 = double_conv_2(32, 64)
        self.dconv_down3 = double_conv_2(64, 128)
        self.dconv_down4 = double_conv_2(128, 256)
        self.dconv_down5_new = double_conv_3(256, 512)

        self.dconv_down6 = double_conv_2(64, 32)
        self.dconv_down7 = double_conv_2(128, 64)
        self.dconv_down8 = double_conv_2(96, 32)
        self.dconv_down9 = double_conv_2(256, 128)
        self.dconv_down10 = double_conv_2(192, 64)
        self.dconv_down11 = double_conv_2(128, 32)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)      
        self.upsample_new1 = decon(512,256)
        self.upsample_new2 = decon(256,128)
        self.upsample_new3 = decon(128,64)
        self.upsample_new4 = decon(64,32)

        self.dconv_up4 = double_conv_1(512, 256)
        self.dconv_up3 = double_conv_1(384, 128)
        self.dconv_up2 = double_conv_1(256, 64)
        self.dconv_up1 = double_conv_1(160, 32)
        
        self.conv_last = nn.Conv2d(32, 1, 1)
        
        
    def forward(self, x0):
        conv1 = self.dconv_down1(x0)         #128*128*32
        x = self.maxpool(conv1)             #64*64*32

        conv2 = self.dconv_down2(x)         #64*64*64
        x = self.maxpool(conv2)             #32*32*64    

        up1_2 = self.upsample_new4(conv2)   #128*128*32
        conv1_2 = torch.cat([up1_2,conv1],dim=1)  #128*128*64
        conv1_2 = self.dconv_down6(conv1_2) #128*128*32
        
        conv3 = self.dconv_down3(x)         #32*32*128
        x = self.maxpool(conv3)             #16*16*128

        up2_2 = self.upsample_new3(conv3)   #64*64*64
        conv2_2 = torch.cat([up2_2,conv2],dim=1)  #64*64*128
        conv2_2 = self.dconv_down7(conv2_2) #64*64*64

        up1_3 = self.upsample_new4(conv2_2) #128*128*32
        conv1_3 = torch.cat([up1_3,conv1,conv1_2],dim=1)  #128*128*96
        conv1_3 = self.dconv_down8(conv1_3) #128*128*32
        
        conv4 = self.dconv_down4(x)         #16*16*256
        x = self.maxpool(conv4)             #8*8*256
        
        #UNet Plus
        up3_2 = self.upsample_new2(conv4)  #32*32*128
        conv3_2 = torch.cat([up3_2,conv3],dim=1) #32*32*256     
        conv3_2 = self.dconv_down9(conv3_2)#32*32*128
        
        #UNet Plus
        up2_3 = self.upsample_new3(conv3_2)#64*64*64
        conv2_3 = torch.cat([up2_3, conv2, conv2_2],dim=1)#64*64*192
        conv2_3 = self.dconv_down10(conv2_3)#64*64*64
        
        #UNet Plus
        up1_4 = self.upsample_new4(conv2_3)#128*128*32
        conv1_4 = torch.cat([up1_4, conv1, conv1_2, conv1_3],dim=1)#128*128*128
        conv1_4 = self.dconv_down11(conv1_4)#128*128*32

        x = self.dconv_down5_new(x)         #SY #8*8*512      

        x = self.upsample_new1(x)           #16*16*256
        x = torch.cat([x, conv4], dim=1)    #16*16*512
        
        x = self.dconv_up4(x)               #16*16*256
        x = self.upsample_new2(x)           #32*32*128
        x = torch.cat([x, conv3, conv3_2], dim=1)    #32*32*384
        
        x = self.dconv_up3(x)               #32*32*128
        x = self.upsample_new3(x)           #64*64*64
        x = torch.cat([x, conv2, conv2_2, conv2_3], dim=1)    #64*64*256   

        x = self.dconv_up2(x)               #64*64*64
        x = self.upsample_new4(x)           #128*128*32
        x = torch.cat([x, conv1, conv1_2, conv1_3, conv1_4], dim=1)    #128*128*160  #conv1_5
        
        x = self.dconv_up1(x)               #128*128*32

        out1 = self.conv_last(conv1_2)
        out2 = self.conv_last(conv1_3)
        out3 = self.conv_last(conv1_4)
        out4 = self.conv_last(x)             #128*128*1    
        
        return out4
