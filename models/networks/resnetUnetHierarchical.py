import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class DecoderOccupancyBlock(nn.Module):
    def __init__(self, n_channel_in, n_class_out):
        super(DecoderOccupancyBlock, self).__init__()

        # Block predicting the occupancy from the attention output (B x 128 x 12 x 12)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer0_1x1 = convrelu(128, 128, 1, 0)
        self.layer1_1x1 = convrelu(128, 128, 1, 0)
        self.layer2_1x1 = convrelu(128, 64, 1, 0)
        self.layer3_1x1 = convrelu(64, 64, 1, 0)
        self.layer4_1x1 = convrelu(64, 64, 1, 0)
        self.layer5_1x1 = convrelu(64, 32, 1, 0)
        self.layer6_1x1 = convrelu(32, 32, 1, 0)
        self.layer7_1x1 = convrelu(32, 32, 1, 0)
        self.conv_last = nn.Conv2d(32, n_class_out, 1)


    def forward(self, input):

        # ** Should this decoder be connected with map encoder 1? 
        # ** i.e. pass the map encoder layer outputs and connect them here as in the typical UNET fashion

        layer0 = self.layer0_1x1(input)
        layer1 = self.layer1_1x1(layer0)
        x = self.upsample(layer1)

        layer2 = self.layer2_1x1(x)
        layer3 = self.layer3_1x1(layer2)
        x = self.upsample(layer3)

        layer4 = self.layer4_1x1(x)
        layer5 = self.layer5_1x1(layer4)
        x = self.upsample(layer5)

        layer6 = self.layer6_1x1(x)
        layer7 = self.layer7_1x1(layer6)
        x = self.upsample(layer7)

        out = self.conv_last(x)
        return out



class ResNetUnetBlock(nn.Module):
    def __init__(self, n_channel_in, n_class_out, without_attn=False):
        super(ResNetUnetBlock, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        if without_attn:
            self.layer3_1x1 = convrelu(256, 256, 1, 0)
        else:
            self.layer3_1x1 = convrelu(256+128, 256, 1, 0)
        #self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        #self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        #self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up2 = convrelu(128 + 256, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        #self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class_out, 1)

    def forward(self, input, attn_dec_out):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        #print(x_original.shape)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        if attn_dec_out is not None:
            # Concatenate the encoded input (from Unet1) with the attention output
            layer3 = torch.cat([layer3, attn_dec_out], dim=1)
        
        layer3 = self.layer3_1x1(layer3)
        x = self.upsample(layer3)

        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class ResNetUNetHierarchical(nn.Module):
    def __init__(self, out1_n_class, out2_n_class, without_attn=False):
        super().__init__()

        #self.with_img_segm = with_img_segm

        # takes 3 spatial classes, outputs 3 spatial classes
        self.unet1 = ResNetUnetBlock(n_channel_in=out1_n_class, n_class_out=out1_n_class, without_attn=without_attn)
        #self.unet1 = DecoderOccupancyBlock(n_channel_in=out1_n_class, n_class_out=out1_n_class)

        #if with_img_segm:
        # 3 spatial classes + 1 coming from reduction of channels from object classes from img segmentation
        input_n_dim = out1_n_class + 1 #out2_n_class
        self.layer_imgSegm_in = convrelu(out2_n_class, 1, 1, 0)
        #else:
        #    input_n_dim = out1_n_class

        self.unet2 = ResNetUnetBlock(n_channel_in=input_n_dim, n_class_out=out2_n_class, without_attn=without_attn)


    #def forward(self, input, img_segm):
    def forward(self, input, img_segm, attn_dec_out=None):
        #B, T, C, cH, cW = input.shape
        #input = input.view(B*T,C,cH,cW)

        # Input is the attention decoder output
        #out1 = self.unet1(input=input)
        out1 = self.unet1(input, attn_dec_out=attn_dec_out)


        #if self.with_img_segm:
        #B, T, C, cH, cW = img_segm.shape
        #img_segm = img_segm.view(B*T,C,cH,cW)

        # reducing img segm channels from 27 to 1
        img_segm_in = self.layer_imgSegm_in(img_segm)

        #input2 = torch.cat((out1, img_segm), dim=1)
        input2 = torch.cat((out1, img_segm_in), dim=1)
        #out2 = self.unet2(input2, attn_dec_out=input)
        out2 = self.unet2(input2, attn_dec_out=attn_dec_out)

        #else:
        #    out2 = self.unet2(input=out1)

        return out1, out2