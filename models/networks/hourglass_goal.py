import torch.nn as nn
import torch
import numpy as np
# from .layers import ConvBlock, ResBlock
from .layers import Residual
from .conv_lstm import ConvLSTM


class Hourglass(nn.Module):
    def __init__(self, n, in_channels, out_channels):
        super(Hourglass, self).__init__()
        self.up1 = Residual(in_channels, 256)
        self.up2 = Residual(256, 256)
        self.up4 = Residual(256, out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = Residual(in_channels, 256)
        self.low2 = Residual(256, 256)
        self.low5 = Residual(256, 256)
        if n > 1:
            self.low6 = Hourglass(n-1, 256, out_channels)
        else:
            self.low6 = Residual(256, out_channels)
        self.low7 = Residual(out_channels, out_channels)
        # self.up5 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up = self.up1(x)
        up = self.up2(up)
        up = self.up4(up)

        low = self.pool(x)
        low = self.low1(low)
        low = self.low2(low)
        low = self.low5(low)
        low = self.low6(low)
        low = self.low7(low)
        # low = self.up5(low)
        low = nn.functional.interpolate(low, scale_factor=2)

        return up + low

class Lin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Lin, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

class StackedHourglassGoal(nn.Module):
    def __init__(self, out_channels, n, d_model, with_lstm):
        super(StackedHourglassGoal, self).__init__()
        
        self.with_lstm = with_lstm
        self.out_channels = out_channels

        # Layers for map attn input. Need to upsample before concatenating with the map representation
        #self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.r1_attn = Residual(d_model,d_model)
        self.r2_attn = Residual(d_model,d_model*2)
        #self.r2_attn = Residual(d_model,d_model)
        
        self.hg1 = Hourglass(n, d_model*2, 512)
        #self.hg1 = Hourglass(n, d_model, 256)

        self.l1 = Lin(512, 512)
        self.l2 = Lin(512, 256)
        #self.l1 = Lin(256, 256)
        #self.l2 = Lin(256, 128)

        self.out1 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)
        #self.out1 = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)


        self.out_return = nn.Conv2d(out_channels, 512, kernel_size=1, stride=1, padding=0)
        self.cat_conv = nn.Conv2d(256+d_model*2, 512, kernel_size=1, stride=1, padding=0)
        self.hg2 = Hourglass(n, 512, 512)
        #self.out_return = nn.Conv2d(out_channels, 256, kernel_size=1, stride=1, padding=0)
        #self.cat_conv = nn.Conv2d(128+d_model, 256, kernel_size=1, stride=1, padding=0)
        #self.hg2 = Hourglass(n, 256, 256)

        self.l3 = Lin(512, 512)
        self.l4 = Lin(512, 512)
        #self.l3 = Lin(256, 256)
        #self.l4 = Lin(256, 256)

        self.out2 = nn.Conv2d(512, out_channels, 1, 1, padding=0)
        #self.out2 = nn.Conv2d(256, out_channels, 1, 1, padding=0)

        ## For the waypoint location prediction, the channels represent the sequence itself: 
        ## l4 has output B x 512 x H x W and out2 has output B x num_waypoints x H x W)
        ## So we have two choices for using lstm here: 
        ## 1) flatten the HxW and pass through linear LSTM layer 
        ## 2) use ConvLSTM with single feature channel i.e. B x num_waypoints x 1 x H x W (chosen method)
        if self.with_lstm:
            self.lstm_waypoints = ConvLSTM(input_dim=1, 
                                           hidden_dim=1, 
                                           kernel_size=(1,1), 
                                           num_layers=3,
                                           batch_first=True, 
                                           bias=True, 
                                           return_all_layers=False)

    def forward(self, x_attn):
        # Layers for map attn input. Need to upsample before concatenating with the map representation
        #x_attn = self.upsample(x_attn)
        #print('hg in:', x_attn.shape)
        x_attn = self.r1_attn(x_attn)
        x_attn = self.upsample(x_attn)
        x_attn = self.r2_attn(x_attn)
        #print("r2_attn:", x_attn.shape) # r2_attn: torch.Size([1, 256, 64, 64])
        ####

        # First hourglass
        x = self.hg1(x_attn)
        #print("hg1:", x.shape)

        # Linear layers to produce first set of predictions
        x = self.l1(x)
        #print("L1:", x.shape)
        x = self.l2(x)
        #print("L2:", x.shape) # 1 x 256 x 128 x 128

        # First predicted heatmaps
        out1 = self.out1(x)
        out1_ = self.out_return(out1)

        joined = torch.cat([x, x_attn], 1)
        joined = self.cat_conv(joined)
        int1 = joined + out1_

        hg2 = self.hg2(int1)

        l3 = self.l3(hg2)
        l4 = self.l4(l3)

        out2 = self.out2(l4)

        if self.with_lstm:
            out2, _ = self.lstm_waypoints(out2.unsqueeze(2))
            out2 = out2[0].squeeze(2)
        #print('Out2:', out2.shape)

        return out2 #(out1, out2)


    def num_trainable_parameters(self):
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in trainable_parameters])


