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

class StackedHourglass(nn.Module):
    def __init__(self, out_channels, n, d_model, with_lstm):
        super(StackedHourglass, self).__init__()
        
        self.with_lstm = with_lstm
        self.out_channels = out_channels

        # Layers for map attn input. Need to upsample before concatenating with the map representation
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.r1_attn = Residual(d_model,d_model)
        #self.r2_attn = Residual(128,128)
        self.r2_attn = Residual(d_model,d_model*2)
        ###


        # Replacing initial layers for map input processing to accomodate instead map encoding as input
        #self.r1 = Residual(128,128)
        #self.r2 = Residual(128,128)        
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.r1 = Residual(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, 128)
        self.r6 = Residual(128, 256)
        '''


        self.hg1 = Hourglass(n, d_model*2, 512)
        #self.hg1 = Hourglass(4, 512, 512)

        self.l1 = Lin(512, 512)
        self.l2 = Lin(512, 256)

        self.out1 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

        #self.out_return = nn.Conv2d(out_channels, 256+128, kernel_size=1, stride=1, padding=0)
        #self.cat_conv = nn.Conv2d(256+128, 256+128, kernel_size=1, stride=1, padding=0)
        #self.hg2 = Hourglass(4, 256+128, 512)
        self.out_return = nn.Conv2d(out_channels, 512, kernel_size=1, stride=1, padding=0)
        self.cat_conv = nn.Conv2d(256+d_model*2, 512, kernel_size=1, stride=1, padding=0)
        self.hg2 = Hourglass(n, 512, 512)

        self.l3 = Lin(512, 512)
        self.l4 = Lin(512, 512)

        self.out2 = nn.Conv2d(512, out_channels, 1, 1, padding=0)

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
            self.lstm_headings = nn.LSTM(input_size=2, 
                                         hidden_size=2,
                                         num_layers=3,
                                         batch_first=True)

        # Add layers for heading prediction
        self.conv_heading = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        #self.heading_out = nn.Conv2d(256, out_channels, 1, 1, padding=0)
        #self.heading_out = nn.Conv2d(64, out_channels, 1, 1, padding=0)
        self.Lin_heading_1 = nn.Linear(512, 128)
        self.relu_heading_1 = nn.ReLU()
        self.Lin_heading_2 = nn.Linear(128, 64)
        self.relu_heading_2 = nn.ReLU()
        self.Lin_heading_out = nn.Linear(64, out_channels*2)
        self.heading_tanh = torch.nn.Hardtanh(min_val=-1,max_val=1)

        ###

    #def forward(self, x, x_attn):
    def forward(self, x_attn):

        # Layers for processing the map encoding
        #x = self.upsample(x)
        #x = self.r1(x)
        #x = self.upsample(x)
        #x_r2 = self.r2(x)
        #print("r2:", x_r2.shape)
        ####
        '''
        x = self.conv1(x)
        #print("Conv1:", x.shape)
        x = self.r1(x)
        #print("r1:", x.shape)
        pooled = self.pool(x)
        #print("pooled:", pooled.shape) # 1 x 128 x 128 x 128
        x = self.r4(pooled)
        #print("r4:", x.shape)
        x = self.r5(x)
        #print("r5:", x.shape)
        x = self.r6(x)
        #print("r6:", x.shape)
        '''

        # Layers for map attn input. Need to upsample before concatenating with the map representation
        #x_attn = self.upsample(x_attn)
        #print('hg in:', x_attn.shape)
        x_attn = self.r1_attn(x_attn)
        x_attn = self.upsample(x_attn)
        x_attn = self.r2_attn(x_attn)
        #print("r2_attn:", x_attn.shape) # r2_attn: torch.Size([1, 256, 64, 64])
        ####

        ## Concatenate map attn with map representation
        #x_concat = torch.cat((x_r2, x_attn), dim=1) # 1 x 256 x 128 x 128
        #print("Concatenated:", x_concat.shape)

        # First hourglass
        #x = self.hg1(x_concat)
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

        #joined = torch.cat([x, pooled], 1)
        #joined = torch.cat([x, x_concat], 1)
        joined = torch.cat([x, x_attn], 1)
        #print("Joined:", joined.shape)
        joined = self.cat_conv(joined)
        int1 = joined + out1_

        hg2 = self.hg2(int1)
        #print("hg2:", hg2.shape)

        l3 = self.l3(hg2)
        #print("L3:", l3.shape)
        l4 = self.l4(l3)
        #print("L4:", l4.shape)

        out2 = self.out2(l4)

        if self.with_lstm:
            out2, _ = self.lstm_waypoints(out2.unsqueeze(2))
            out2 = out2[0].squeeze(2)
            #h = last_states[0][0]
        #print('Out2:', out2.shape)

        ## Predict headings
        x_heading = self.conv_heading(l4)
        x_heading = x_heading.view(x_heading.shape[0], -1,)
        x_heading = self.Lin_heading_1(x_heading)
        x_heading = self.relu_heading_1(x_heading)
        x_heading = self.Lin_heading_2(x_heading)
        x_heading = self.relu_heading_2(x_heading)
        out_heading = self.Lin_heading_out(x_heading) # B x 20
    
        if self.with_lstm:
            out_heading = out_heading.view(out_heading.shape[0], -1, self.out_channels)
            out_heading = out_heading.permute(0,2,1) # B x num_waypoints x 2
            hidden = (torch.randn(3, out_heading.shape[0], 2).to(out_heading.device), 
                      torch.randn(3, out_heading.shape[0], 2).to(out_heading.device))
            out_heading, hidden = self.lstm_headings(out_heading, hidden)
            out_heading = out_heading.contiguous()
            out_heading = out_heading.view(out_heading.shape[0], -1)

        out_heading = self.heading_tanh(out_heading)
        out_heading = out_heading.view(out_heading.shape[0], -1, self.out_channels)
        #print("out heading:", out_heading.shape)

        return (out1, out2), out_heading


    def num_trainable_parameters(self):
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in trainable_parameters])


