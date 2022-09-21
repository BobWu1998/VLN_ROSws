import torch
import torch.nn as nn
#import torch.nn.functional as F
from torchvision import models
from .conv_lstm import ConvLSTM

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class ResNetUnetBlock(nn.Module):
    def __init__(self, n_channel_in, n_class_out, with_lstm, high_res_heatmaps):
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
        #self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        #self.layer3_1x1 = convrelu(256, 256, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #self.conv_up2 = convrelu(128 + 256, 256, 3, 1)
        #self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        #self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_up1 = convrelu(64 + 128, 128, 3, 1)
        self.conv_up0 = convrelu(64 + 128, 128, 3, 1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class_out, 1)

        self.high_res_heatmaps = high_res_heatmaps
        if self.high_res_heatmaps:
            self.conv_highres_1x1 = convrelu(n_channel_in, n_channel_in, 1, 0)
            self.conv_cov_highres = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ) 

        # Layers for waypoint coverage prediction
        # Formulated as a classification problem where each waypoint can be either uncovered (0), or covered (1)
        self.conv_coverage = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.Lin_cov1 = nn.Linear(72, 32)
        self.relu_cov1 = nn.ReLU()
        self.Lin_cov2 = nn.Linear(32, n_class_out*2)

        self.with_lstm = with_lstm
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


    def forward(self, input):
        input = self.upsample(input)

        if self.high_res_heatmaps:
            input = self.conv_highres_1x1(input)
            input = self.upsample(input)
        
        x_original = self.conv_original_size0(input)
        #print("x_original:", x_original.shape)
        x_original = self.conv_original_size1(x_original)
        #print("x_original:", x_original.shape)

        layer0 = self.layer0(input)
        #print("layer0:", layer0.shape)
        layer1 = self.layer1(layer0)
        #print("layer1:", layer1.shape)
        layer2 = self.layer2(layer1)
        #layer3 = self.layer3(layer2)
        #print(layer2.shape)
        
        #layer3 = self.layer3_1x1(layer3)
        #x = self.upsample(layer3)

        layer2 = self.layer2_1x1(layer2)
        #print("layer2:", layer2.shape)
        x = self.upsample(layer2)

        #layer2 = self.layer2_1x1(layer2)
        #x = torch.cat([x, layer2], dim=1)
        #x = self.conv_up2(x)

        #x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        #print("x:", x.shape)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        #x = self.upsample(x)
        out = self.conv_last(x)

        if self.with_lstm:
            out, _ = self.lstm_waypoints(out.unsqueeze(2))
            out = out[0].squeeze(2)

        # Prediction of covered waypoints 
        # (binary indicator of which waypoints are before or after the agent's current position)
        if self.high_res_heatmaps:
            x = self.conv_cov_highres(x)
        x_cov = self.conv_coverage(x)
        x_cov = x_cov.view(x_cov.shape[0], -1)
        x_cov = self.Lin_cov1(x_cov)
        x_cov = self.relu_cov1(x_cov)
        out_cov = self.Lin_cov2(x_cov)
        out_cov = out_cov.view(x_cov.shape[0], -1, 2)

        return out, out_cov


class ResNetUNetGoalPred(nn.Module):
    def __init__(self, n_channel_in, n_class_out, with_lstm, high_res_heatmaps=False):
        super().__init__()

        self.unet1 = ResNetUnetBlock(n_channel_in=n_channel_in, n_class_out=n_class_out, with_lstm=with_lstm, high_res_heatmaps=high_res_heatmaps)

        #self.unet2 = ResNetUnetBlock(n_channel_in=input_n_dim, n_class_out=out2_n_class)


    def forward(self, input):


        out1 = self.unet1(input=input)
        #print(out1.shape)

        #out2 = self.unet2(input=out1)

        return out1 #, out2