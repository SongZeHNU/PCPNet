import torch
import torch.nn as nn
import torch.nn.functional as F
import models.convnext as convnext




class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RefMFFmodule(nn.Module):
    def __init__(self,in_ch,dim):
        super(RefMFFmodule, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,dim,3,padding=1)

        self.conv1 = nn.Conv2d(dim,dim,3,padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(dim,dim,kernel_size=7, stride=4, padding=3)  #overlap
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu2 = nn.ReLU(inplace=True)
    

        self.conv3 = nn.Conv2d(dim,dim,3,padding=1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(dim,dim,3,padding=1)
        self.bn4 = nn.BatchNorm2d(dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        # #####

        self.conv5 = nn.Conv2d(dim,dim,3,padding=1)
        self.bn5 = nn.BatchNorm2d(dim)
        self.relu5 = nn.ReLU(inplace=True)
    

        self.conv_d4 = nn.Conv2d(2*dim,dim,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(dim)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(2*dim,dim,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(dim)
        self.relu_d3 = nn.ReLU(inplace=True)

      
        self.conv_d1 = nn.Conv2d(2*dim,dim,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(dim)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(dim,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))     
        hx = self.relu2(self.bn2(self.conv2(hx1)))
       

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))
      

        hx = self.upscore2(hx5)
       

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore4(d3)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual



class MLPBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int, out_dim:int, dropout = 0.1):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, out_dim)
    def forward(self,x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x

class Mixer_chan(nn.Module):
    def __init__(self, dim= 512,channel_dim= 1024, out_channel=32, dropout = 0.75):
        super(Mixer_chan, self).__init__()
        
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            MLPBlock(dim, channel_dim, out_channel)
        )

    def forward(self,x):
        #x = x + self.token_mixer(x)
        out = self.channel_mixer(x)
        return out




class mba(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(mba, self).__init__()
        self.gelu = nn.GELU()
        
        self.convbranch1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,
                              kernel_size=(3,3), stride=1,
                              padding=1, dilation=1, groups = in_channel//out_channel, bias=False),    #in_channel//out_channel
            nn.BatchNorm2d(in_channel),
            #nn.GELU(),
            nn.Conv2d(in_channel, in_channel,
                              kernel_size=(5,5), stride=1,
                              padding=2, dilation=1, groups = in_channel//out_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            #nn.GELU(),
            nn.Conv2d(in_channel, in_channel,
                              kernel_size=(7,7), stride=1,
                              padding=3, dilation=1, groups = in_channel//out_channel, bias=False),
            nn.BatchNorm2d(in_channel)
        )
       
        
        self.mlpbranch = nn.Sequential(
            Mixer_chan(dim = in_channel, channel_dim = 4*in_channel, out_channel = out_channel),
        )
       
    def forward(self, x):
        convx = self.convbranch1(x) + x
        convx = convx.transpose(1,3)
        mlpx = self.mlpbranch(convx)
        mlpx = mlpx.transpose(1,3)
        x = self.gelu(mlpx)
        return x



class nromalDecoder(nn.Module):
    def __init__(self, channel):
        super(nromalDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
       
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_upsample41 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample42 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample43 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_downsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = x2
        x3_1 = x3  
        x4_1 = x4

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x4_2 = torch.cat((x4_1, self.conv_upsample6(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x, x1_1, x2_2, x3_2






class Network(nn.Module):
    def __init__(self, args, channel=48):
        super(Network, self).__init__()
        
        self.model = convnext.convnext_tiny(pretrained = True)

        #to channel 
        self.eem1_1 = mba(96, channel)     #channel
        self.eem2_1 = mba(192, channel) 
        self.eem3_1 = mba(384, channel)
        self.eem4_1 = mba(768, channel)

        # Decoder 
        self.ND = nromalDecoder(channel)
        self.refine = RefMFFmodule(1, 64)
 
   
    def forward(self, x):
        # Feature Extraction
        x = self.model(x)

        b1 = self.eem1_1(x[0])        # channel -> 48
        b2 = self.eem2_1(x[1])        # channel -> 48
        b3 = self.eem3_1(x[2])        # channel -> 48
        b4 = self.eem4_1(x[3])        # channel -> 48
     
        S_g, h4, h3, h2 = self.ND(b4, b3, b2, b1)
        #coarse map
        S_g1 = F.interpolate(S_g, scale_factor=4, mode='bilinear')  
        S_g_pred = self.refine(S_g1)

    


        return S_g_pred , S_g1

