import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torchvision.transforms import Resize

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size =3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),  
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.2),  
        )
    def forward(self, x):
        return self.double_conv(x)

class DoubleConvRes(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        self.res_connect = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2,bias=False)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        res = self.res_connect(x)
        double_conv = self.double_conv(x)
        return self.act(res+double_conv)


class DoubleConvStriped(nn.Module):
    """Striped Conv"""

    def __init__(self, in_channels, out_channels,kernel_size = 3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)


class MSABlock(nn.Module):
    """MSA block"""

    def __init__(self,channels):
        super().__init__()
        self.strip_conv1 = DoubleConvStriped(channels,channels,kernel_size=3)
        self.strip_conv2 = DoubleConvStriped(channels,channels,kernel_size=7)
        self.strip_conv3 = DoubleConvStriped(channels,channels,kernel_size=11)
        self.conv1x1 = nn.Conv2d(3*channels, 1, kernel_size=1,bias=False)
        self.attn_func = nn.Sigmoid()

    def forward(self, x):
        strip1 = self.strip_conv1(x)
        strip2 = self.strip_conv2(x)
        strip3 = self.strip_conv3(x)
        strip_concat = torch.cat([strip1,strip2,strip3],dim=1)
        attn = self.attn_func(self.conv1x1(strip_concat))
        out = attn*x
        return out

class MSA(nn.Module):
    """MSA"""

    def __init__(self,c1,c2,c3,c4):
        super().__init__()
        self.msa_1 = MSABlock(c1)
        self.msa_2 = MSABlock(c2)
        self.msa_3 = MSABlock(c3)
        self.msa_4 = MSABlock(c4)

    def forward(self, x1,x2,x3,x4):
        x1_ = self.msa_1(x1)
        x2_ = self.msa_2(x2)
        x3_ = self.msa_3(x3)
        x4_ = self.msa_4(x4)
        return x1_,x2_,x3_,x4_
    
    
class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, droprate=0.2, is_res=False):
        super().__init__()
        self.is_res = is_res 
        
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(droprate)    
        )
        self.conv2_layer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(droprate)  
        )
        
        if is_res:
            if in_channels == out_channels:
                self.resid_layer = nn.Identity()
            else:
                self.resid_layer = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1_layer(x)
        x2 = self.conv2_layer(x1)
        if self.is_res:
            out = x2 + self.resid_layer(x)
            return out / 1.414
        else:
            return x2
        
        
class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, droprate=0.2, activation=True):
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()
        self.drop = nn.Dropout(droprate) if droprate > 0 else nn.Identity()
        self.out_channels = out_channels

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)

        y = self.fc(x_flat)
        y = self.act(y)
        y = self.drop(y)

        y = y.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)

        return y
    
    
class HorizontalGRU(nn.Module):
    """
    Horizontal (width-wise) context modeling using GRU.
    Input:  x [B, C, H, W]
    Output: y [B, out_channels, H, W]
    """
    def __init__(self, channels, out_channels, hidden_size=None, bidirectional=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.hidden_size = out_channels//(2 if bidirectional else 1) if hidden_size is None else hidden_size
        self.bidirectional = bidirectional

        self.pool = nn.AdaptiveAvgPool2d((1, None))  # (1, W) keep W dynamic

        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )

        gru_out_dim = self.hidden_size * (2 if bidirectional else 1)


        proj_in = self.hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(proj_in, out_channels) if proj_in != out_channels else nn.Identity()

    def forward(self, x, H, W):
        B, C, _, _ = x.shape

        strip_pooling = nn.AdaptiveMaxPool2d((1, W))
        strip_x = strip_pooling(x)          # (B,C,1,W)
        strip_x = strip_x.squeeze(2).transpose(1, 2)  # (B,W,C)

        y, _ = self.gru(strip_x)             # (B,W,hidden)
        y = self.proj(y)                     # (B,W,outC)

        y = y.transpose(1, 2).unsqueeze(2)   # (B,outC,1,W)
        y = y.expand(-1, -1, H, -1)           # (B,outC,H,W)
        return y
    
class VerticalGRU(nn.Module):
    """
    Vertical context modeling along Height (H):
    x (B,C,H,W) -> pool to (B,C,H,1) -> seq (B,H,C) -> GRU -> (B,H,outC)
    -> (B,outC,H,1) -> expand to (B,outC,H,W)
    """
    def __init__(self, in_channels, out_channels, hidden_size=None, num_layers=1,
                 bidirectional=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        hidden_size = out_channels//(2 if bidirectional else 1) if hidden_size is None else hidden_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.strip_pooling = None  # created in forward because H is provided there

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        proj_in = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(proj_in, out_channels) if proj_in != out_channels else nn.Identity()

    def forward(self, x, H, W):
        # x: (B,C,H,W)
        B, C, _, _ = x.shape

        strip_pooling = nn.AdaptiveMaxPool2d((H, 1))
        strip_x = strip_pooling(x)                 # (B,C,H,1)
        strip_x = strip_x.squeeze(-1).transpose(1, 2)  # (B,H,C)

        y, _ = self.gru(strip_x)                   # (B,H,hidden*(1 or 2))
        y = self.proj(y)                           # (B,H,outC)

        y = y.transpose(1, 2).unsqueeze(-1)        # (B,outC,H,1)
        y = y.expand(-1, -1, -1, W)                # (B,outC,H,W)
        return y
    
    
class SpatialAttention2D(nn.Module):
    """
    x: (B,C,H,W) -> tokens (B,N,C) where N=H*W
    -> QKV attention -> (B,N,outC) -> reshape (B,outC,H,W)
    """
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.qkv = nn.Linear(in_channels, out_channels * 3, bias=True)
        self.proj = nn.Linear(out_channels, out_channels, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, H=None, W=None):
        B, C, Hx, Wx = x.shape
        H = Hx if H is None else H
        W = Wx if W is None else W

        # (B,C,H,W) -> (B,N,C)
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        N = tokens.shape[1]

        qkv = self.qkv(tokens)  # (B,N,3*outC)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B,N,outC) -> (B,heads,N,head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,heads,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B,heads,N,head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.out_channels)  # (B,N,outC)

        # out = self.proj(out)
        # out = self.proj_drop(out)

        # back to (B,outC,H,W)
        out = out.transpose(1, 2).contiguous().view(B, self.out_channels, H, W)
        return out

class HorizontalAttention(nn.Module):
    def __init__(self,channels,out_channels):
        super(HorizontalAttention,self).__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.l1 = nn.Linear(self.channels,self.out_channels)
        self.l2 = nn.Linear(self.channels,self.out_channels)
        self.l3 = nn.Linear(self.channels,self.out_channels)

    def forward(self, x,H,W):
        strip_pooling = nn.AdaptiveAvgPool2d((1, W))
        strip_x = strip_pooling(x).reshape(x.shape[0],-1,W)
        strip_x = strip_x.transpose(2,1)  # b w c

        Q = self.l1(strip_x) # b w c
        K = self.l2(strip_x) # b w c
        V = self.l3(strip_x) # b w c
        qk = torch.matmul(Q, K.transpose(2,1))
        qk = qk / math.sqrt(self.out_channels)
        qk = nn.Softmax(dim=-1)(qk)
        qkv = torch.matmul(qk, V)
        qkv = qkv.transpose(2,1)
        qkv = torch.unsqueeze(qkv,dim=2)
        qkv_expend = qkv.expand((-1,-1,H,-1))
        return qkv_expend

class VerticalAttention(nn.Module):
    def __init__(self,channels,out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.l1 = nn.Linear(self.channels,self.out_channels)
        self.l2 = nn.Linear(self.channels,self.out_channels)
        self.l3 = nn.Linear(self.channels,self.out_channels)


    def forward(self, x,H,W):
        strip_pooling = nn.AdaptiveMaxPool2d((H,1))
        strip_x = strip_pooling(x).reshape(x.shape[0],-1,H)
        strip_x = strip_x.transpose(2,1)  # b H c
        Q = self.l1(strip_x) # b w c
        K = self.l2(strip_x) # b w c
        V = self.l3(strip_x) # b w c
        qk = torch.matmul(Q, K.transpose(2,1))
        qk = qk / math.sqrt(self.out_channels)
        qk = nn.Softmax(dim=-1)(qk)
        qkv = torch.matmul(qk, V)
        qkv = qkv.transpose(2,1)
        qkv = torch.unsqueeze(qkv,dim=3)
        qkv_expend = qkv.expand((-1,-1,-1,W))
        return qkv_expend


class GSA(nn.Module):
    """GSA"""
    def __init__(self,c1,c2,c3,c4,out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(c1+c2+c3+c4, out_channels, kernel_size=1,bias=False)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.out_channels = out_channels
        self.horizontal_attention = HorizontalAttention(out_channels,self.out_channels) #HorizontalGRU(channels=out_channels, out_channels=self.out_channels, bidirectional=True) #
        self.vertical_attention = VerticalAttention(out_channels,self.out_channels) #VerticalGRU(in_channels=out_channels, out_channels=self.out_channels, bidirectional=True) #
        
        self.spatial_attention = SpatialAttention2D(in_channels=out_channels, out_channels=self.out_channels, num_heads=1)


    def forward(self, x1,x2,x3,x4):
        t_h, t_w = x1.shape[-2:]
        up = nn.Upsample(size=(t_h, t_w), mode='bilinear', align_corners=True)
        x2_ = up(x2)
        x3_ = up(x3)
        x4_ = up(x4)
        x_concat = torch.cat([x1,x2_,x3_,x4_],dim=1)
        x_concat_ = self.conv1x1(x_concat)
        # hor_attn = self.horizontal_attention(x_concat_,t_h, t_w)
        # ver_attn = self.vertical_attention(x_concat_,t_h, t_w)
        # out = hor_attn+ver_attn+x_concat_
        spatial_attn = self.vertical_attention(x_concat_, t_h, t_w) + self.horizontal_attention(x_concat_,t_h, t_w)
        out = spatial_attn + x_concat_
        x1_out = out
        x2_out = Resize(x2.shape[-2:])(out)
        x3_out = Resize(x3.shape[-2:])(out)
        x4_out = Resize(x4.shape[-2:])(out)
        return x1_out,x2_out,x3_out,x4_out



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UPBilinear(nn.Module):
    def __init__(self, in_channels, mid_channels,out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                )
        self.conv = DoubleConv(in_channels+mid_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is None:
            x = x1
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.conv(x)



class StripedWriNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(StripedWriNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        init_c = 32
        self.init_c = init_c
        # self.inc = Stage0(n_channels, init_c)
        self.inc = DoubleConv(n_channels, init_c)
        self.down1 = Down(init_c, init_c*2)
        self.down2 = Down(init_c*2, init_c*4)
        self.down3 = Down(init_c*4, init_c*8)
        self.down4 = Down(init_c*8, init_c*16)

      
        self.msa = MSA(init_c,init_c*2,init_c*4,init_c*8)
        self.gsa = GSA(init_c,init_c*2,init_c*4,init_c*8,init_c)

        self.up1 = UPBilinear(init_c*16, init_c*8,init_c*8)
        self.up2 = UPBilinear(init_c*8, init_c*4,init_c*4)
        self.up3 = UPBilinear(init_c*4, init_c*2,init_c*2)
        self.up4 = UPBilinear(init_c*2, init_c*1,init_c*1)
        
        # self.proj_x3 = nn.Conv2d(init_c*4, init_c*2, kernel_size=1, bias=False)
        # self.proj_x4 = nn.Conv2d(init_c*8, init_c*4, kernel_size=1, bias=False)

        self.outc = OutConv(init_c*1, n_classes)
        
        self.DoubleCNN1 = DoubleConv(init_c, 2*init_c, kernel_size = 3)
        self.DoubleCNN2 = DoubleConv(2*init_c, 2*init_c,kernel_size = 3)
        self.DoubleCNN3 = DoubleConv(2*init_c, 2*init_c,kernel_size = 3)
        self.DoubleCNN4 = DoubleConv(2*init_c, 2*init_c,kernel_size = 3)
        self.MLP1 = MLP(2*init_c,n_classes)
        
        
        self.x2_mir = None
        self.x3_mir = None
        self.x4_mir = None
        self.x5_mir = None
        
        self.x2_ref = None
        self.x3_ref = None
        self.x4_ref = None
        self.x5_ref = None
        

    def forward(self, x):
        input_shape = x.shape[-2:]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x2_mir = self.DoubleCNN1(x1)
        x3_mir = self.DoubleCNN2(x2_mir)
        x4_mir = self.DoubleCNN3(x3_mir)
        x5_mir = self.DoubleCNN4(x4_mir)
        
        bottle = self.MLP1(x5_mir)
     
        
        # self.x2_ref = x2 #
        # self.x3_ref = x3 # F.interpolate(x3, size=self.x3_mir.shape[-2:], mode="bilinear", align_corners=False)
        # self.x4_ref = x4 # F.interpolate(x4, size=self.x4_mir.shape[-2:], mode="bilinear", align_corners=False)
        # self.x5_ref = x5 # F.interpolate(x5, size=self.x5_mir.shape[-2:], mode="bilinear", align_corners=False)
        # self.x2_mir = nn.MaxPool2d(2)(x2_mir)
        # self.x3_mir = nn.MaxPool2d(4)(x3_mir)
        # self.x4_mir = nn.MaxPool2d(8)(x4_mir)
        # self.x5_mir = nn.MaxPool2d(16)(x5_mir)

        
        
        # x1, x2, x3, x4 = self.msa(x1,x2,x3,x4)
        # x1, x21, x31, x41 = self.gsa(x1,x2,x3,x4)
        

        # x3 = self.proj_x3(x3)
        # x4 = self.proj_x4(x4)
        
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        
        x = self.outc(x)
        
        x = bottle+x

        out = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return out
    
    def loss_with_aux_ce(self, outputs, target, w=(0.01, 0.01, 0.01, 0.01)):
        if self.x2_ref is not None:
            loss_main = F.cross_entropy(outputs, target)
            loss_aux1 = F.cross_entropy(self.x2_mir, self.x2_ref)
            loss_aux2 = F.cross_entropy(self.x3_mir, self.x3_ref)
            loss_aux3 = F.cross_entropy(self.x4_mir, self.x4_ref)
            loss_aux4 = F.cross_entropy(self.x5_mir, self.x5_ref)

            return loss_main + w[0]*loss_aux1 + w[1]*loss_aux2 + w[2]*loss_aux3 + w[3]*loss_aux4
        else:
            return F.cross_entropy(outputs, target)



if __name__ == '__main__':
    input = torch.rand([2,3,256,256])
    model = StripedWriNet(n_channels=3, n_classes=2)
    print(model(input).shape)
