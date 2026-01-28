import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .former import Former, CustomLayerNorm, MultiRangeGridNetBlock_Att


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        x = torch.log(x)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (b, d)


class multi_OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=2, embed_dim=64, bias=False):
        super(multi_OverlapPatchEmbed, self).__init__()

        # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_1x1 = nn.Conv2d(in_c, embed_dim//2, kernel_size=1, dilation=1, padding=0)
        self.conv_3x3_d1 = nn.Conv2d(in_c, embed_dim//2, kernel_size=3, dilation=1, padding=1)
        self.conv_3x3_d2 = nn.Conv2d(in_c, embed_dim//2, kernel_size=3, dilation=2, padding=2)
        self.conv_3x3_d3 = nn.Conv2d(in_c, embed_dim//2, kernel_size=3, dilation=3, padding=3)

    def forward(self, x):
        out1 = self.conv_1x1(x)
        out2 = self.conv_3x3_d1(x)
        out3 = self.conv_3x3_d2(x)
        out4 = self.conv_3x3_d3(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)  # 256
        return out

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs, temb_dim=None):
        super().__init__()
        self.low_freqs = n_freqs // 4
        # self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.low_conv = multi_OverlapPatchEmbed(in_channels, out_channels)
        self.low_conv_all = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)
        self.high_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5), stride=(1, 3), padding=(1, 1))
        self.high_conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 3), padding=(1, 0))
        self.high_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 7), stride=(1, 3), padding=(1, 2))
        self.high_conv_all = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)

        if temb_dim is not None:
            self.t_proj = nn.Linear(temb_dim, out_channels)

    def forward(self, x, temb=None):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]
        # print(x_high.shape)

        x_low = self.low_conv(x_low)
        x_low = self.low_conv_all(x_low)
        x_high_0 = self.high_conv(x_high)

        x_high_1 = self.high_conv_1(x_high)
        x_high_2 = self.high_conv_2(x_high)
        # print(x_high.shape, x_high_1.shape, x_high_2.shape)
        x_high = self.high_conv_all(torch.cat([x_high_0, x_high_1, x_high_2], dim=1))
        x = torch.cat([x_low, x_high], dim=-1)
        if temb is not None:
            x = x + self.t_proj(temb)[:, :, None, None]

        x = self.norm(x)
        x = self.act(x)
        return x

class DSConv_three(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs, temb_dim=None):
        super().__init__()
        self.low_freqs = n_freqs // 4
        self.mid_freqs = n_freqs // 4
        # self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.low_conv = multi_OverlapPatchEmbed(in_channels, out_channels)
        self.low_conv_all = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)


        self.mid_conv   = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,2), padding=(1,2), dilation=(1,2))
        self.mid_conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,2), padding=(1,1), dilation=(1,1))
        self.mid_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,2), padding=(1,3), dilation=(1,3))
        self.mid_conv_all = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)

        self.high_conv   = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,4), padding=(1,0), dilation=(1,2))  # d=2 → p=0（0-2=-2）
        self.high_conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,4), padding=(1,0), dilation=(1,1))  # d=1 → p=-1
        self.high_conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,4), padding=(1,1), dilation=(1,3))  # d=3 → p=1（1-3=-2）
        self.high_conv_all = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)

        if temb_dim is not None:
            self.t_proj = nn.Linear(temb_dim, out_channels)

    def forward(self, x, temb=None):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_mid = x[..., self.low_freqs:self.low_freqs+self.mid_freqs]
        x_high = x[..., self.low_freqs+self.mid_freqs:]
        # print(x_high.shape)

        x_low = self.low_conv(x_low)
        x_low = self.low_conv_all(x_low)

        x_mid_0 = self.mid_conv(x_mid)
        x_mid_1 = self.mid_conv_1(x_mid)
        x_mid_2 = self.mid_conv_2(x_mid)
        x_mid = self.mid_conv_all(torch.cat([x_mid_0, x_mid_1, x_mid_2], dim=1))


        x_high_0 = self.high_conv(x_high)
        x_high_1 = self.high_conv_1(x_high)
        x_high_2 = self.high_conv_2(x_high)
        # print(x_high_0.shape, x_high_1.shape, x_high_2.shape)
        x_high = self.high_conv_all(torch.cat([x_high_0, x_high_1, x_high_2], dim=1))
        x = torch.cat([x_low, x_mid, x_high], dim=-1)
        if temb is not None:
            x = x + self.t_proj(temb)[:, :, None, None]

        x = self.norm(x)
        x = self.act(x)
        return x

class USConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs, temb_dim=None):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.mid_freqs = n_freqs // 4
        self.low_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.mid_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), r=2)

        self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), r=4)

        self.norm = CustomLayerNorm((1, n_freqs * 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)
        if temb_dim is not None:
            self.t_proj = nn.Linear(temb_dim, out_channels)

    def forward(self, x, temb=None):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_mid = x[..., self.low_freqs:self.low_freqs+self.mid_freqs]
        # print(x_mid.shape)

        x_high = x[..., self.low_freqs+self.mid_freqs:]
        

        x_low = self.low_conv(x_low)
        x_mid = self.mid_conv(x_mid)
        # print(x_low.shape, x_mid.shape)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_mid, x_high], dim=-1)
        if temb is not None:
            x = x + self.t_proj(temb)[:, :, None, None]
        x = self.norm(x)
        x = self.act(x)

        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad(x)
        # print(x.shape)
        out = self.conv(x) # 48*3
        # print(out.shape)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=64, temb_dim=None):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels // 4, (1, 1), (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )

        self.conv_2 = DSConv(num_channels // 4, num_channels // 2, n_freqs=257, temb_dim=temb_dim)
        self.conv_3 = DSConv(num_channels // 2, num_channels // 4 * 3, n_freqs=128, temb_dim=temb_dim)
        self.conv_4 = DSConv(num_channels // 4 * 3, num_channels, n_freqs=64, temb_dim=temb_dim)
    def forward(self, x, temb=None):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x, temb)
        out_list.append(x)  # 128
        x = self.conv_3(x, temb)
        out_list.append(x)  # 64
        x = self.conv_4(x, temb)
        out_list.append(x)  # 32
        return out_list

class Encoder_2(nn.Module):
    def __init__(self, in_channels, num_channels=64, temb_dim=None):
        super(Encoder_2, self).__init__()

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, ks, padding=padding),  # 48
            nn.GroupNorm(1, num_channels, eps=1e-5),
            nn.PReLU()  # 48
        )

        self.conv2 = DSConv_three(num_channels, num_channels, n_freqs=257, temb_dim=temb_dim)

    def forward(self, x, temb=None):
        # x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x, temb)

        return x



class Decoder(nn.Module):
    def __init__(self, num_channels=64, temb_dim=None, out_channels=1):
        super(Decoder, self).__init__()
        self.up1 = USConv(num_channels * 2, num_channels // 4 * 3, n_freqs=32, temb_dim=temb_dim)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2, n_freqs=64, temb_dim=temb_dim)  # 128
        self.up3 = USConv(num_channels // 2 * 2, num_channels // 4, n_freqs=128, temb_dim=temb_dim)  # 256
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels // 4, num_channels // 4, (3, 2), padding=1),  # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
            nn.Conv2d(num_channels // 4, out_channels, (1, 1)),
        )

    def forward(self, x, encoder_out_list, temb=None):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1), temb)  # 64
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1), temb)  # 128
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1), temb)  # 256
        x = self.conv(x)  # (B,1,T,F)
        return x


class Decoder_2(nn.Module):
    def __init__(self, num_channels=64, temb_dim=None, out_channels=1):
        super(Decoder_2, self).__init__()

        self.up3 = USConv(num_channels * 2, num_channels, n_freqs=128, temb_dim=temb_dim)  # 256
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (3, 2), padding=1),  # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels),
            nn.Conv2d(num_channels, out_channels, (1, 1)),
        )

    def forward(self, x, encoder_out_list, temb=None):
        x = self.up3(torch.cat([x, encoder_out_list], dim=1), temb)  # 256
        x = self.conv(x)  # (B,1,T,F))
        return x


class Interaction(nn.Module):
    def __init__(self, num_channels, temb_dim=None):
        super().__init__()
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=(5, 5), padding=(2, 2))
        self.sigmoid = nn.Sigmoid()
        if temb_dim is not None:
            self.t_proj = nn.Linear(temb_dim, num_channels)

    def forward(self, feat_g, feat_p, temb=None):
        x = self.conv(feat_g + feat_p)
        if temb is not None:
            x = x + self.t_proj(temb)[:, :, None, None]
        mask = self.sigmoid(x)
        outs = mask * feat_p + feat_g
        return outs


class FreqAwareInteraction(nn.Module):
    def __init__(self, num_channels, n_freqs, temb_dim=None):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=(5, 5), padding=(2, 2))
        
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, n_freqs)), 
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.Sigmoid() 
        )

        if temb_dim is not None:
            self.t_proj = nn.Linear(temb_dim, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_g, feat_p, temb=None):
        x = self.conv(feat_g + feat_p)
        if temb is not None:
            x = x + self.t_proj(temb)[:, :, None, None]
        spatial_mask = self.sigmoid(x)

        freq_mask = self.freq_gate(feat_p) # (B, C, 1, F)
        final_mask = spatial_mask * freq_mask
        outs = final_mask * feat_p + feat_g
        return outs

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Downsample_2(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
                                  nn.PixelShuffle(2),
                                  nn.Conv2d(n_feat//4, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),)

    def forward(self, x):
        return self.body(x)
    
class Upsample_2(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.PixelShuffle(2),
                                nn.Conv2d(n_feat//2, n_feat, kernel_size=3, stride=1, padding=1, bias=False),
                                  )

    def forward(self, x):
        return self.body(x)

class mamba_unet(nn.Module):
    def __init__(self, num_channels):
        super(mamba_unet, self).__init__()
        self.encoder_level1 = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])  # output1
        
        self.down1_2 = Downsample(num_channels) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels*2,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])  # output2
        self.down2_3 = Downsample(int(num_channels*2)) ## From Level 2 to Level 3
        self.latent = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels*4,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])  # output3

        self.up3_2 = Upsample(int(num_channels*4)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(num_channels*4), int(num_channels*2), kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels*2,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])   # # output4
        self.up2_1 = Upsample(int(num_channels*2))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(num_channels*2), int(num_channels*1**1), kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])  # output5
        
    def forward(self, inp): # [B, C, T, F]

        out_list = []

        out_enc_level1 = self.encoder_level1(inp)  # [B, C, T, F]
        out_list.append(out_enc_level1)  # 1
        
        inp_enc_level2 = self.down1_2(out_enc_level1) 
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # [B, 2*C, T/2, F/2]
        out_list.append(out_enc_level2)  # 2

        inp_enc_level3 = self.down2_3(out_enc_level2)    
        latent = self.latent(inp_enc_level3)  # [B, 4*C, T/4, F/4]
        out_list.append(latent)  # 3

        inp_dec_level2 = self.up3_2(latent)  # [B, 2*C, T/2, F/2]

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1) # [B, 4*C, T/2, F/2]  ###########
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # [B, 2*C, T/2, F/2]
        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # [B, 2*C, T/2, F/2]
        out_list.append(out_dec_level2)  # 4

        inp_dec_level1 = self.up2_1(out_dec_level2)  # [B, C, T, F]

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1) # [B, 2*C, T, F]
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)  # [B, C, T, F]
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_list.append(out_dec_level1)  # 5

        return out_dec_level1, out_list

class ScoreNet_v3(nn.Module):
    def __init__(
            self, 
            num_channels=24, 
            temb_dim=256, 
            n_blocks=5,
            n_heads=4,
            dropout_p=0.1,
            n_fft=512, 
            hop_length=192,
        ):
        super().__init__()
        self.n_fft = n_fft 
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length

        self.embed = nn.Sequential(
            GaussianFourierProjection(temb_dim // 2),
            nn.Linear(temb_dim // 2, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
        )

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_channels, ks, padding=padding),  # 48
            nn.GroupNorm(1, num_channels, eps=1e-5),
            nn.PReLU()  # 48
        )
        self.conv2 = DSConv_three(num_channels, num_channels, n_freqs=257, temb_dim=None)

        self.encoder_g = Encoder_2(in_channels=1, num_channels=num_channels, temb_dim=temb_dim)
        self.blocks_p = mamba_unet(num_channels)

        self.encoder_level1 = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])  # output1
        
        self.down1_2 = Downsample(num_channels) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels*2,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])  # output2
        self.down2_3 = Downsample(int(num_channels*2)) ## From Level 2 to Level 3
        self.latent = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels*4,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])  # output3  ######################

        self.up3_2 = Upsample(int(num_channels*4)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(num_channels*4), int(num_channels*2), kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels*2,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])   # # output4
        self.up2_1 = Upsample(int(num_channels*2))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(num_channels*2), int(num_channels*1**1), kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(*[MultiRangeGridNetBlock_Att(
                                            emb_dim=num_channels,
                                            emb_ks=4,
                                            emb_hs=1,
                                            n_freqs=257,
                                            n_head=4,
                                            approx_qk_dim=512,
                                            activation="prelu",
                                            eps=1.0e-5)  for i in range(1)])  # output5
        self.decoder_g = Decoder_2(num_channels=num_channels, temb_dim=temb_dim, out_channels=1)
        self.up3 = USConv(num_channels * 2, num_channels, n_freqs=128, temb_dim=None)  # 256
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, (3, 2), padding=1),  # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels),
            nn.Conv2d(num_channels, 2, (1, 1)),
        )
        self.interactions0 = FreqAwareInteraction(num_channels=num_channels,n_freqs=128, temb_dim=temb_dim)
        self.interactions1 = FreqAwareInteraction(num_channels=num_channels,n_freqs=128, temb_dim=temb_dim)
        self.interactions2 = FreqAwareInteraction(num_channels=num_channels*2,n_freqs=64, temb_dim=temb_dim)
        self.interactions3 = FreqAwareInteraction(num_channels=num_channels*4,n_freqs=32, temb_dim=temb_dim)
        self.interactions4 = FreqAwareInteraction(num_channels=num_channels*2,n_freqs=64, temb_dim=temb_dim)
        self.interactions5 = FreqAwareInteraction(num_channels=num_channels,n_freqs=128, temb_dim=temb_dim)

    def apply_stft(self, x, return_complex=True):
        # x:(B,T)
        assert x.ndim == 2
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            return_complex=return_complex,
        ).transpose(1, 2)  # (B,T,F)
        return spec

    def apply_istft(self, x, length=None):
        # x:(B,T,F)
        assert x.ndim == 3
        x = x.transpose(1, 2)  # (B,F,T)
        audio = torch.istft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            length=length,
            return_complex=False
        )  # (B,T)
        return audio

    @staticmethod
    def power_compress(x):
        # x:(B,T,F)
        mag = torch.abs(x) ** 0.3 * 0.3
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    @staticmethod
    def power_uncompress(x):
        # x:(B,T,F)
        mag = (torch.abs(x) / 0.3) ** (1.0 / 0.3)
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))


    def extract_feature(self, src, tgt=None):
        if tgt is None:
            tgt = src
        src_spec = self.power_compress(self.apply_stft(src))  # (B,T,F)
        src_mag = src_spec.abs().unsqueeze(1)
        src_ri = torch.stack([src_spec.real, src_spec.imag], dim=1)

        tgt_spec = self.power_compress(self.apply_stft(tgt))  # (B,T,F)
        tgt_mag = tgt_spec.abs().unsqueeze(1)
        tgt_ri = torch.stack([tgt_spec.real, tgt_spec.imag], dim=1)

        return src_mag, src_ri, tgt_mag, tgt_ri


    def generate_wav(self, est_mag, est_pha, length):
        est_mag = torch.clip(est_mag, min=0)
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
        est_audio = self.apply_istft(self.power_uncompress(est_spec), length=length)
        return est_audio


    def forward_p(self, src_mag_ri):
        # x: (b,3,t,f)

        encoded_list = self.conv1(src_mag_ri)  # torch.Size([1, 48, 251, 257])
        encoded_list = self.conv2(encoded_list)  # torch.Size([1, 48, 251, 257])
        x = encoded_list

        feat_list = []
        feat_list.append(x)
        x, out_list = self.blocks_p(x)
        x = self.up3(torch.cat([x, encoded_list], dim=1), None)  # 256
        x = self.conv(x)  # (B,1,T,F))
        feat_list.extend(out_list)
        # x = self.deconv(x)

        return x, feat_list
    

    def forward_g(self, x, feat_list, t):
        # x: (b,1,t,f)
        temb = self.embed(t)
        # print(x.shape)
        encoded_list = self.encoder_g(x, temb)
        # x = encoded_list[-1]
        x = encoded_list

        x = self.interactions0(x, feat_list[0], temb)
        # for idx, block in enumerate(self.blocks_g):
        #     x = block(x)
        #     x = self.interactions[idx+1](x, feat_list[idx+1], temb)
        x = self.encoder_level1(x)
        x1 = x
        x = self.interactions1(x, feat_list[1], temb)
        x = self.down1_2(x)
        x = self.encoder_level2(x)
        x2 = x
        x = self.interactions2(x, feat_list[2], temb)
        x = self.down2_3(x)
        x = self.latent(x)
        x3 = x
        x = self.interactions3(x, feat_list[3], temb)
        x = self.up3_2(x)
        x = torch.cat([x, x2], 1)
        x = self.reduce_chan_level2(x)
        x = self.decoder_level2(x)
        x = self.interactions4(x, feat_list[4], temb)
        x = self.up2_1(x)
        x = torch.cat([x, x1], 1)
        x = self.reduce_chan_level1(x)
        x = self.decoder_level1(x)
        x = self.interactions5(x, feat_list[5], temb)

        
        x = self.decoder_g(x, encoded_list, temb)  # (B,1,T,F)
        return x


    def forward(self, x, t=None):
        # print(x.shape)
        # x: (b,4,t,f), t: (b,)
        if t is None:
            t = torch.tensor([0.999,], device=x.device)

        xt, src_mag_ri = x[:, :1], x[:, 1:]
        est_ri, feat_list = self.forward_p(src_mag_ri)
        sigma_z = self.forward_g(xt, feat_list, t)
        return est_ri, sigma_z


if __name__ == '__main__':
    m = ScoreNet_v3().to("cuda")
    x = torch.randn(1, 4, 124, 257).to("cuda")
    t = torch.rand(1, ).to("cuda")
    est_ri, sigma_z = m(x, t)
    print("params:", sum(p.numel() for p in m.parameters() if p.requires_grad))
    print(est_ri.shape)
    print(sigma_z.shape)
    
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(m, (4, 124, 257), as_strings=True, print_per_layer_stat=False)
        print(f"MACs: {macs}")
        print(f"Params: {params}")
