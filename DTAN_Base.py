import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from DTA import DTA1024, DTA512, DTA256


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        out = self.conv(input)
        return out

class PreConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.block(x)


class DTA_Module(nn.Module):
    def __init__(self, in_ch, dta_block, out_ch):
        super().__init__()
        self.pre = PreConv(in_ch, out_ch)
        self.dta = dta_block
        self.post = nn.Conv2d(out_ch * 2, out_ch,  1)

    def forward(self, x):
        x_pre = self.pre(x)
        x_dta = self.dta(x_pre)
        x_cat = torch.cat([x_pre, x_dta], dim=1)
        return self.post(x_cat)


class MSF_Module(nn.Module):
    def __init__(self, enc_ch, dec_ch, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.enc_to_2c = nn.Conv2d(enc_ch, dec_ch, kernel_size=1)
        self.drop = nn.Dropout2d(p=0.1)
        self.fuse = nn.Conv2d(dec_ch, dec_ch, kernel_size=1)

    def forward(self, f_enc, f_dec):
        score = self.enc_to_2c(f_enc)
        score = F.adaptive_avg_pool2d(score, output_size=f_dec.shape[-2:])
        score = self.alpha * score
        score = self.drop(score)
        f_dec = f_dec * (1.0 + score)
        return self.fuse(f_dec)


class MARBlock(nn.Module):
    def __init__(self, c1, c2, c3, d=64, p_drop=0.1):
        super().__init__()
        self.q_proj = nn.Conv2d(c1, d, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(c2, d, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(c3, d, kernel_size=1, bias=True)

        self.pool_k = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool_v = nn.AvgPool2d(kernel_size=4, stride=4)

        self.up1 = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)

        self.drop = nn.Dropout2d(p=p_drop)
        self.fuse = nn.Conv2d(d + c3, d, 1)

    def forward(self, f1, f2, f3):
        b = f1.size(0)

        q = self.q_proj(f1)
        k = self.k_proj(self.pool_k(f2))
        v = self.v_proj(self.pool_v(f3))

        q_bnC = q.flatten(2).transpose(1, 2).contiguous()
        k_bnC = k.flatten(2).transpose(1, 2).contiguous()
        v_bnC = v.flatten(2).transpose(1, 2).contiguous()

        d = q_bnC.size(-1)
        attn = torch.matmul(q_bnC, k_bnC.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(attn, dim=-1)

        out_bnC = torch.matmul(attn, v_bnC)
        out = out_bnC.transpose(1, 2).reshape(b, d, q.size(2), q.size(3))

        out = self.up1(out)
        out = self.drop(out)
        out = self.up2(out)
        out = self.drop(out)

        fused = torch.cat([out, f3], dim=1)
        fused = self.fuse(fused)
        return fused

class DTAN_Base(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_classes=1):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        # Encoder
        self.enc1 = DoubleConv(in_ch, c1)
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(c1, c2)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc3_pre = PreConv(c2, c3)
        self.enc3_dta = DTA256(c3)
        self.enc3_post = nn.Conv2d(c3 * 2, c3, 1)
        self.down3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc4_pre = PreConv(c3, c4)
        self.enc4_dta = DTA512(c4)
        self.enc4_post = nn.Conv2d(c4 * 2, c4, 1)
        self.down4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc5_pre = PreConv(c4, c5)
        self.enc5_dta = DTA1024(c5)
        self.enc5_post = nn.Conv2d(c5 * 2, c5, 1)

        self.dec4_pre = PreConv(c4, c4)
        self.dec4_post = nn.Conv2d(c4 * 2, c4,1)

        self.dec3_pre = PreConv(c3, c3)
        self.dec3_post = nn.Conv2d(c3 * 2, c3, 1)

        # Decoder upsamplers
        self.up1 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)  # -> 32x32
        self.up2 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)  # -> 64x64
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)  # -> 128x128
        self.up4 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)  # -> 256x256

        # Skip-concat fusion convs
        self.dec1_merge = DoubleConv(c4 * 2, c4)  # with e4 at 32x32
        self.dec2_merge = DoubleConv(c3 * 2, c3)  # with e3 at 64x64
        self.dec3_merge = DoubleConv(c2 * 2, c2)  # with e2 at 128x128
        self.dec4_merge = DoubleConv(c1 * 2, c1)  # with e1 at 256x256

        # DTA on decoder paths
        self.dec1_dta = DTA512(c4)
        self.dec1_post = nn.Conv2d(c4 * 2, c4, 3, 1, 1)

        self.dec2_dta = DTA256(c3)
        self.dec2_post = nn.Conv2d(c3 * 2, c3, 3, 1, 1)

        # Four MSFs with exact scale mapping
        self.msf_16  = MSF_Module(enc_ch=c4, dec_ch=c5, alpha=1.0)  # 32x32 encoder -> 16x16 decoder (bottleneck)
        self.msf_32  = MSF_Module(enc_ch=c3, dec_ch=c4, alpha=1.0)  # 64x64 encoder -> 32x32 decoder
        self.msf_64  = MSF_Module(enc_ch=c2, dec_ch=c3, alpha=1.0)  # 128x128 encoder -> 64x64 decoder
        self.msf_128 = MSF_Module(enc_ch=c1, dec_ch=c2, alpha=1.0)  # 256x256 encoder -> 128x128 decoder

        # MAR + head
        self.mar = MARBlock(c1=c3, c2=c2, c3=c1, d=c1, p_drop=0.1)
        self.reduce_after_mar = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)         # 256x256, c1
        p1 = self.down1(e1)

        e2 = self.enc2(p1)        # 128x128, c2
        p2 = self.down2(e2)

        e3_pre = self.enc3_pre(p2)    # 64x64, c3
        e3_dta = self.enc3_dta(e3_pre)
        e3 = self.enc3_post(torch.cat([e3_pre, e3_dta], dim=1))
        p3 = self.down3(e3)

        e4_pre = self.enc4_pre(p3)   # 32x32, c4
        e4_dta = self.enc4_dta(e4_pre)
        e4 = self.enc4_post(torch.cat([e4_pre, e4_dta], dim=1))
        p4 = self.down4(e4)

        e5_pre = self.enc5_pre(p4)   # 16x16, c5
        e5_dta = self.enc5_dta(e5_pre)
        e5 = self.enc5_post(torch.cat([e5_pre, e5_dta], dim=1))

        # MSF#4: 32x32 encoder -> 16x16 decoder (bottleneck)
        e5 = self.msf_16(e4, e5)

        # Decoder stage at 32x32
        d1_up = self.up1(e5)

        d1_merge = torch.cat([d1_up, e4], dim=1)
        d1 = self.dec1_merge(d1_merge)

        d1_pre = self.dec4_pre(d1)
        d1_dta = self.dec1_dta(d1_pre)
        d1 = self.dec4_post(torch.cat([d1_pre, d1_dta], dim=1))


        d1 = self.msf_32(e3, d1)

        # Decoder stage at 64x64


        d2_up = self.up2(d1)

        d2_merge = torch.cat([d2_up, e3], dim=1)
        d2 = self.dec2_merge(d2_merge)

        d2_pre = self.dec3_pre(d2)
        d2_dta = self.dec2_dta(d2_pre)
        d2 = self.dec3_post(torch.cat([d2_pre, d2_dta], dim=1))

        d2 = self.msf_64(e2, d2)

        # Decoder stage at 128x128

        d3_up = self.up3(d2)

        d3_merge = torch.cat([d3_up, e2], dim=1)
        d3 = self.dec3_merge(d3_merge)


        d3 = self.msf_128(e1, d3)


        # Decoder stage at 256x256
        d4_up = self.up4(d3)
        d4_merge = torch.cat([d4_up, e1], dim=1)
        d4 = self.dec4_merge(d4_merge)                            # 256x256, c1

        # MAR taps
        c7 = d2  # 64x64,  c3
        c8 = d3  # 128x128, c2
        c9 = d4  # 256x256, c1

        mar_fused = self.mar(c7, c8, c9)
        out_feat = torch.cat([mar_fused, c9], dim=1)
        out_feat = self.reduce_after_mar(out_feat)
        out = self.head(out_feat)
        return out
