import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from DTA import DTA1024, DTA512, DTA256


def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(in_ch, out_ch, 3, 1, 1),
            conv_bn_relu(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        return self.block(x)


class PreConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = conv_bn_relu(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        return self.block(x)


class DTA_Module(nn.Module):
    def __init__(self, in_ch, dta_block, out_ch):
        super().__init__()
        self.pre = PreConv(in_ch, out_ch)
        self.dta = dta_block
        self.post = conv_bn_relu(out_ch * 2, out_ch, 3, 1, 1)

    def forward(self, x):
        x_pre = self.pre(x)
        x_dta = self.dta(x_pre)
        x_cat = torch.cat([x_pre, x_dta], dim=1)
        return self.post(x_cat)


class MSF_Module(nn.Module):
    def __init__(self, enc_ch, dec_ch, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.enc_to_2c = conv_bn_relu(enc_ch, 2 * enc_ch, 3, 1, 1)
        self.score_proj = nn.Conv2d(2 * enc_ch, dec_ch, kernel_size=1, bias=True)
        self.drop = nn.Dropout2d(p=0.1)
        self.fuse = conv_bn_relu(dec_ch, dec_ch, 3, 1, 1)

    def forward(self, f_enc, f_dec):
        score = self.enc_to_2c(f_enc)
        score = self.score_proj(score)
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
        self.fuse = conv_bn_relu(d + c3, d, 3, 1, 1)

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

        self.enc1 = DoubleConv(in_ch, c1)
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(c1, c2)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(c2, c3)
        self.enc3_dta = DTA256(c3)
        self.enc3_post = conv_bn_relu(c3 * 2, c3, 3, 1, 1)
        self.down3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc4_pre = PreConv(c3, c4)
        self.enc4_dta = DTA512(c4)
        self.enc4_post = conv_bn_relu(c4 * 2, c4, 3, 1, 1)
        self.down4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc5_pre = PreConv(c4, c5)
        self.enc5_dta = DTA1024(c5)
        self.enc5_post = conv_bn_relu(c5 * 2, c5, 3, 1, 1)

        self.up1 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec1_dta = DTA512(c4)
        self.dec1_post = conv_bn_relu(c4 * 2, c4, 3, 1, 1)

        self.up2 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.msf2 = MSF_Module(enc_ch=c3, dec_ch=c3, alpha=1.0)
        self.dec2_dta = DTA256(c3)
        self.dec2_post = conv_bn_relu(c3 * 2, c3, 3, 1, 1)

        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.msf3 = MSF_Module(enc_ch=c2, dec_ch=c2, alpha=1.0)

        self.up4 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.msf4 = MSF_Module(enc_ch=c1, dec_ch=c1, alpha=1.0)

        self.mar = MARBlock(c1=c3, c2=c2, c3=c1, d=c1, p_drop=0.1)

        self.reduce_after_mar = conv_bn_relu(c1 * 2, c1, 3, 1, 1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.down1(e1)

        e2 = self.enc2(p1)
        p2 = self.down2(e2)

        e3_raw = self.enc3(p2)
        e3_dta = self.enc3_dta(e3_raw)
        e3 = self.enc3_post(torch.cat([e3_raw, e3_dta], dim=1))
        p3 = self.down3(e3)

        e4_pre = self.enc4_pre(p3)
        e4_dta = self.enc4_dta(e4_pre)
        e4 = self.enc4_post(torch.cat([e4_pre, e4_dta], dim=1))
        p4 = self.down4(e4)

        e5_pre = self.enc5_pre(p4)
        e5_dta = self.enc5_dta(e5_pre)
        e5 = self.enc5_post(torch.cat([e5_pre, e5_dta], dim=1))

        d1_up = self.up1(e5)
        d1_dta = self.dec1_dta(d1_up)
        d1 = self.dec1_post(torch.cat([d1_up, d1_dta], dim=1))

        d2_up = self.up2(d1)
        d2_msf = self.msf2(e3, d2_up)
        d2_dta = self.dec2_dta(d2_msf)
        d2 = self.dec2_post(torch.cat([d2_msf, d2_dta], dim=1))

        d3_up = self.up3(d2)
        d3 = self.msf3(e2, d3_up)

        d4_up = self.up4(d3)
        d4 = self.msf4(e1, d4_up)

        c7 = d2
        c8 = d3
        c9 = d4

        mar_fused = self.mar(c7, c8, c9)
        out_feat = torch.cat([mar_fused, c9], dim=1)
        out_feat = self.reduce_after_mar(out_feat)

        out = self.head(out_feat)
        return out
