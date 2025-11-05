import torch
import torch.nn as nn

class GLTATT(nn.Module):
    def __init__(self, in_ch):
        super(GLTATT, self).__init__()

    def forward(self, input):

        input_B_D_hwLG = input

        input_B_hwLG_D = input_B_D_hwLG.contiguous().permute(0, 2, 1)
        input_B_hwLG_hwLG = torch.bmm(torch.relu(input_B_hwLG_D),torch.relu(input_B_D_hwLG))
        input_B_hwLG_hwLG_ = input_B_hwLG_hwLG/2
        d = torch.sum(input_B_hwLG_hwLG_,dim=1)
        d[d !=0]= torch.sqrt(1.0/d[d !=0])
        input_B_hwLG_hwLG_ *= d.unsqueeze(1)
        input_B_hwLG_hwLG_ *= d.unsqueeze(2)
        input_B_hwLG_D = torch.bmm(input_B_hwLG_hwLG_, input_B_hwLG_D)
        input_B_D_hwLG_out = input_B_hwLG_D.contiguous().permute(0, 2, 1)

        out = input_B_D_hwLG_out

        return out

class DTA256(nn.Module):
    def __init__(self, inplanes,planes):
        super(DTA256, self).__init__()

        self.q = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.k = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.v = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)

        #######conv1
        self.conv1_2ch_1ch = nn.Conv2d(planes+inplanes, planes, kernel_size=1, stride=1)
        # self.conv1_4ch_1ch= nn.Conv2d(planes*4, planes, kernel_size=1, stride=1)

        self.conv1_1ch_1ch = nn.Conv2d(inplanes,inplanes, kernel_size=1, stride=1)

        #######conv3
        self.convout = nn.Conv2d(planes, inplanes, kernel_size=3, stride=1,padding=1)

        #######bn
        self.bnout = nn.BatchNorm2d(inplanes)
        # self.bn2ch = nn.BatchNorm2d(planes * 2)
        self.bn = nn.BatchNorm2d(planes)

        self.linearh = nn.Linear(1, 32)
        self.linearH = nn.Linear(1, 64)
        self.avg_down = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.GLTATT = GLTATT(256)


    def forward(self,x):

        residual = x
        b, c, H, W = x.size()

        GT_B_D_1_1 = self.avg_down(x) #BD11

        x_ = torch.chunk(x, 2, dim=2)
        x1_ = x_[0]
        x2_ = x_[1]
        x1_12_ = torch.chunk(x1_, 2, dim=3)
        x11_ = x1_12_[0]
        x12_ = x1_12_[1]
        x2_12_ = torch.chunk(x2_, 2, dim=3)
        x21_ = x2_12_[0]
        x22_ = x2_12_[1]

        x11_b, x11_c, h, w = x11_.size()

        LT1_B_D_1_1 = self.avg_down(x11_)  #BD11
        LT2_B_D_1_1 = self.avg_down(x12_)
        LT3_B_D_1_1 = self.avg_down(x21_)
        LT4_B_D_1_1 = self.avg_down(x22_)

        GT_B_D_1 = GT_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT1_B_D_1 = LT1_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT2_B_D_1 = LT2_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT3_B_D_1 = LT3_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT4_B_D_1 = LT4_B_D_1_1.contiguous().view(x11_b, x11_c, -1)

        x11_B_D_hw = x11_.contiguous().view(x11_b, x11_c, -1)
        x12_B_D_hw = x12_.contiguous().view(x11_b, x11_c, -1)
        x21_B_D_hw = x21_.contiguous().view(x11_b, x11_c, -1)
        x22_B_D_hw = x22_.contiguous().view(x11_b, x11_c, -1)

        x11_B_D_hwLG = torch.cat([x11_B_D_hw, LT1_B_D_1, GT_B_D_1 ], dim=2)
        x12_B_D_hwLG = torch.cat([x12_B_D_hw, LT2_B_D_1, GT_B_D_1 ], dim=2)
        x21_B_D_hwLG = torch.cat([x21_B_D_hw, LT3_B_D_1, GT_B_D_1 ], dim=2)
        x22_B_D_hwLG = torch.cat([x22_B_D_hw, LT4_B_D_1, GT_B_D_1 ], dim=2)

        # x11_B_hwLG_D = x11_B_D_hwLG.contiguous().permute(0, 2, 1)
        # x12_B_hwLG_D = x12_B_D_hwLG.contiguous().permute(0, 2, 1)
        # x21_B_hwLG_D = x21_B_D_hwLG.contiguous().permute(0, 2, 1)
        # x22_B_hwLG_D = x22_B_D_hwLG.contiguous().permute(0, 2, 1)

############################att

        # x11_B_hwLG_hwLG = torch.bmm(torch.relu(x11_B_hwLG_D),torch.relu(x11_B_D_hwLG))
        # x11_B_hwLG_hwLG_ = x11_B_hwLG_hwLG/2
        # d11 = torch.sum(x11_B_hwLG_hwLG_,dim=1)
        # d11[d11 !=0]= torch.sqrt(1.0/d11[d11 !=0])
        # x11_B_hwLG_hwLG_ *= d11.unsqueeze(1)
        # x11_B_hwLG_hwLG_ *= d11.unsqueeze(2)
        # x11_B_hwLG_D = torch.bmm(x11_B_hwLG_hwLG_, x11_B_hwLG_D)
        # x11_B_D_hwLG_out = x11_B_hwLG_D.contiguous().permute(0, 2, 1)
        #
        #
        # x12_B_hwLG_hwLG = torch.bmm(torch.relu(x12_B_hwLG_D),torch.relu(x12_B_D_hwLG))
        # x12_B_hwLG_hwLG_ = x12_B_hwLG_hwLG/2
        # d12 = torch.sum(x12_B_hwLG_hwLG_,dim=1)
        # d12[d12 !=0]= torch.sqrt(1.0/d12[d12 !=0])
        # x12_B_hwLG_hwLG_ *= d12.unsqueeze(1)
        # x12_B_hwLG_hwLG_ *= d12.unsqueeze(2)
        # x12_B_hwLG_D = torch.bmm(x12_B_hwLG_hwLG_, x12_B_hwLG_D)
        # x12_B_D_hwLG_out = x12_B_hwLG_D.contiguous().permute(0, 2, 1)
        #
        # x21_B_hwLG_hwLG = torch.bmm(torch.relu(x21_B_hwLG_D),torch.relu(x21_B_D_hwLG))
        # x21_B_hwLG_hwLG_ = x21_B_hwLG_hwLG/2
        # d21 = torch.sum(x21_B_hwLG_hwLG_,dim=1)
        # d21[d21 !=0]= torch.sqrt(1.0/d21[d21 !=0])
        # x21_B_hwLG_hwLG_ *= d21.unsqueeze(1)
        # x21_B_hwLG_hwLG_ *= d21.unsqueeze(2)
        # x21_B_hwLG_D = torch.bmm(x21_B_hwLG_hwLG_, x21_B_hwLG_D)
        # x21_B_D_hwLG_out = x21_B_hwLG_D.contiguous().permute(0, 2, 1)
        #
        # x22_B_hwLG_hwLG = torch.bmm(torch.relu(x22_B_hwLG_D),torch.relu(x22_B_D_hwLG))
        # x22_B_hwLG_hwLG_ = x22_B_hwLG_hwLG/2
        # d22 = torch.sum(x22_B_hwLG_hwLG_,dim=1)
        # d22[d22 !=0]= torch.sqrt(1.0/d22[d22 !=0])
        # x22_B_hwLG_hwLG_ *= d22.unsqueeze(1)
        # x22_B_hwLG_hwLG_ *= d22.unsqueeze(2)
        # x22_B_hwLG_D = torch.bmm(x22_B_hwLG_hwLG_, x22_B_hwLG_D)
        # x22_B_D_hwLG_out = x22_B_hwLG_D.contiguous().permute(0, 2, 1)

        ######################### single_layer
        x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG)
        x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG)
        x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG)
        x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG)

        ######################### double_layer
        # x11_B_D_hwLG_out1 = self.GLTATT(x11_B_D_hwLG)
        # x12_B_D_hwLG_out1 = self.GLTATT(x12_B_D_hwLG)
        # x21_B_D_hwLG_out1 = self.GLTATT(x21_B_D_hwLG)
        # x22_B_D_hwLG_out1 = self.GLTATT(x22_B_D_hwLG)
        #
        # x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG_out1)
        # x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG_out1)
        # x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG_out1)
        # x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG_out1)

        ######################### 3_layer
        # x11_B_D_hwLG_out1 = self.GLTATT(x11_B_D_hwLG)
        # x12_B_D_hwLG_out1 = self.GLTATT(x12_B_D_hwLG)
        # x21_B_D_hwLG_out1 = self.GLTATT(x21_B_D_hwLG)
        # x22_B_D_hwLG_out1 = self.GLTATT(x22_B_D_hwLG)
        #
        # x11_B_D_hwLG_out2 = self.GLTATT(x11_B_D_hwLG_out1)
        # x12_B_D_hwLG_out2 = self.GLTATT(x12_B_D_hwLG_out1)
        # x21_B_D_hwLG_out2 = self.GLTATT(x21_B_D_hwLG_out1)
        # x22_B_D_hwLG_out2 = self.GLTATT(x22_B_D_hwLG_out1)
        #
        # x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG_out2)
        # x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG_out2)
        # x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG_out2)
        # x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG_out2)





        x11_out_B_D_hw,LT1_out_B_D_1,GT1_out_B_D_1 = torch.split(x11_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x12_out_B_D_hw,LT2_out_B_D_1,GT2_out_B_D_1 = torch.split(x12_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x21_out_B_D_hw,LT3_out_B_D_1,GT3_out_B_D_1 = torch.split(x21_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x22_out_B_D_hw,LT4_out_B_D_1,GT4_out_B_D_1 = torch.split(x22_B_D_hwLG_out, [h*w, 1, 1], dim=2)

        x11_out_B_D_h_w = x11_out_B_D_hw.view(b, c, h, w)
        x12_out_B_D_h_w = x12_out_B_D_hw.view(b, c, h, w)
        x21_out_B_D_h_w = x21_out_B_D_hw.view(b, c, h, w)
        x22_out_B_D_h_w = x22_out_B_D_hw.view(b, c, h, w)

        ########################## GT
        GT1_out_B_D_1_1 = GT1_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT2_out_B_D_1_1 = GT2_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT3_out_B_D_1_1 = GT3_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT4_out_B_D_1_1 = GT4_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)


        GT1_out_B_D_1_H = self.linearH(GT1_out_B_D_1_1)
        GT2_out_B_D_1_H = self.linearH(GT2_out_B_D_1_1)
        GT3_out_B_D_1_H = self.linearH(GT3_out_B_D_1_1)
        GT4_out_B_D_1_H = self.linearH(GT4_out_B_D_1_1)


        GT1234_out_B_D_4_H = torch.cat([GT1_out_B_D_1_H, GT2_out_B_D_1_H,GT3_out_B_D_1_H,GT4_out_B_D_1_H], dim=2)
        GT_out_B_D_4_H = self.conv1_1ch_1ch(GT1234_out_B_D_4_H)
        # ########################## GT
        # GT1_out_B_D_1_1 = GT1_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        # GT1_out_B_D_1_W = self.linearH(GT1_out_B_D_1_1)
        # GT1_out_B_D_1_H = self.linearH(GT1_out_B_D_1_1)
        # GT1_out_B_D_H_1 = GT1_out_B_D_1_H.view(b,c,H,1)
        # GT1_out_B_D_H_W = GT1_out_B_D_H_1 * GT1_out_B_D_1_W
        # #GT1_out_B_D_H_W_sigmoid = self.sigmoid(GT1_out_B_D_H_W)
        #
        # GT2_out_B_D_1_1 = GT2_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        # GT2_out_B_D_1_W = self.linearH(GT2_out_B_D_1_1)
        # GT2_out_B_D_1_H = self.linearH(GT2_out_B_D_1_1)
        # GT2_out_B_D_H_1 = GT2_out_B_D_1_H.view(b,c,H,1)
        # GT2_out_B_D_H_W = GT2_out_B_D_H_1 * GT2_out_B_D_1_W
        # #GT2_out_B_D_H_W_sigmoid = self.sigmoid(GT2_out_B_D_H_W)
        #
        # GT3_out_B_D_1_1 = GT3_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        # GT3_out_B_D_1_W = self.linearH(GT3_out_B_D_1_1)
        # GT3_out_B_D_1_H = self.linearH(GT3_out_B_D_1_1)
        # GT3_out_B_D_H_1 = GT3_out_B_D_1_H.view(b,c,H,1)
        # GT3_out_B_D_H_W = GT3_out_B_D_H_1 * GT3_out_B_D_1_W
        # #GT3_out_B_D_H_W_sigmoid = self.sigmoid(GT3_out_B_D_H_W)
        #
        # GT4_out_B_D_1_1 = GT4_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        # GT4_out_B_D_1_W = self.linearH(GT4_out_B_D_1_1)
        # GT4_out_B_D_1_H = self.linearH(GT4_out_B_D_1_1)
        # GT4_out_B_D_H_1 = GT4_out_B_D_1_H.view(b,c,H,1)
        # GT4_out_B_D_H_W = GT4_out_B_D_H_1 * GT4_out_B_D_1_W
        # #GT4_out_B_D_H_W_sigmoid = self.sigmoid(GT4_out_B_D_H_W)
        #
        # GT1234_out_B_D_H_W = torch.cat([GT1_out_B_D_H_W, GT2_out_B_D_H_W,GT3_out_B_D_H_W,GT4_out_B_D_H_W], dim=1)
        # GT_out_B_D_H_W = self.conv1_4ch_1ch(GT1234_out_B_D_H_W)
        # GT_out_B_D_H_W_sigmoid = self.sigmoid(GT_out_B_D_H_W)

        ########################## LT
        LT1_out_B_D_1_1 = LT1_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        LT1_out_B_D_1_w = self.linearh(LT1_out_B_D_1_1)
        LT1_out_B_D_1_h = self.linearh(LT1_out_B_D_1_1)
        LT1_out_B_D_h_1 = LT1_out_B_D_1_h.view(b,c,h,1)
        LT1_out_B_D_h_w = LT1_out_B_D_h_1 * LT1_out_B_D_1_w
        LT1_out_B_D_h_w_sigmoid = self.sigmoid(LT1_out_B_D_h_w)

        LT2_out_B_D_1_1 = LT2_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        LT2_out_B_D_1_w = self.linearh(LT2_out_B_D_1_1)
        LT2_out_B_D_1_h = self.linearh(LT2_out_B_D_1_1)
        LT2_out_B_D_h_1 = LT2_out_B_D_1_h.view(b,c,h,1)
        LT2_out_B_D_h_w = LT2_out_B_D_h_1 * LT2_out_B_D_1_w
        LT2_out_B_D_h_w_sigmoid = self.sigmoid(LT2_out_B_D_h_w)

        LT3_out_B_D_1_1 = LT3_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        LT3_out_B_D_1_w = self.linearh(LT3_out_B_D_1_1)
        LT3_out_B_D_1_h = self.linearh(LT3_out_B_D_1_1)
        LT3_out_B_D_h_1 = LT3_out_B_D_1_h.view(b,c,h,1)
        LT3_out_B_D_h_w = LT3_out_B_D_h_1 * LT3_out_B_D_1_w
        LT3_out_B_D_h_w_sigmoid = self.sigmoid(LT3_out_B_D_h_w)

        LT4_out_B_D_1_1 = LT4_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        LT4_out_B_D_1_w = self.linearh(LT4_out_B_D_1_1)
        LT4_out_B_D_1_h = self.linearh(LT4_out_B_D_1_1)
        LT4_out_B_D_h_1 = LT4_out_B_D_1_h.view(b,c,h,1)
        LT4_out_B_D_h_w = LT4_out_B_D_h_1 * LT4_out_B_D_1_w
        LT4_out_B_D_h_w_sigmoid = self.sigmoid(LT4_out_B_D_h_w)

        x11_LT1_B_D_h_w = x11_out_B_D_h_w * LT1_out_B_D_h_w_sigmoid
        x12_LT2_B_D_h_w = x12_out_B_D_h_w * LT2_out_B_D_h_w_sigmoid
        x21_LT3_B_D_h_w = x21_out_B_D_h_w * LT3_out_B_D_h_w_sigmoid
        x22_LT4_B_D_h_w = x22_out_B_D_h_w * LT4_out_B_D_h_w_sigmoid

        x1_out = torch.cat([x11_LT1_B_D_h_w,x12_LT2_B_D_h_w], dim=3)
        x2_out = torch.cat([x21_LT3_B_D_h_w,x22_LT4_B_D_h_w], dim=3)
        x_LT_out = torch.cat([x1_out, x2_out ], dim=2)

        x_LT_out_B_D_HW = x_LT_out.contiguous().view(b, c, -1)
        GT_out_B_D_4H = GT_out_B_D_4_H.contiguous().view(b, c, -1)
        GT_out_B_4H_D = GT_out_B_D_4H.contiguous().permute(0, 2, 1)
        GT_B_4H_HW = torch.bmm(torch.relu(GT_out_B_4H_D), torch.relu(x_LT_out_B_D_HW))
        GT_B_4H_HW_ = GT_B_4H_HW / 2
        d = torch.sum(GT_B_4H_HW_, dim=1)
        d[d != 0] = torch.sqrt(1.0 / d[d != 0])
        GT_B_4H_HW_ *= d.unsqueeze(1)
        GT_out_B_D_HW = torch.bmm(GT_out_B_D_4H, GT_B_4H_HW_)

        GT_out_B_D_H_W = GT_out_B_D_HW.contiguous().view(b, c, H, W)

        x_LT_GT_out = x_LT_out + GT_out_B_D_H_W

        x_cat = torch.cat([residual,x_LT_GT_out], dim=1)
        x_out = self.conv1_2ch_1ch(x_cat)
        x_out = self.bn(x_out)
        x_out = torch.relu(x_out)
        x_out = self.convout(x_out)
        x_out = self.bnout(x_out)
        x_out = torch.relu(x_out)

        out = x_out


        return out


class DTA512(nn.Module):
    def __init__(self, inplanes,planes):
        super(DTA512, self).__init__()
        self.q = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.k = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.v = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)

        #######conv1
        self.conv1_2ch_1ch = nn.Conv2d(planes + inplanes, planes, kernel_size=1, stride=1)
        # self.conv1_4ch_1ch = nn.Conv2d(planes * 4, planes, kernel_size=1, stride=1)
        self.conv1_1ch_1ch = nn.Conv2d(inplanes,inplanes, kernel_size=1, stride=1)
        #######conv3
        self.convout = nn.Conv2d(planes, inplanes, kernel_size=3, stride=1, padding=1)

        #######bn
        self.bnout = nn.BatchNorm2d(inplanes)
        # self.bn2ch = nn.BatchNorm2d(planes * 2)
        self.bn = nn.BatchNorm2d(planes)

        self.linearh = nn.Linear(1, 16)
        self.linearH = nn.Linear(1, 32)
        self.avg_down = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.GLTATT = GLTATT(512)

    def forward(self,x):
        residual = x
        b, c, H, W = x.size()

        GT_B_D_1_1 = self.avg_down(x)  # BD11

        x_ = torch.chunk(x, 2, dim=2)
        x1_ = x_[0]
        x2_ = x_[1]
        x1_12_ = torch.chunk(x1_, 2, dim=3)
        x11_ = x1_12_[0]
        x12_ = x1_12_[1]
        x2_12_ = torch.chunk(x2_, 2, dim=3)
        x21_ = x2_12_[0]
        x22_ = x2_12_[1]

        x11_b, x11_c, h, w = x11_.size()

        LT1_B_D_1_1 = self.avg_down(x11_)  # BD11
        LT2_B_D_1_1 = self.avg_down(x12_)
        LT3_B_D_1_1 = self.avg_down(x21_)
        LT4_B_D_1_1 = self.avg_down(x22_)

        GT_B_D_1 = GT_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT1_B_D_1 = LT1_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT2_B_D_1 = LT2_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT3_B_D_1 = LT3_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT4_B_D_1 = LT4_B_D_1_1.contiguous().view(x11_b, x11_c, -1)

        x11_B_D_hw = x11_.contiguous().view(x11_b, x11_c, -1)
        x12_B_D_hw = x12_.contiguous().view(x11_b, x11_c, -1)
        x21_B_D_hw = x21_.contiguous().view(x11_b, x11_c, -1)
        x22_B_D_hw = x22_.contiguous().view(x11_b, x11_c, -1)

        x11_B_D_hwLG = torch.cat([x11_B_D_hw, LT1_B_D_1, GT_B_D_1], dim=2)
        x12_B_D_hwLG = torch.cat([x12_B_D_hw, LT2_B_D_1, GT_B_D_1], dim=2)
        x21_B_D_hwLG = torch.cat([x21_B_D_hw, LT3_B_D_1, GT_B_D_1], dim=2)
        x22_B_D_hwLG = torch.cat([x22_B_D_hw, LT4_B_D_1, GT_B_D_1], dim=2)


        ######################### single_layer
        x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG)
        x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG)
        x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG)
        x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG)

        ######################### double_layer
        # x11_B_D_hwLG_out1 = self.GLTATT(x11_B_D_hwLG)
        # x12_B_D_hwLG_out1 = self.GLTATT(x12_B_D_hwLG)
        # x21_B_D_hwLG_out1 = self.GLTATT(x21_B_D_hwLG)
        # x22_B_D_hwLG_out1 = self.GLTATT(x22_B_D_hwLG)
        #
        # x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG_out1)
        # x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG_out1)
        # x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG_out1)
        # x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG_out1)

        ######################### 3_layer
        # x11_B_D_hwLG_out1 = self.GLTATT(x11_B_D_hwLG)
        # x12_B_D_hwLG_out1 = self.GLTATT(x12_B_D_hwLG)
        # x21_B_D_hwLG_out1 = self.GLTATT(x21_B_D_hwLG)
        # x22_B_D_hwLG_out1 = self.GLTATT(x22_B_D_hwLG)
        #
        # x11_B_D_hwLG_out2 = self.GLTATT(x11_B_D_hwLG_out1)
        # x12_B_D_hwLG_out2 = self.GLTATT(x12_B_D_hwLG_out1)
        # x21_B_D_hwLG_out2 = self.GLTATT(x21_B_D_hwLG_out1)
        # x22_B_D_hwLG_out2 = self.GLTATT(x22_B_D_hwLG_out1)
        #
        # x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG_out2)
        # x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG_out2)
        # x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG_out2)
        # x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG_out2)






        x11_out_B_D_hw, LT1_out_B_D_1, GT1_out_B_D_1 = torch.split(x11_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x12_out_B_D_hw, LT2_out_B_D_1, GT2_out_B_D_1 = torch.split(x12_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x21_out_B_D_hw, LT3_out_B_D_1, GT3_out_B_D_1 = torch.split(x21_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x22_out_B_D_hw, LT4_out_B_D_1, GT4_out_B_D_1 = torch.split(x22_B_D_hwLG_out, [h*w, 1, 1], dim=2)

        x11_out_B_D_h_w = x11_out_B_D_hw.view(b, c, h, w)
        x12_out_B_D_h_w = x12_out_B_D_hw.view(b, c, h, w)
        x21_out_B_D_h_w = x21_out_B_D_hw.view(b, c, h, w)
        x22_out_B_D_h_w = x22_out_B_D_hw.view(b, c, h, w)



        ########################## GT
        GT1_out_B_D_1_1 = GT1_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT2_out_B_D_1_1 = GT2_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT3_out_B_D_1_1 = GT3_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT4_out_B_D_1_1 = GT4_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)


        GT1_out_B_D_1_H = self.linearH(GT1_out_B_D_1_1)
        GT2_out_B_D_1_H = self.linearH(GT2_out_B_D_1_1)
        GT3_out_B_D_1_H = self.linearH(GT3_out_B_D_1_1)
        GT4_out_B_D_1_H = self.linearH(GT4_out_B_D_1_1)


        GT1234_out_B_D_4_H = torch.cat([GT1_out_B_D_1_H, GT2_out_B_D_1_H,GT3_out_B_D_1_H,GT4_out_B_D_1_H], dim=2)
        GT_out_B_D_4_H = self.conv1_1ch_1ch(GT1234_out_B_D_4_H)


        ########################## LT
        LT1_out_B_D_1_1 = LT1_out_B_D_1.contiguous().view(x11_b, x11_c, 1, 1)
        LT1_out_B_D_1_w = self.linearh(LT1_out_B_D_1_1)
        LT1_out_B_D_1_h = self.linearh(LT1_out_B_D_1_1)
        LT1_out_B_D_h_1 = LT1_out_B_D_1_h.view(b, c, h, 1)
        LT1_out_B_D_h_w = LT1_out_B_D_h_1 * LT1_out_B_D_1_w
        LT1_out_B_D_h_w_sigmoid = self.sigmoid(LT1_out_B_D_h_w)

        LT2_out_B_D_1_1 = LT2_out_B_D_1.contiguous().view(x11_b, x11_c, 1, 1)
        LT2_out_B_D_1_w = self.linearh(LT2_out_B_D_1_1)
        LT2_out_B_D_1_h = self.linearh(LT2_out_B_D_1_1)
        LT2_out_B_D_h_1 = LT2_out_B_D_1_h.view(b, c, h, 1)
        LT2_out_B_D_h_w = LT2_out_B_D_h_1 * LT2_out_B_D_1_w
        LT2_out_B_D_h_w_sigmoid = self.sigmoid(LT2_out_B_D_h_w)

        LT3_out_B_D_1_1 = LT3_out_B_D_1.contiguous().view(x11_b, x11_c, 1, 1)
        LT3_out_B_D_1_w = self.linearh(LT3_out_B_D_1_1)
        LT3_out_B_D_1_h = self.linearh(LT3_out_B_D_1_1)
        LT3_out_B_D_h_1 = LT3_out_B_D_1_h.view(b, c, h, 1)
        LT3_out_B_D_h_w = LT3_out_B_D_h_1 * LT3_out_B_D_1_w
        LT3_out_B_D_h_w_sigmoid = self.sigmoid(LT3_out_B_D_h_w)

        LT4_out_B_D_1_1 = LT4_out_B_D_1.contiguous().view(x11_b, x11_c, 1, 1)
        LT4_out_B_D_1_w = self.linearh(LT4_out_B_D_1_1)
        LT4_out_B_D_1_h = self.linearh(LT4_out_B_D_1_1)
        LT4_out_B_D_h_1 = LT4_out_B_D_1_h.view(b, c, h, 1)
        LT4_out_B_D_h_w = LT4_out_B_D_h_1 * LT4_out_B_D_1_w
        LT4_out_B_D_h_w_sigmoid = self.sigmoid(LT4_out_B_D_h_w)

        x11_LT1_B_D_h_w = x11_out_B_D_h_w * LT1_out_B_D_h_w_sigmoid
        x12_LT2_B_D_h_w = x12_out_B_D_h_w * LT2_out_B_D_h_w_sigmoid
        x21_LT3_B_D_h_w = x21_out_B_D_h_w * LT3_out_B_D_h_w_sigmoid
        x22_LT4_B_D_h_w = x22_out_B_D_h_w * LT4_out_B_D_h_w_sigmoid

        x1_out = torch.cat([x11_LT1_B_D_h_w, x12_LT2_B_D_h_w], dim=3)
        x2_out = torch.cat([x21_LT3_B_D_h_w, x22_LT4_B_D_h_w], dim=3)
        x_LT_out = torch.cat([x1_out, x2_out], dim=2)

        x_LT_out_B_D_HW = x_LT_out.contiguous().view(b, c, -1)
        GT_out_B_D_4H = GT_out_B_D_4_H.contiguous().view(b, c, -1)
        GT_out_B_4H_D = GT_out_B_D_4H.contiguous().permute(0, 2, 1)
        GT_B_4H_HW = torch.bmm(torch.relu(GT_out_B_4H_D), torch.relu(x_LT_out_B_D_HW))
        GT_B_4H_HW_ = GT_B_4H_HW / 2
        d = torch.sum(GT_B_4H_HW_, dim=1)
        d[d != 0] = torch.sqrt(1.0 / d[d != 0])
        GT_B_4H_HW_ *= d.unsqueeze(1)
        GT_out_B_D_HW = torch.bmm(GT_out_B_D_4H, GT_B_4H_HW_)

        GT_out_B_D_H_W = GT_out_B_D_HW.contiguous().view(b, c, H, W)

        x_LT_GT_out = x_LT_out + GT_out_B_D_H_W

        x_cat = torch.cat([residual, x_LT_GT_out], dim=1)
        x_out = self.conv1_2ch_1ch(x_cat)
        x_out = self.bn(x_out)
        x_out = torch.relu(x_out)
        x_out = self.convout(x_out)
        x_out = self.bnout(x_out)
        x_out = torch.relu(x_out)

        out = x_out

        return out


class DTA1024(nn.Module):
    def __init__(self, inplanes,planes):
        super(DTA1024, self).__init__()
        self.q = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.k = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.v = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)

        #######conv1
        self.conv1_2ch_1ch = nn.Conv2d(planes+inplanes, planes, kernel_size=1, stride=1)
        # self.conv1_4ch_1ch= nn.Conv2d(planes*4, planes, kernel_size=1, stride=1)
        self.conv1_1ch_1ch = nn.Conv2d(inplanes,inplanes, kernel_size=1, stride=1)
        #######conv3
        self.convout = nn.Conv2d(planes, inplanes, kernel_size=3, stride=1,padding=1)

        #######bn
        self.bnout = nn.BatchNorm2d(inplanes)
        # self.bn2ch = nn.BatchNorm2d(planes * 2)
        self.bn = nn.BatchNorm2d(planes)

        self.linearh = nn.Linear(1, 8)
        self.linearH = nn.Linear(1, 16)
        self.avg_down = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.GLTATT = GLTATT(1024)

    def forward(self,x):
        residual = x
        b, c, H, W = x.size()

        GT_B_D_1_1 = self.avg_down(x)  # BD11

        x_ = torch.chunk(x, 2, dim=2)
        x1_ = x_[0]
        x2_ = x_[1]
        x1_12_ = torch.chunk(x1_, 2, dim=3)
        x11_ = x1_12_[0]
        x12_ = x1_12_[1]
        x2_12_ = torch.chunk(x2_, 2, dim=3)
        x21_ = x2_12_[0]
        x22_ = x2_12_[1]

        x11_b, x11_c, h, w = x11_.size()

        LT1_B_D_1_1 = self.avg_down(x11_)  # BD11
        LT2_B_D_1_1 = self.avg_down(x12_)
        LT3_B_D_1_1 = self.avg_down(x21_)
        LT4_B_D_1_1 = self.avg_down(x22_)

        GT_B_D_1 = GT_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT1_B_D_1 = LT1_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT2_B_D_1 = LT2_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT3_B_D_1 = LT3_B_D_1_1.contiguous().view(x11_b, x11_c, -1)
        LT4_B_D_1 = LT4_B_D_1_1.contiguous().view(x11_b, x11_c, -1)

        x11_B_D_hw = x11_.contiguous().view(x11_b, x11_c, -1)
        x12_B_D_hw = x12_.contiguous().view(x11_b, x11_c, -1)
        x21_B_D_hw = x21_.contiguous().view(x11_b, x11_c, -1)
        x22_B_D_hw = x22_.contiguous().view(x11_b, x11_c, -1)

        x11_B_D_hwLG = torch.cat([x11_B_D_hw, LT1_B_D_1, GT_B_D_1], dim=2)
        x12_B_D_hwLG = torch.cat([x12_B_D_hw, LT2_B_D_1, GT_B_D_1], dim=2)
        x21_B_D_hwLG = torch.cat([x21_B_D_hw, LT3_B_D_1, GT_B_D_1], dim=2)
        x22_B_D_hwLG = torch.cat([x22_B_D_hw, LT4_B_D_1, GT_B_D_1], dim=2)


        ######################### single_layer
        x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG)
        x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG)
        x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG)
        x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG)

        ######################### double_layer
        # x11_B_D_hwLG_out1 = self.GLTATT(x11_B_D_hwLG)
        # x12_B_D_hwLG_out1 = self.GLTATT(x12_B_D_hwLG)
        # x21_B_D_hwLG_out1 = self.GLTATT(x21_B_D_hwLG)
        # x22_B_D_hwLG_out1 = self.GLTATT(x22_B_D_hwLG)
        #
        # x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG_out1)
        # x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG_out1)
        # x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG_out1)
        # x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG_out1)

        ######################### 3_layer
        # x11_B_D_hwLG_out1 = self.GLTATT(x11_B_D_hwLG)
        # x12_B_D_hwLG_out1 = self.GLTATT(x12_B_D_hwLG)
        # x21_B_D_hwLG_out1 = self.GLTATT(x21_B_D_hwLG)
        # x22_B_D_hwLG_out1 = self.GLTATT(x22_B_D_hwLG)
        #
        # x11_B_D_hwLG_out2 = self.GLTATT(x11_B_D_hwLG_out1)
        # x12_B_D_hwLG_out2 = self.GLTATT(x12_B_D_hwLG_out1)
        # x21_B_D_hwLG_out2 = self.GLTATT(x21_B_D_hwLG_out1)
        # x22_B_D_hwLG_out2 = self.GLTATT(x22_B_D_hwLG_out1)
        #
        # x11_B_D_hwLG_out = self.GLTATT(x11_B_D_hwLG_out2)
        # x12_B_D_hwLG_out = self.GLTATT(x12_B_D_hwLG_out2)
        # x21_B_D_hwLG_out = self.GLTATT(x21_B_D_hwLG_out2)
        # x22_B_D_hwLG_out = self.GLTATT(x22_B_D_hwLG_out2)



        x11_out_B_D_hw, LT1_out_B_D_1, GT1_out_B_D_1 = torch.split(x11_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x12_out_B_D_hw, LT2_out_B_D_1, GT2_out_B_D_1 = torch.split(x12_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x21_out_B_D_hw, LT3_out_B_D_1, GT3_out_B_D_1 = torch.split(x21_B_D_hwLG_out, [h*w, 1, 1], dim=2)
        x22_out_B_D_hw, LT4_out_B_D_1, GT4_out_B_D_1 = torch.split(x22_B_D_hwLG_out, [h*w, 1, 1], dim=2)

        x11_out_B_D_h_w = x11_out_B_D_hw.view(b, c, h, w)
        x12_out_B_D_h_w = x12_out_B_D_hw.view(b, c, h, w)
        x21_out_B_D_h_w = x21_out_B_D_hw.view(b, c, h, w)
        x22_out_B_D_h_w = x22_out_B_D_hw.view(b, c, h, w)

        ########################## GT

        GT1_out_B_D_1_1 = GT1_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT2_out_B_D_1_1 = GT2_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT3_out_B_D_1_1 = GT3_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)
        GT4_out_B_D_1_1 = GT4_out_B_D_1.contiguous().view(x11_b, x11_c, 1,1)

        GT1_out_B_D_1_H = self.linearH(GT1_out_B_D_1_1)
        GT2_out_B_D_1_H = self.linearH(GT2_out_B_D_1_1)
        GT3_out_B_D_1_H = self.linearH(GT3_out_B_D_1_1)
        GT4_out_B_D_1_H = self.linearH(GT4_out_B_D_1_1)

        GT1234_out_B_D_4_H = torch.cat([GT1_out_B_D_1_H, GT2_out_B_D_1_H,GT3_out_B_D_1_H,GT4_out_B_D_1_H], dim=2)
        GT_out_B_D_4_H = self.conv1_1ch_1ch(GT1234_out_B_D_4_H)


        ########################## LT
        LT1_out_B_D_1_1 = LT1_out_B_D_1.contiguous().view(x11_b, x11_c, 1, 1)
        LT1_out_B_D_1_w = self.linearh(LT1_out_B_D_1_1)
        LT1_out_B_D_1_h = self.linearh(LT1_out_B_D_1_1)
        LT1_out_B_D_h_1 = LT1_out_B_D_1_h.view(b, c, h, 1)
        LT1_out_B_D_h_w = LT1_out_B_D_h_1 * LT1_out_B_D_1_w
        LT1_out_B_D_h_w_sigmoid = self.sigmoid(LT1_out_B_D_h_w)

        LT2_out_B_D_1_1 = LT2_out_B_D_1.contiguous().view(x11_b, x11_c, 1, 1)
        LT2_out_B_D_1_w = self.linearh(LT2_out_B_D_1_1)
        LT2_out_B_D_1_h = self.linearh(LT2_out_B_D_1_1)
        LT2_out_B_D_h_1 = LT2_out_B_D_1_h.view(b, c, h, 1)
        LT2_out_B_D_h_w = LT2_out_B_D_h_1 * LT2_out_B_D_1_w
        LT2_out_B_D_h_w_sigmoid = self.sigmoid(LT2_out_B_D_h_w)

        LT3_out_B_D_1_1 = LT3_out_B_D_1.contiguous().view(x11_b, x11_c, 1, 1)
        LT3_out_B_D_1_w = self.linearh(LT3_out_B_D_1_1)
        LT3_out_B_D_1_h = self.linearh(LT3_out_B_D_1_1)
        LT3_out_B_D_h_1 = LT3_out_B_D_1_h.view(b, c, h, 1)
        LT3_out_B_D_h_w = LT3_out_B_D_h_1 * LT3_out_B_D_1_w
        LT3_out_B_D_h_w_sigmoid = self.sigmoid(LT3_out_B_D_h_w)

        LT4_out_B_D_1_1 = LT4_out_B_D_1.contiguous().view(x11_b, x11_c, 1, 1)
        LT4_out_B_D_1_w = self.linearh(LT4_out_B_D_1_1)
        LT4_out_B_D_1_h = self.linearh(LT4_out_B_D_1_1)
        LT4_out_B_D_h_1 = LT4_out_B_D_1_h.view(b, c, h, 1)
        LT4_out_B_D_h_w = LT4_out_B_D_h_1 * LT4_out_B_D_1_w
        LT4_out_B_D_h_w_sigmoid = self.sigmoid(LT4_out_B_D_h_w)

        x11_LT1_B_D_h_w = x11_out_B_D_h_w * LT1_out_B_D_h_w_sigmoid
        x12_LT2_B_D_h_w = x12_out_B_D_h_w * LT2_out_B_D_h_w_sigmoid
        x21_LT3_B_D_h_w = x21_out_B_D_h_w * LT3_out_B_D_h_w_sigmoid
        x22_LT4_B_D_h_w = x22_out_B_D_h_w * LT4_out_B_D_h_w_sigmoid

        x1_out = torch.cat([x11_LT1_B_D_h_w, x12_LT2_B_D_h_w], dim=3)
        x2_out = torch.cat([x21_LT3_B_D_h_w, x22_LT4_B_D_h_w], dim=3)
        x_LT_out = torch.cat([x1_out, x2_out], dim=2)

        x_LT_out_B_D_HW = x_LT_out.contiguous().view(b, c, -1)
        GT_out_B_D_4H = GT_out_B_D_4_H.contiguous().view(b, c, -1)
        GT_out_B_4H_D = GT_out_B_D_4H.contiguous().permute(0, 2, 1)
        GT_B_4H_HW = torch.bmm(torch.relu(GT_out_B_4H_D),torch.relu(x_LT_out_B_D_HW))
        GT_B_4H_HW_ = GT_B_4H_HW/2
        d = torch.sum(GT_B_4H_HW_,dim=1)
        d[d !=0]= torch.sqrt(1.0/d[d !=0])
        GT_B_4H_HW_ *= d.unsqueeze(1)
        GT_out_B_D_HW = torch.bmm(GT_out_B_D_4H, GT_B_4H_HW_)


        GT_out_B_D_H_W = GT_out_B_D_HW.contiguous().view(b, c, H,W)



        x_LT_GT_out = x_LT_out + GT_out_B_D_H_W

        x_cat = torch.cat([residual, x_LT_GT_out], dim=1)
        x_out = self.conv1_2ch_1ch(x_cat)
        x_out = self.bn(x_out)
        x_out = torch.relu(x_out)
        x_out = self.convout(x_out)
        x_out = self.bnout(x_out)
        x_out = torch.relu(x_out)

        out = x_out

        return out