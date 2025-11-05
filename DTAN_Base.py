import torch.nn as nn
import torch

from .DTA import DTA1024,DTA512,DTA256


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
        super(PreConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        out = self.conv(input)
        return out


class DTAN_Base(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DTAN_Base, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.conv0 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.conv3_pre = PreConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.conv4_pre = PreConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.conv5_pre = PreConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.conv6_pre = PreConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.conv7_pre = PreConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)



        self.conv_in3_out64 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv_in64_out64 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_in64_out128 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_in128_out128 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv_in128_out256 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv_in256_out256 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_in256_out512 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv_in512_out512 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_in512_out1024 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv_in1024_out1024 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv_in2048_out1024 =  nn.Conv2d(2048, 1024, 3, padding=1)
        self.conv_in1024_out512 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv_in512_out256 = nn.Conv2d(512, 256, 3, padding=1)
        # self.conv_in256_out128 = nn.Conv2d(256, 128, 3, padding=1)
        # self.conv_in128_out64 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv_in64_out1 = nn.Conv2d(64, 1, 3, padding=1)

        self.conv_in256_out64 = nn.Conv2d(256, 64, 3, padding=1)
        self.conv_in128_out64 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv_in192_out64 = nn.Conv2d(192, 64, 3, padding=1)
        self.up_64_128 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.up_128_256 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up_64_128 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        # self.BN_64 = nn.BatchNorm2d(64)
        self.BN_128 = nn.BatchNorm2d(128)
        self.BN_256 = nn.BatchNorm2d(256)
        self.BN_512 = nn.BatchNorm2d(512)
        self.BN_1024 = nn.BatchNorm2d(1024)

        # self.ReLU_64 = nn.ReLU(inplace=True)
        self.ReLU_128 = nn.ReLU(inplace=True)
        self.ReLU_256 = nn.ReLU(inplace=True)
        self.ReLU_512 = nn.ReLU(inplace=True)
        self.ReLU_1024 = nn.ReLU(inplace=True)



        # # d=64
        self.DTA_1024=DTA1024(1024,1024)
        self.DTA_512=DTA512(512,512)
        self.DTA_256=DTA256(256,256)


        self.conv_down128 = nn.Conv2d(32,32, 2,stride=2)
        self.conv_down64 = nn.Conv2d(64,64, 2,stride=2)
        self.conv_down32 = nn.Conv2d(128,128, 2,stride=2)
        self.conv_down16 = nn.Conv2d(256,256, 2,stride=2)

        self.avg_down128 = nn.AdaptiveAvgPool2d((128, 128))
        self.avg_down64 = nn.AdaptiveAvgPool2d((64, 64))
        self.avg_down32 = nn.AdaptiveAvgPool2d((32, 32))
        self.avg_down16 = nn.AdaptiveAvgPool2d((16, 16))

        self.avg_down1 = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

        self.conv_in1280_out1024 = nn.Conv2d(1280, 1024, 3, padding=1)
        self.conv_in640_out512 =  nn.Conv2d(640,512, 3, padding=1)
        self.conv_in320_out256 = nn.Conv2d(320,256, 3, padding=1)
        self.conv_in160_out128 =  nn.Conv2d(160,128, 3, padding=1)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout25 = nn.Dropout(0.25)


    def forward(self, x):

        c1 = self.conv1(x)

        ############################################c1_multi

        c1_B_128_256_256 = self.conv_in64_out128(c1)
        c1_B_128_128_128 = self.avg_down128(c1_B_128_256_256)
        c1_B_128_128_128 = c1_B_128_128_128 * 0.3
        c1_B_128_128_128 = self.dropout10(c1_B_128_128_128)

        ############################################

        p1 = self.pool1(c1)
        c2 = self.conv2(p1)

        ############################################c2_multi

        c2_B_256_128_128 = self.conv_in128_out256(c2)
        c2_B_256_64_64 = self.avg_down64(c2_B_256_128_128)
        c2_B_256_64_64 = c2_B_256_64_64 * 0.3
        c2_B_256_64_64 = self.dropout10(c2_B_256_64_64)

        ############################################

        p2 = self.pool2(c2)
        c3 = self.conv3_pre(p2)

        ##############################################GLTc3
        c3_res = c3

        c3GLT= self.DTA_256(c3)

        c3all = torch.cat([c3_res, c3GLT], dim=1)
        c3 = self.conv_in512_out256(c3all)
        c3 = self.BN_256(c3)
        c3 = self.ReLU_256(c3)
        ##############################################c3_multi

        c3_B_512_64_64 = self.conv_in256_out512(c3)
        c3_B_512_32_32 = self.avg_down32(c3_B_512_64_64)
        c3_B_512_32_32 = c3_B_512_32_32 * 0.3
        c3_B_512_32_32 = self.dropout10(c3_B_512_32_32)

        ############################################
        p3 = self.pool3(c3)

        c4 = self.conv4_pre(p3)
        ##############################################GLTc4

        c4_res = c4

        c4GLT = self.DTA_512(c4)


        c4all = torch.cat([c4_res, c4GLT], dim=1)
        c4 = self.conv_in1024_out512(c4all)
        c4 = self.BN_512(c4)
        c4 = self.ReLU_512(c4)

        ##############################################c4_multi
        c4_B_1024_32_32 = self.conv_in512_out1024(c4)
        c4_B_1024_16_16 = self.avg_down16(c4_B_1024_32_32)
        c4_B_1024_16_16 = c4_B_1024_16_16 * 0.3
        c4_B_1024_16_16 = self.dropout10(c4_B_1024_16_16)

        ##############################################
        p4 = self.pool4(c4)
        c5 = self.conv5_pre(p4)
        ##############################################GLTc5

        c5_res = c5

        c5GLT = self.DTA_1024(c5)

        c5all = torch.cat([c5_res, c5GLT], dim=1)
        c5 = self.conv_in2048_out1024(c5all)
        c5 = self.BN_1024(c5)
        c5 = self.ReLU_1024(c5)

        ############################################## c5-c4

        c5_c4 = c5 * c4_B_1024_16_16
        c5_ = c5 + c5_c4
        c5 = self.conv_in1024_out1024(c5_)

        ##############################################
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6_pre(merge6)
        # ##############################################GLTc6
        c6_res = c6

        c6GLT = self.DTA_512(c6)


        c6all = torch.cat([c6_res, c6GLT], dim=1)
        c6 = self.conv_in1024_out512(c6all)
        c6 = self.BN_512(c6)
        c6 = self.ReLU_512(c6)

        ############################################## c6-c3

        c6_c3 = c6 * c3_B_512_32_32
        c6_ = c6 + c6_c3
        c6 = self.conv_in512_out512(c6_)

        ##############################################

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7_pre(merge7)
        ##############################################GLTc7

        c7_res = c7

        c7GLT = self.DTA_256(c7)


        c7all = torch.cat([c7_res, c7GLT], dim=1)
        c7 = self.conv_in512_out256(c7all)
        c7 = self.BN_256(c7)
        c7 = self.ReLU_256(c7)

        ############################################## c7_c2

        c7_c2 = c7 * c2_B_256_64_64
        c7_ = c7 + c7_c2
        c7 = self.conv_in256_out256(c7_)




        ##############################################
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        ############################################## c8_c1

        c8_c1 = c8 * c1_B_128_128_128
        c8_ = c8 + c8_c1
        c8 = self.conv_in128_out128(c8_)


        ##############################################

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        ##############################################

        #c7 256 64 64
        out3 = c7
        out3 = self.conv_in256_out64(out3)
        out3_b, out3_c, out3_h, out3_w = out3.size()
        out3_B_64_6464 = out3.contiguous().view(out3_b, out3_c, -1)

        #c8 128 128 128
        out2_B_128_64_64 = self.avg_down64(c8)
        out2 = self.conv_in128_out64(out2_B_128_64_64)
        out2_b, out2_c, out2_h, out2_w = out2.size()
        out2_B_64_6464 = out2.contiguous().view(out2_b, out2_c, -1)

        #c9 64 256 256
        out1 = c9
        out1_b, out1_c, out1_h, out1_w = out1.size()
        out1_B_64_256256 = out1.contiguous().view(out1_b, out1_c, -1)

        out1_B_64_64_64 = self.avg_down64(c9)
        out1_B_64_6464 = out1_B_64_64_64.contiguous().view(out1_b, out1_c, -1)

        out3_B_6464_64 = out3_B_64_6464.contiguous().permute(0, 2, 1)
        out23_B_6464_6464 = torch.bmm(torch.relu(out3_B_6464_64),torch.relu(out2_B_64_6464))

        out23_B_6464_6464 = out23_B_6464_6464/2
        d = torch.sum(out23_B_6464_6464,dim=1)
        d[d !=0]= torch.sqrt(1.0/d[d !=0])
        out23_B_6464_6464 *= d.unsqueeze(1)
        out23_B_6464_6464 *= d.unsqueeze(2)

        out123_B_64_6464 = torch.bmm(torch.relu(out1_B_64_6464),torch.relu(out23_B_6464_6464))
        out123_B_64_64_64 = out123_B_64_6464.view(out1_b, out1_c, out3_h, out3_w)

        out123_B_64_128_128 = self.up_64_128(out123_B_64_64_64)

        out123_B_64_128_128 = self.dropout10(out123_B_64_128_128)

        out123_B_64_256_256 = self.up_128_256(out123_B_64_128_128)

        out123_B_64_256_256 = self.dropout10(out123_B_64_256_256)


        out_B_192_256_256 = torch.cat([out123_B_64_256_256, c9 ,out123_B_64_256_256], dim=1)
        c9 = self.conv_in192_out64 (out_B_192_256_256)
        c10 = self.conv10(c9)
        ##############################################

        # c10 = self.conv10(c9)

        return c10