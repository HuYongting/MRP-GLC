import time
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
# from loss import FourConsistencyLoss,Sim
class Encoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Encoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.moduleConv3 = Basic(128, 256)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)

        return tensorConv1, tensorConv2, tensorConv3

class EncoderMotion(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(EncoderMotion, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorConv4 = self.moduleBatchNorm(self.moduleReLU(tensorConv4))

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3

class Decoder(torch.nn.Module):
    def __init__(self, n_512=2, n_frame=1,num=4):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic((512 * n_512), 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(256*(1+num), 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(128*(1+num), 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(64*(1+num), 3 * n_frame, 64)

    def forward(self, x, skip1, skip2, skip3):

        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)

        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)

        output = self.moduleDeconv1(cat2)

        return output

class Memory(torch.nn.Module):
    def __init__(self,memory_size=100,memory_dim=512):
        super(Memory, self).__init__()
        self.memory_shape = [memory_size, memory_dim]
        self.memory_w = nn.init.normal_(torch.empty(self.memory_shape), mean=0.0, std=1.0)
        self.memory_w = nn.Parameter(self.memory_w, requires_grad=True)

    def forward(self,fea):   #  (b,512,32,32)
        # memory addressing
        b,c,h,w = fea.size()
        fea_reshape = fea.permute(0,2,3,1).reshape(b*h*w,c)
        query_norm = F.normalize(fea_reshape, dim=1)
        memory_norm = F.normalize(self.memory_w, dim=1)
        s = torch.mm(query_norm, memory_norm.transpose(dim0=0, dim1=1))
        addressing_vec = F.softmax(s, dim=1)
        memory_feature = torch.mm(addressing_vec, self.memory_w)
        memory_feature = memory_feature.reshape(b,h,w,c).permute(0,3,1,2)
        updated_fea = torch.cat([fea, memory_feature], dim=1)
        return updated_fea

class DCN(nn.Module):
    def __init__(self, inplanes = 512, planes = 512, kernel_size=3, stride=1, padding=1, bias=False):
        super(DCN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x,offset):
        out = self.conv1(offset)
        out = self.conv2(x, out)  # input offset
        return out

class ScaleFusion(torch.nn.Module):
    def __init__(self,channel=512):
        super(ScaleFusion, self).__init__()
        self.conv2 = nn.Conv2d(channel // 2, channel, kernel_size = 1, stride=1,padding=0, bias=False)
        self.conv4 = nn.Conv2d(channel // 4, channel, kernel_size = 1, stride=1,padding=0, bias=False)
        self.conv = nn.Conv2d(channel * 3, channel, kernel_size = 1, stride=1,padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,fea,fea2,fea4):   #  (b,512,32,32)
        fea2 = self.relu(self.conv2(fea2))
        fea4 = self.relu(self.conv4(fea4))
        y = torch.cat([fea,fea2,fea4],dim=1)
        y = self.relu(self.conv(y))
        return y

class MotionSkipFusion(torch.nn.Module):
    def __init__(self,channel=512):
        super(MotionSkipFusion, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.fea_dcn = DCN(inplanes = channel * 2, planes =channel * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip1_dcn = DCN(inplanes = channel // 8 , planes =channel // 8 , kernel_size=3, stride=1, padding=1, bias=False)
        self.skip2_dcn = DCN(inplanes = channel // 4, planes =channel // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip3_dcn = DCN(inplanes = channel // 2, planes =channel // 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.fea_up = nn.ConvTranspose2d(in_channels=channel * 2, out_channels=channel // 2, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.skip1_up = nn.ConvTranspose2d(in_channels=channel * 2, out_channels=channel, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.skip2_up = nn.ConvTranspose2d(in_channels=channel // 4, out_channels=channel // 8, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.skip3_up = nn.ConvTranspose2d(in_channels=channel // 2, out_channels=channel // 4, kernel_size=3, stride=2, padding=1,output_padding=1)

        self.skip1_conv = nn.Conv2d(channel // 4, channel // 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.skip2_conv = nn.Conv2d(channel // 2, channel // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.skip3_conv = nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self,feas_motion):   #
        fea, skip1, skip2, skip3 = feas_motion[0],feas_motion[1],feas_motion[2],feas_motion[3]
        # feature fusion
        # fea
        fea_up = self.fea_up(fea)
        fea = self.fea_dcn(fea,fea)
        # skip3
        skip3 = self.relu(self.skip3_conv(torch.cat([skip3,fea_up],dim=1)))
        skip3_up = self.skip3_up(skip3)
        skip3 = self.skip3_dcn(skip3,skip3)
        # skip2
        skip2 = self.relu(self.skip2_conv(torch.cat([skip2,skip3_up],dim=1)))
        skip2_up = self.skip2_up(skip2)
        skip2 = self.skip2_dcn(skip2,skip2)
        # skip1
        skip1 = self.relu(self.skip1_conv(torch.cat([skip1,skip2_up],dim=1)))
        skip1 = self.skip1_dcn(skip1,skip1)
        return fea,skip1,skip2,skip3

class bottleneck_mid(torch.nn.Module):
    def __init__(self):
        super(bottleneck_mid, self).__init__()

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # pooling

                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
            )

        self.conv512 = Basic_(256,512)
        self.conv256 = Basic_(256,256)
        self.conv128 = Basic_(256,128)

        self.memory512 = Memory(10,512)
        self.memory256 = Memory(10,256)
        self.memory128 = Memory(10,128)

        self.prototype_fusion = ScaleFusion(512)
        self.residual_fusion = ScaleFusion(512)


    def forward(self,imgs):  #
        out512 = self.memory512(self.conv512(imgs))
        out256 = self.memory256(self.conv256(imgs))
        out128 = self.memory128(self.conv128(imgs))
        res512 = out512[:,:512,:]-out512[:,512:,:]
        res256 = out256[:,:256,:]-out256[:,256:,:]
        res128 = out128[:,:128,:]-out128[:,128:,:]

        prototype = self.prototype_fusion(out512[:,512:,:],out256[:,256:,:],out128[:,128:,:])
        residual = self.residual_fusion(res512,res256,res128)

        return torch.cat([prototype,residual],dim=1)

class SkipCross(torch.nn.Module):
    def __init__(self):
        super(SkipCross, self).__init__()

        def conv(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(intInput, intOutput, kernel_size=5, stride=2, padding=1),  # pooling
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )

        def deconv(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(intInput, intOutput, kernel_size=5, stride=2, padding=1,output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def conv1x1(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(intInput, intOutput, kernel_size=1, stride=1, padding=0),  # pooling
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )

        self.conv_fea = conv(512 *2 ,512*2)
        self.conv_skip1 = conv(64,64)
        self.conv_skip2 = conv(128,128)
        self.conv_skip3 = conv(256,256)

        self.deconv_fea = deconv(512 * 2,512*2)
        self.deconv_skip1 = deconv(64,64)
        self.deconv_skip2 = deconv(128,128)
        self.deconv_skip3 = deconv(256,256)

        self.fea = conv1x1(512 * 4,512 * 2)
        self.skip1 = conv1x1(64 * 2,64)
        self.skip2 = conv1x1(128 * 2,128)
        self.skip3 = conv1x1(256 * 2,256)

        self.memory = Memory(10,512)

    def forward(self,frames,pers,B):  #
        fea_frames, skip1_frames, skip2_frames, skip3_frames = frames[0],frames[1],frames[2],frames[3]
        fea_pers, skip1_pers, skip2_pers, skip3_pers = pers[0],pers[1],pers[2],pers[3]
        fea_pers = self.memory(fea_pers)

        fea_pers, skip1_pers = self.conv_fea(fea_pers),self.conv_skip1(skip1_pers)
        skip2_pers, skip3_pers = self.conv_skip2(skip2_pers),self.conv_skip3(skip3_pers)
        sim = Sim(B,fea_pers, skip1_pers, skip2_pers, skip3_pers)

        fea, skip1 = self.conv_fea(fea_frames),self.conv_skip1(skip1_frames)
        skip2, skip3 = self.conv_skip2(skip2_frames),self.conv_skip3(skip3_frames)
        fea, skip1 = self.deconv_fea(fea),self.deconv_skip1(skip1)
        skip2, skip3 = self.deconv_skip2(skip2),self.deconv_skip3(skip3)

        fea = self.fea(torch.cat([fea_frames,fea],dim=1))
        skip1 = self.skip1(torch.cat([skip1_frames,skip1],dim=1))
        skip2 = self.skip2(torch.cat([skip2_frames,skip2],dim=1))
        skip3 = self.skip3(torch.cat([skip3_frames,skip3],dim=1))

        # fea = self.fea(torch.cat([fea_pers,fea],dim=1))
        # skip1 = self.skip1(torch.cat([skip1_pers,skip1],dim=1))
        # skip2 = self.skip2(torch.cat([skip2_pers,skip2],dim=1))
        # skip3 = self.skip3(torch.cat([skip3_pers,skip3],dim=1))
        return [fea, skip1, skip2, skip3],sim

class Model(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=5,num=4):
        super(Model, self).__init__()
        self.encoderAppearance = Encoder(5, n_channel)
        self.encoderPerFrames = EncoderMotion(2, n_channel)
        self.midAppearance = bottleneck_mid()
        self.skipCross = SkipCross()
        self.decoder = Decoder(n_512=2,n_frame=1,num=1)
        self.T = num

    def forward(self,imgs):  # 0:motion
        skip1_app, skip2_app, skip3_app = self.encoderAppearance(imgs)
        fea_app = self.midAppearance(skip3_app)
        B, TC, H, W = imgs.shape
        imgs_reshape = imgs.reshape(B*self.T, TC//self.T, H, W)
        fea_motion, skip1_motion, skip2_motion, skip3_motion = self.encoderPerFrames(imgs_reshape)
        fea_frames,sim = self.skipCross([fea_app,skip1_app, skip2_app, skip3_app],[fea_motion, skip1_motion, skip2_motion, skip3_motion],B)
        out_recon_motion = self.decoder(fea_frames[0],fea_frames[1],fea_frames[2],fea_frames[3])
        return out_recon_motion,sim


if __name__ == '__main__':
    imgs = torch.rand(1,12,256,256).cpu()
    motions = torch.rand(1,12,256,256).cpu()
    model = Model().cpu()
    res,_ = model.forward(imgs)
    print(res.shape)
