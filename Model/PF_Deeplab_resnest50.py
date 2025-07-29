import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import timm


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class CPCA(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


class PFBAM(nn.Module):
    def __init__(self, in_channel):
        super(PFBAM, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = simam_module()
        self.cpca = CPCA(in_channels=in_channel, out_channels=in_channel)

    def forward(self, x):
        x = self.simam(x)
        simam_x = x
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        edge = self.cpca(edge)  # 直接传递 edge 张量作为输入
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * simam_x + simam_x
        out = self.cpca(out)  # 同样直接传递 out 张量作为输入
        return out


class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    # x:待被施加空间注意力的浅层特征
    # y:用于计算reverse attention map的深层特征
    def forward(self, x, y):
        a = torch.sigmoid(-y)  # reverse并压缩至0~1区间内以用作空间注意力map
        x = self.convert(x)  # 统一x, y通道数
        x = a.expand(-1, self.channel, -1, -1).mul(x)  # x, y相乘，完成空间注意力
        y = y + self.convs(x)  # 残差连接
        return y


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.pool(x)
        x5 = self.conv1(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        result = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return result


class PF_Deeplab_resnest50(nn.Module):
    def __init__(self, num_classes):
        super(PF_Deeplab_resnest50, self).__init__()

        # Encoder编码部分
        resnet = timm.create_model("resnest50d.in1k", pretrained=True,
                                   pretrained_cfg_overlay=dict(file='../Pre_weight/resnest50_pytorch_model.bin'),
                                   features_only=True, in_chans=3, output_stride=8)
        # 使用timm库加载预训练的主干网络模型，设置 features_only=True 使其返回各个阶段的特征图而非最终的分类结果。
        self.encoder1 = nn.Sequential(*list(resnet.children())[0:3])
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5])
        # 将第二层结果作为低阶特征层结果
        self.encoder3 = nn.Sequential(*list(resnet.children())[5:6])
        self.encoder4 = nn.Sequential(*list(resnet.children())[6:7])
        self.encoder5 = nn.Sequential(*list(resnet.children())[7:8])

        # ASPP模块
        self.aspp = ASPP(in_channels=2048)  # 根据模型输出通道数调整

        # EMA
        self.ema = EMA(1280)
        self.ema2 = EMA(256)

        # RA
        self.ra = RA(256, 256)
        self.convra = nn.Conv2d(256, 256, kernel_size=1)
        self.bnra = nn.BatchNorm2d(256)
        self.relura = nn.ReLU(256)

        # PFBAM
        self.pfbam = PFBAM(512)

        # 解码部分
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.up_sample1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.up_sample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.up_sample3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv5 = nn.Conv2d(128, num_classes, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_classes)
        self.Sigmoid = nn.Sigmoid()

        # 定义encoder234的卷积
        self.convencoder5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.convencoder4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.convencoder3 = nn.Conv2d(512, 256, kernel_size=1)
        self.bnencoder = nn.BatchNorm2d(256)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.encoder1(x)
        # encoder1 2x64x128x128
        print(f"After encoder1: {x.shape}")
        low_feature = self.encoder2(x)
        # encoder2(low_feature) 2x256x64x64
        print(f"After encoder2 (low_feature): {low_feature.shape}")
        x = self.encoder3(low_feature)
        print(f"After encoder3: {x.shape}")
        # encoder3 2x512x32x32

        # encoder3卷积
        x_encoder3 = self.convencoder3(x)
        x_encoder3 = self.bnencoder(x_encoder3)
        x_encoder3 = self.relu(x_encoder3)

        x = self.encoder4(x)

        # encoder4卷积
        x_encoder4 = self.convencoder4(x)
        x_encoder4 = self.bnencoder(x_encoder4)
        x_encoder4 = self.relu(x_encoder4)

        print(f"After encoder4: {x.shape}")
        # encoder4 2x1024x32x32
        x = self.encoder5(x)

        # encoder5卷积
        x_encoder5 = self.convencoder5(x)
        x_encoder5 = self.bnencoder(x_encoder5)
        x_encoder5 = self.relu(x_encoder5)

        print(f"After encoder5: {x.shape}")
        # encoder5 2x2048x32x32

        # 将encoder5赋值为ra_high_feature
        ra_high_feature = x

        high_feature = self.aspp(x)
        print(f"After aspp: {high_feature.shape}")

        # 在aspp后加入EMA
        high_feature = self.ema(high_feature)

        high_feature = self.conv1(high_feature)
        high_feature = self.bn1(high_feature)
        high_feature = self.relu(high_feature)

        # encoder345相加high_feature
        high_feature = high_feature + x_encoder5
        high_feature = high_feature + x_encoder4
        high_feature = high_feature + x_encoder3

        # print(f"Before up_sample1: {high_feature.shape}")
        high_feature = self.up_sample1(high_feature)
        print(f"After up_sample1: {high_feature.shape}")
        print(f"low_feature: {low_feature.shape}")

        if low_feature.shape[2:] != high_feature.shape[2:]:
            low_feature = F.interpolate(low_feature, size=high_feature.shape[2:], mode='bilinear', align_corners=True)
        print(f"low_feature: {low_feature.shape}")

        # EMA
        low_feature = self.ema2(low_feature)
        low_feature = self.conv2(low_feature)
        low_feature = self.bn2(low_feature)
        low_feature = self.relu(low_feature)
        # print(f"high_feature: {high_feature.shape}")
        # 重新定义赋值low_feature
        ra_low_feature = low_feature

        # 加入RA
        # print(f"high_feature: {high_feature.shape}")
        # print(f"ra_high_feature: {ra_high_feature.shape}")
        # ra_high_feature = F.interpolate(ra_high_feature, size=high_feature.shape[2:], mode='bilinear',
        #                                 align_corners=True)
        # print(f"ra_high_feature: {ra_high_feature.shape}")
        high_feature = self.ra(ra_low_feature, high_feature)
        # 加入卷积bnrelu
        high_feature = self.convra(high_feature)
        high_feature = self.bnra(high_feature)
        high_feature = self.relura(high_feature)

        middle_feature = torch.cat((high_feature, low_feature), dim=1)

        # PFBAM
        middle_feature = self.pfbam(middle_feature)

        middle_feature = self.conv3(middle_feature)
        middle_feature = self.bn3(middle_feature)
        middle_feature = self.relu(middle_feature)
        middle_feature = self.up_sample2(middle_feature)

        middle_feature = self.conv4(middle_feature)
        middle_feature = self.bn4(middle_feature)
        middle_feature = self.relu(middle_feature)
        middle_feature = self.up_sample3(middle_feature)

        out_feature = self.conv5(middle_feature)
        out_feature = self.bn5(out_feature)
        out_feature = self.Sigmoid(out_feature)

        return out_feature


if __name__ == "__main__":
    model = PF_Deeplab_resnest50(num_classes=1)
    summary(model, input_size=(3, 256, 256), device="cpu")
