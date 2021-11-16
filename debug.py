# import torch
# import torch.nn as nn


# def autopad(k, p=None):  # kernel, padding
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p

# class TransformerLayer(nn.Module):
#     # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
#     def __init__(self, c, num_heads):
#         super().__init__()
#         self.q = nn.Linear(c, c, bias=False)
#         self.k = nn.Linear(c, c, bias=False)
#         self.v = nn.Linear(c, c, bias=False)

#         self.attention_norm = nn.LayerNorm(c, eps=1e-6)
#         self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)

#         self.ffn_norm = nn.LayerNorm(c, eps=1e-6)
#         self.ffn = Mlp(c)

#     def forward(self, x):
#         h = x
#         x = self.attention_norm(x) # add
#         x = self.ma(self.q(x), self.k(x), self.v(x))[0] + h

#         h = x
#         x = self.ffn_norm(x)       # add
#         x = self.ffn(x) + h
#         return x

# class Mlp(nn.Module):
#     def __init__(self, c):
#         super(Mlp, self).__init__()
#         self.fc1 = nn.Linear(c, 5120)
#         self.fc2 = nn.Linear(5120, c)
#         self.act_fn = torch.nn.functional.gelu
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act_fn(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x

# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         return self.act(self.conv(x))

# class Encoder(nn.Module):
#     # Vision Transformer https://arxiv.org/abs/2010.11929
#     def __init__(self, c1, c2, num_heads, num_layers):
#         super().__init__()
#         self.conv = None
#         if c1 != c2:
#             self.conv = Conv(c1, c2)
#         self.linear = nn.Linear(c2, c2)  # learnable position embedding
#         self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
#         self.c2 = c2

#     def forward(self, x):
#         if self.conv is not None:
#             x = self.conv(x)
#         b, _, w, h = x.shape
#         # print(x.shape)
#         p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
#         return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)

# x = torch.ones(1, 1280, 8, 8)
# m = Encoder(1280, 1280, 16, 3)
# r = m(x)

# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# """
# Creates a GhostNet Model as defined in:
# GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
# https://arxiv.org/abs/1911.11907
# Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# __all__ = ['ghost_net']


# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# def hard_sigmoid(x, inplace: bool = False):
#     if inplace:
#         return x.add_(3.).clamp_(0., 6.).div_(6.)
#     else:
#         return F.relu6(x + 3.) / 6.


# class SqueezeExcite(nn.Module):
#     def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
#                  act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
#         super(SqueezeExcite, self).__init__()
#         self.gate_fn = gate_fn
#         reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
#         self.act1 = act_layer(inplace=True)
#         self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

#     def forward(self, x):
#         x_se = self.avg_pool(x)
#         x_se = self.conv_reduce(x_se)
#         x_se = self.act1(x_se)
#         x_se = self.conv_expand(x_se)
#         x = x * self.gate_fn(x_se)
#         return x    

    
# class ConvBnAct(nn.Module):
#     def __init__(self, in_chs, out_chs, kernel_size,
#                  stride=1, act_layer=nn.ReLU):
#         super(ConvBnAct, self).__init__()
#         self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_chs)
#         self.act1 = act_layer(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn1(x)
#         x = self.act1(x)
#         return x


# class GhostModule(nn.Module):
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
#         super(GhostModule, self).__init__()
#         self.oup = oup
#         init_channels = math.ceil(oup / ratio)
#         new_channels = init_channels*(ratio-1)

#         self.primary_conv = nn.Sequential(
#             nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
#             nn.BatchNorm2d(init_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )

#         self.cheap_operation = nn.Sequential(
#             nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
#             nn.BatchNorm2d(new_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )

#     def forward(self, x):
#         x1 = self.primary_conv(x)
#         x2 = self.cheap_operation(x1)
#         out = torch.cat([x1,x2], dim=1)
#         return out[:,:self.oup,:,:]


# class GhostBottleneck(nn.Module):
#     """ Ghost bottleneck w/ optional SE"""

#     def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
#                  stride=1, act_layer=nn.ReLU, se_ratio=0.):
#         super(GhostBottleneck, self).__init__()
#         has_se = se_ratio is not None and se_ratio > 0.
#         self.stride = stride

#         # Point-wise expansion
#         self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

#         # Depth-wise convolution
#         if self.stride > 1:
#             self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
#                              padding=(dw_kernel_size-1)//2,
#                              groups=mid_chs, bias=False)
#             self.bn_dw = nn.BatchNorm2d(mid_chs)

#         # Squeeze-and-excitation
#         if has_se:
#             self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
#         else:
#             self.se = None

#         # Point-wise linear projection
#         self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
#         # shortcut
#         if (in_chs == out_chs and self.stride == 1):
#             self.shortcut = nn.Sequential()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
#                        padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
#                 nn.BatchNorm2d(in_chs),
#                 nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_chs),
#             )


#     def forward(self, x):
#         residual = x

#         # 1st ghost bottleneck
#         x = self.ghost1(x)

#         # Depth-wise convolution
#         if self.stride > 1:
#             x = self.conv_dw(x)
#             x = self.bn_dw(x)

#         # Squeeze-and-excitation
#         if self.se is not None:
#             x = self.se(x)

#         # 2nd ghost bottleneck
#         x = self.ghost2(x)
        
#         x += self.shortcut(residual)
#         return x


# class GhostNet(nn.Module):
#     def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
#         super(GhostNet, self).__init__()
#         # setting of inverted residual blocks
#         self.cfgs = cfgs
#         self.dropout = dropout

#         # building first layer
#         output_channel = _make_divisible(16 * width, 4)
#         self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(output_channel)
#         self.act1 = nn.ReLU(inplace=True)
#         input_channel = output_channel

#         # building inverted residual blocks
#         stages = []
#         block = GhostBottleneck
#         for cfg in self.cfgs:
#             layers = []
#             for k, exp_size, c, se_ratio, s in cfg:
#                 output_channel = _make_divisible(c * width, 4)
#                 hidden_channel = _make_divisible(exp_size * width, 4)
#                 layers.append(block(input_channel, hidden_channel, output_channel, k, s,
#                               se_ratio=se_ratio))
#                 input_channel = output_channel
#             stages.append(nn.Sequential(*layers))

#         output_channel = _make_divisible(exp_size * width, 4)
#         stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
#         input_channel = output_channel
        
#         self.blocks = nn.Sequential(*stages)        

#         # building last several layers
#         output_channel = 1280
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
#         self.act2 = nn.ReLU(inplace=True)
#         self.classifier = nn.Linear(output_channel, num_classes)

#     def forward(self, x):
#         x = self.conv_stem(x)
#         x = self.bn1(x)
#         x = self.act1(x)
#         x = self.blocks(x)
#         x = self.global_pool(x)
#         x = self.conv_head(x)
#         x = self.act2(x)
#         x = x.view(x.size(0), -1)
#         if self.dropout > 0.:
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.classifier(x)
#         return x


# def ghostnet(**kwargs):
#     """
#     Constructs a GhostNet model
#     """
#     cfgs = [
#         # k, t, c, SE, s 
#         # stage1
#         [[3,  16,  16, 0, 1]],
#         # stage2
#         [[3,  48,  24, 0, 2]],
#         [[3,  72,  24, 0, 1]],
#         # stage3
#         [[5,  72,  40, 0.25, 2]],
#         [[5, 120,  40, 0.25, 1]],
#         # stage4
#         [[3, 240,  80, 0, 2]],
#         [[3, 200,  80, 0, 1],
#          [3, 184,  80, 0, 1],
#          [3, 184,  80, 0, 1],
#          [3, 480, 112, 0.25, 1],
#          [3, 672, 112, 0.25, 1]
#         ],
#         # stage5
#         [[5, 672, 160, 0.25, 2]],
#         [[5, 960, 160, 0, 1],
#          [5, 960, 160, 0.25, 1],
#          [5, 960, 160, 0, 1],
#          [5, 960, 160, 0.25, 1]
#         ]
#     ]
#     return GhostNet(cfgs, **kwargs)


# if __name__=='__main__':
#     model = ghostnet()
#     model.eval()
#     print(model)
#     input = torch.randn(32,3,320,256)
#     y = model(input)
#     print(y.size())

import math
import torch
from torch import nn
import torch.functional as F


efficientnet_lite_params = {
    # width_coefficient, depth_coefficient, image_size, dropout_rate
    'efficientnet_lite0': [1.0, 1.0, 224, 0.2],
    'efficientnet_lite1': [1.0, 1.1, 240, 0.2],
    'efficientnet_lite2': [1.1, 1.2, 260, 0.3],
    'efficientnet_lite3': [1.2, 1.4, 280, 0.3],
    'efficientnet_lite4': [1.4, 1.8, 300, 0.3],
}


def round_filters(filters, multiplier, divisor=8, min_width=None):
    """Calculate and round number of filters based on width multiplier."""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class drop_connect(nn.Module):
    def __init__(self, drop_connect_rate):
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x, training):
        if not training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor) # 1
        x = (x / keep_prob) * binary_mask
        return x



class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, se_ratio, has_se=False):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = nn.ReLU6(inplace=True)

        self.drop_connect

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._relu(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1  and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x


class EfficientNetLite(nn.Module):
    def __init__(self, widthi_multiplier, depth_multiplier, num_classes, drop_connect_rate, dropout_rate):
        super(EfficientNetLite, self).__init__()

        # Batch norm parameters
        momentum = 0.01
        epsilon = 1e-3
        self.drop_connect_rate = drop_connect_rate

        mb_block_settings = [
            #repeat|kernal_size|stride|expand|input|output|se_ratio
                [1, 3, 1, 1, 32,  16,  0.25],
                [2, 3, 2, 6, 16,  24,  0.25],
                [2, 5, 2, 6, 24,  40,  0.25],
                [3, 3, 2, 6, 40,  80,  0.25],
                [3, 5, 1, 6, 80,  112, 0.25],
                [4, 5, 2, 6, 112, 192, 0.25],
                [1, 3, 1, 6, 192, 320, 0.25]
            ]

        # Stem
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            num_repeat, kernal_size, stride, expand_ratio, input_filters, output_filters, se_ratio = stage_setting
            # Update block input and output filters based on width multiplier.
            input_filters = input_filters if i == 0 else round_filters(input_filters, widthi_multiplier)
            output_filters = round_filters(output_filters, widthi_multiplier)
            num_repeat= num_repeat if i == 0 or i == len(mb_block_settings) - 1  else round_repeats(num_repeat, depth_multiplier)
            

            # The first block needs to take care of stride and filter size increase.
            stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))
            if num_repeat > 1:
                input_filters = output_filters
                stride = 1
            for _ in range(num_repeat - 1):
                stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))
            
            self.blocks.append(stage)

        # Head
        in_channels = round_filters(mb_block_settings[-1][5], widthi_multiplier)
        out_channels = 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.fc = torch.nn.Linear(out_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        idx = 0
        for stage in self.blocks:
            # print(stage)
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                    print(drop_connect_rate)
                x = block(x, drop_connect_rate)
                idx +=1
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()
    
    def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
        

def build_efficientnet_lite(name, num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = efficientnet_lite_params[name]
    model = EfficientNetLite(width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate)
    return model


if __name__ == '__main__':
    model_name = 'efficientnet_lite0'
    model = build_efficientnet_lite(model_name, 1000)
    model.eval()

    # from utils.flops_counter import get_model_complexity_info

    wh = efficientnet_lite_params[model_name][2]
    input_shape = (4, 3, wh, wh)
    model(torch.ones(input_shape))
    # flops, params = get_model_complexity_info(model, input_shape)
    