import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLSNet(nn.Module):

    def __init__(self,
                 n_classes: int = 19,
                 encoder_depth: int = 18,
                 pretrained: bool = True,
                 up_mode: str = 'bilinear',
                 bn_module=nn.BatchNorm2d,
                 dilations: tuple = None,
                 stop_level: int = 3,
                 multiple_oc_size: int = 4,
                 n_oc: int = 3):
        """
        :param stop_level: level for pruning calc.
            1: first level,
            2: second level,
            3: third level (return all level preds),
        """
        super(MLSNet, self).__init__()

        encoder = ResNetEncoder(encoder_depth=encoder_depth, pretrained=pretrained,
                                bn_module=bn_module, relu_inplace=True)
        activation = nn.ReLU(inplace=True)

        self.encoder = encoder.encoder
        self.channels_list = encoder.channels_list
        self.depth = len(self.channels_list)
        self.up_mode = up_mode
        self.align_corners = False if self.up_mode == 'bilinear' else None
        self.stop_level = stop_level

        if dilations is None:
            dilations = [1] * self.depth

        self.decoder = nn.ModuleList([
            nn.ModuleList([None for _ in range(i)])
            for i in reversed(range(1, self.depth))
        ])

        for i in reversed(range(self.depth - 1)):
            for j in range(0, self.depth - i - 1):
                in_ch = self.channels_list[i] + self.channels_list[i + 1]
                out_ch = self.channels_list[i]
                self.decoder[i][j] = ConvLayer(in_ch, out_ch, kernel_size=3,
                                               padding=dilations[j], dilation=dilations[j],
                                               activation=activation, bn_module=bn_module)

        if n_oc > 0:
            dilation_rates = [(6, 12, 24), (4, 8, 12), (2, 4, 6)]
            for i in range(n_oc):
                self.decoder[i][0].add_module(f'oc_{i}',
                                              ASP_OC_Module(self.channels_list[i], self.channels_list[i],
                                                            size=2 ** (multiple_oc_size - i),
                                                            dilations=dilation_rates[i], bn_module=bn_module))

        self.cls = nn.ModuleList(
            nn.Conv2d(self.channels_list[0], n_classes, kernel_size=1)
            for _ in range(self.depth - 1))

        self._init_weight()

    def _init_weight(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
        for m in self.cls.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, return_all_preds=True):
        """
        :param x: input tensor (shape: B, C, H, W)
        :param return_all_preds:
            True:
                All level predictions are returned. It is necessary when training.
            False:
                Only one prediction are returned according to self.stop_level.
                This enables to avoid redundant calculation and thus saves inference time.
        :return: list of predictions from each level classifiers
        """

        X = [[None for _ in range(i + 1)] for i in reversed(range(self.depth))]
        preds = []

        for i in range(self.depth):

            # encoder part
            if i == 0:
                X[0][0] = self.encoder[0](x)
            else:
                X[i][0] = self.encoder[i](X[i - 1][0])

            # decoder part
            for j in range(i):

                if i - (j + 1) == 0:
                    cat_feat = torch.cat([
                        X[i - (j + 1)][j],
                        F.interpolate(X[i - j][j], scale_factor=2, mode=self.up_mode,
                                      align_corners=self.align_corners)
                    ], dim=1)
                    X[i - (j + 1)][j + 1] = self.decoder[i - (j + 1)][j](cat_feat)

            if i > 0:
                if return_all_preds or i == self.stop_level:
                    preds.append(self.cls[i - 1](X[0][i]))

                if i == self.stop_level:
                    break  # pruning

        return preds

    def adaptive_inference(self, x, threshold=0.93):
        """adaptive inference which enables to save computation time
        by pruning depending on the hardness of input."""

        X = [[None for _ in range(i + 1)] for i in reversed(range(self.depth))]

        for i in range(self.depth):

            # encoder part
            if i == 0:
                X[0][0] = self.encoder[0](x)
            else:
                X[i][0] = self.encoder[i](X[i - 1][0])

            # decoder part
            for j in range(i):
                cat_feat = torch.cat([
                    X[i - (j + 1)][j],
                    F.interpolate(X[i - j][j], scale_factor=2, mode=self.up_mode,
                                  align_corners=self.align_corners)
                ], dim=1)
                X[i - (j + 1)][j + 1] = self.decoder[i - (j + 1)][j](cat_feat)

            if i > 0:
                logits = self.cls[i - 1](X[0][i])
                avg_conf = F.softmax(logits, dim=1).max(dim=1)[0].mean()
                if avg_conf > threshold:
                    break

        return logits, i


class NoOperation(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOperation, self).__init__()

    def forward(self, x):
        return x


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 bn_module=nn.BatchNorm2d, activation=nn.ReLU(inplace=True), use_cbam=False):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation))

        if bn_module is not None:
            self.add_module('bn', bn_module(out_channels))

        if activation is not None:
            self.add_module('act', activation)

        if use_cbam:
            from . import attention
            self.add_module('cbam', attention.CBAM(out_channels, bn_module=bn_module))


class EncoderBase(nn.Module):

    def __init__(self, encoder, channels_list):
        super(EncoderBase, self).__init__()
        self.encoder = encoder
        self.channels_list = channels_list

    def forward(self, x):

        bridges = []
        for down in self.encoder:
            x = down(x)
            bridges.append(x)

        return bridges


class BasicEncoder(EncoderBase):

    def __init__(self, input_channels=1, depth=4,
                 channels=32, pooling='max',
                 bn_module=nn.BatchNorm2d, activation=nn.ReLU(inplace=True), se_module=None):

        depth = depth
        channels_list = [channels * 2 ** i for i in range(depth)]

        if pooling == 'avg':
            pooling = nn.AvgPool2d(2)
        elif pooling == 'max':
            pooling = nn.MaxPool2d(2)

        down_path = nn.ModuleList()
        prev_channels = input_channels
        for i in range(depth):
            layers = [
                pooling if i != 0 else NoOperation(),
                ConvLayer(prev_channels, channels * 2 ** i, 3,
                          padding=1, bn_module=bn_module,
                          activation=activation),
                ConvLayer(channels * 2 ** i, channels * 2 ** i, 3,
                          padding=1, bn_module=bn_module,
                          activation=activation),
            ]
            if se_module is not None:
                layers.append(se_module(channels * 2 ** i))

            down_path.append(nn.Sequential(*layers))
            prev_channels = channels * 2 ** i

        super(BasicEncoder, self).__init__(down_path, channels_list)


class ResNetEncoder(EncoderBase):

    def __init__(self, encoder_depth=18, pretrained=True,
                 bn_module=nn.BatchNorm2d, relu_inplace=False):

        if encoder_depth == 18:
            backbone = resnet_csail.resnet18(pretrained=pretrained, bn_module=bn_module, relu_inplace=relu_inplace)
            channels_list = [64, 128, 256, 512]
        elif encoder_depth == 34:
            backbone = torchvision.models.resnet34(pretrained=pretrained)
            channels_list = [64, 128, 256, 512]
        elif encoder_depth == 50:
            backbone = resnet_csail.resnet50(pretrained=pretrained, bn_module=bn_module, relu_inplace=relu_inplace)
            channels_list = [256, 512, 1024, 2048]
        elif encoder_depth == 101:  # res4b22 corresponds to layer3[-1]
            backbone = resnet_csail.resnet101(pretrained=pretrained, bn_module=bn_module, relu_inplace=relu_inplace)
            channels_list = [256, 512, 1024, 2048]
        elif encoder_depth == 152:
            backbone = torchvision.models.resnet152(pretrained=pretrained)
            channels_list = [256, 512, 1024, 2048]

        elif encoder_depth == 38:
            from . import wider_resnet
            backbone = wider_resnet.net_wider_resnet38_a2()
            if pretrained:
                state_dict = torch.load('/opt/segmentation/weights/wide_resnet38_deeplab_vistas.pth.tar')
                backbone.load_state_dict(state_dict['state_dict']['body'], strict=True)
            channels_list = [256, 512, 1024, 4096]

        elif encoder_depth == 'X_101':
            from experiments import resnet_not_inplace
            backbone = resnet_not_inplace.get_pretrained_backbone(
                cfg_path='/opt/segmentation/maskrcnn-benchmark/configs/dtc/MX_101_32x8d_FPN.yaml',
                bn_module=bn_module
            )
            channels_list = [256, 512, 1024, 2048]

        else:
            raise ValueError('invalid value (encoder_depth)')

        if encoder_depth == 38:
            encoder = nn.ModuleList([
                nn.Sequential(backbone.mod1, backbone.pool2, backbone.mod2, backbone.pool3, backbone.mod3),
                backbone.mod4,
                backbone.mod5,
                nn.Sequential(backbone.mod6, backbone.mod7),
            ])
        elif encoder_depth in [18, 50, 101]:
            encoder = nn.ModuleList([
                nn.Sequential(
                    backbone.conv1, backbone.bn1, backbone.relu1,
                    backbone.conv2, backbone.bn2, backbone.relu2,
                    backbone.conv3, backbone.bn3, backbone.relu3,
                    backbone.maxpool,
                    backbone.layer1,
                ),
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ])
        elif encoder_depth in [34, 152]:
            encoder = nn.ModuleList([
                nn.Sequential(
                    backbone.conv1, backbone.bn1, backbone.relu,
                    backbone.maxpool,
                    backbone.layer1,
                ),
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ])
        elif encoder_depth == 'X_101':
            encoder = nn.ModuleList([
                nn.Sequential(backbone.stem, backbone.layer1),
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ])

        super(ResNetEncoder, self).__init__(encoder, channels_list)


class SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1,
                 bn_module=nn.BatchNorm2d):
        super(SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            bn_module(self.key_channels),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        query = key.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, sizes=([1]), bn_module=nn.BatchNorm2d):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, out_channels, key_channels, value_channels, size, bn_module)
            for size in sizes
        ])

    @staticmethod
    def _make_stage(in_channels, output_channels, key_channels, value_channels, size, bn_module):
        return SelfAttentionBlock(in_channels,
                                  key_channels,
                                  value_channels,
                                  output_channels,
                                  size,
                                  bn_module)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        return context


class ASP_OC_Module(nn.Module):
    """
    OC-Module (bit modified version)
    ref: https://github.com/PkuRainBow/OCNet/blob/master/LICENSE
    """
    def __init__(self, in_features=2048, out_features=2048, dilations=(2, 5, 9), bn_module=nn.BatchNorm2d, size=1):
        super(ASP_OC_Module, self).__init__()
        internal_features = in_features // 4
        self.context = nn.Sequential(
            nn.Conv2d(in_features, internal_features, kernel_size=3, padding=1, dilation=1, bias=True),
            bn_module(internal_features),
            BaseOC_Context_Module(in_channels=internal_features, out_channels=internal_features,
                                  key_channels=internal_features // 2, value_channels=internal_features,
                                  sizes=([size]), bn_module=bn_module))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_features, internal_features, kernel_size=1, padding=0, dilation=1, bias=False),
            bn_module(internal_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_features, internal_features, kernel_size=3, padding=dilations[0], dilation=dilations[0],
                      bias=False),
            bn_module(internal_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_features, internal_features, kernel_size=3, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            bn_module(internal_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_features, internal_features, kernel_size=3, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            bn_module(internal_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(internal_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            bn_module(out_features),
            nn.Dropout2d(0.1)
        )

    @staticmethod
    def _cat_each(feat1, feat2, feat3, feat4, feat5):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output
