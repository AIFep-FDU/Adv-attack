import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activation='ReLU'):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut or stride == 2) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                               padding=0, bias=False) or None

    def forward(self, x):
        self.x = x
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.convShortcut is None else self.convShortcut(x), out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride, drop_rate=0.0, block_type='basic_block', activation='ReLU'):
        super(NetworkBlock, self).__init__()
        if block_type == 'basic_block':
            block = BasicBlock
        else:
            raise('Unknown block: %s' % (block_type))

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, activation)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activation):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes, i == 0 and stride or 1, drop_rate, activation=activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class RobustWideResNet(nn.Module):
    def __init__(self, num_classes=10, channel_configs=[16, 160, 320, 640],
                 depth_configs=[5, 5, 5], stride_config=[1, 2, 2], limit=100*1000):
        super(RobustWideResNet, self).__init__()
        assert len(channel_configs) - 1 == len(depth_configs) == len(stride_config)
        self.channel_configs = channel_configs
        self.depth_configs = depth_configs
        self.stride_config = stride_config
        self.get_feature = False
        self.get_stem_out = False

        self.stem_conv = nn.Conv2d(3, channel_configs[0], kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList([])

        for i, stride in enumerate(stride_config):
            self.blocks.append(NetworkBlock(nb_layers=depth_configs[i], in_planes=channel_configs[i],
                                            out_planes=channel_configs[i+1], stride=stride,))

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channel_configs[-1])
        self.relu = nn.ReLU(inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_configs[-1], num_classes)
        self.fc_size = channel_configs[-1]
        self.__forward_counter = 0
        self.__limit = limit

    def get_counter(self):
        return self.__forward_counter

    def get_limit(self):
        return self.__limit

    def forward_features(self, x):
        self.__forward_counter += x.shape[0]
        features = []
        out = self.stem_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
            features.append(out)
        out = self.relu(self.bn1(out))
        out = self.global_pooling(out)
        features.append(out)
        out = out.view(-1, self.fc_size)
        out = self.fc(out)
        features.append(out)
        return features, out

    def forward(self, x):
        self.__forward_counter += x.shape[0]
        out = self.stem_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
        out = self.relu(self.bn1(out))
        out = self.global_pooling(out)
        out = out.view(-1, self.fc_size)
        out = self.fc(out)
        return out
