import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaGCNConv, MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict

"""
ConvNet
"""
def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size, wh_size):
        super(ConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        
        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size*wh_size*wh_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        
        return features, logits
    
"""
ResNet
from https://github.com/kjunelee/MetaOptNet
This ResNet network was designed following the practice of the following papers:
TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).
"""

class DropBlock(MetaModule):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        #self.gamma = gamma
        #self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            #print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            #print (block_mask.size())
            #print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        #print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            #block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            
        block_mask = 1 - padded_mask #[:height, :width]
        return block_mask


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu3 = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, params=None):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x, params=get_subdict(params, 'conv1'))
        out = self.bn1(out, params=get_subdict(params, 'bn1'))
        out = self.relu1(out)

        out = self.conv2(out, params=get_subdict(params, 'conv2'))
        out = self.bn2(out, params=get_subdict(params, 'bn2'))
        out = self.relu2(out)

        out = self.conv3(out, params=get_subdict(params, 'conv3'))
        out = self.bn3(out, params=get_subdict(params, 'bn3'))

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)
        
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class BasicBlockWithoutResidual(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlockWithoutResidual, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu3 = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(stride)
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, params=None):
        self.num_batches_tracked += 1

        out = self.conv1(x, params=get_subdict(params, 'conv1'))
        out = self.bn1(out, params=get_subdict(params, 'bn1'))
        out = self.relu1(out)

        out = self.conv2(out, params=get_subdict(params, 'conv2'))
        out = self.bn2(out, params=get_subdict(params, 'bn2'))
        out = self.relu2(out)

        out = self.conv3(out, params=get_subdict(params, 'conv3'))
        out = self.bn3(out, params=get_subdict(params, 'bn3'))
        out = self.relu3(out)
        
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class ResNet(MetaModule):
    def __init__(self, blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, out_features=5, wh_size=None):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(blocks[0], 64, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer2 = self._make_layer(blocks[1], 128, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer3 = self._make_layer(blocks[2], 256, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(blocks[3], 512, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        self.classifier = MetaLinear(512*wh_size*wh_size, out_features)
        
        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MetaSequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                           kernel_size=1, stride=1, bias=False),
                MetaBatchNorm2d(planes * block.expansion, track_running_stats=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return MetaSequential(*layers)

    def forward(self, x, params=None):
        x = self.layer1(x, params=get_subdict(params, 'layer1'))
        x = self.layer2(x, params=get_subdict(params, 'layer2'))
        x = self.layer3(x, params=get_subdict(params, 'layer3'))
        x = self.layer4(x, params=get_subdict(params, 'layer4'))
        if self.keep_avg_pool:
            x = self.avgpool(x)
        features = x.view((x.size(0), -1))
        logits = self.classifier(self.dropout(features), params=get_subdict(params, 'classifier'))
        return features, logits