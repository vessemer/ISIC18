import math
import torch
import torch.nn as nn
from .abstract_model import (
    EncoderDecoder, get_slice, 
    Upscale, 
    AbstractModel,
    UnetDecoderBlock, 
    ConvBottleneck, 
    SumBottleneck, 
    UnetBNDecoderBlock, 
    PathAggregationEncoderDecoder, 
    UnetDoubleDecoderBlock, 
    DPEncoderDecoder
)
import torch.nn.functional as F

import os
import torch.utils.model_zoo as model_zoo
# from . import resnet, vgg, inception
# from .inplace_abn.models.wider_resnet import init_wider_resnet
from .inplace_abn.abn import ABN
from .dpn import dpn92, dpn68


encoder_params = {
    'dpn92':
        {'filters': [64, 336, 704, 1552, 2688],
         'decoder': [256, 256, 256, 256, 256],
         'fpn': 128,
         'init_op': dpn92,
         'url':'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth'},
    'dpn68': 
        {'filters': [10, 144, 320, 704, 832],
         'decoder': [144, 144, 144, 144, 144],
         'fpn': 72,
         'init_op': dpn68,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth',
        },
}


class FPNDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class FPNSumBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        enc = self.layer(enc)
        dec = dec + enc
        return dec


class FPNConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, scale_factor=None):
        super().__init__()
        if scale_factor is not None:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_channels, out_channels, 3, padding=1),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.layer(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class DPFPEncoderDecoder(AbstractModel):
    #should be successor of encoder decoder
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34', model_type='unet', dropout=.2):
        if not hasattr(self, 'decoder_type'):
            self.decoder_type = Upscale.upsample_bilinear
        if not hasattr(self, 'fpn_block_type'):
            self.fpn_block = FPNConvBlock
        
        self.type = model_type
        if self.type == 'unet':
            self.decoder_block = UnetDecoderBlock
            self.bottleneck_type = ConvBottleneck
        elif self.type == 'fpn':
            self.decoder_block = FPNDecoderBlock
            self.bottleneck_type = FPNSumBottleneck

        self.filters = encoder_params[encoder_name]['filters']
        self.dec_filters = encoder_params[encoder_name]['decoder']
        self.fpn_filters = encoder_params[encoder_name]['fpn']

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.decoder_stages = [self.get_decoder(idx) for idx in range(2, len(self.filters))]
        self.decoder_stages = nn.ModuleList(self.decoder_stages[::-1]) # 3 only, 'cause 1st layer has no downsampling

        self.bottlenecks = [self.get_bottleneck(idx) for idx in range(2, len(self.filters))]
        self.bottlenecks = nn.ModuleList(self.bottlenecks[::-1]) #todo init from type 

        if self.type == 'unet':
            self.last_decoder_stage = ConvBlock(self.filters[1], self.filters[1], self.dec_filters[0])
        elif self.type == 'fpn':
            self.last_decoder_stage = ConvBlock(self.filters[1], self.filters[1], self.filters[1])

        self.fpn_stages = nn.ModuleList([self.get_fpn_block(idx) for idx in range(4)])

        self.final = self.make_final_classifier(len(self.fpn_stages) * self.fpn_filters, num_classes)

        if dropout:
            self.dropout = nn.Dropout2d(p=dropout)

        # for fold 1 only
        #self.final_upscale = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op'](in_channels=num_channels)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        enc_results = enc_results[::-1]

        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)

        dec_results = []
        for idx, bottleneck in enumerate(self.bottlenecks):
            x = self.decoder_stages[idx](x)
            dec_results.append(x.clone())
            x = self.bottlenecks[idx](x, enc_results[idx + 1])

        x = self.last_decoder_stage(x)
        dec_results.append(x.clone())
        dec_results = dec_results[::-1]

        fpn_results = []
        for idx, stage in enumerate(self.fpn_stages):
            x = stage(dec_results[idx])
            fpn_results.append(x.clone())

        fpn_results = torch.cat(fpn_results, dim=1)
        fpn_results = self.dropout(fpn_results)

        f = self.final(fpn_results)
        #f = self.final_upscale(f)

        return f

    def get_decoder(self, layer):
        if self.type == 'unet':
            return self.decoder_block(
                self.filters[layer], 
                self.filters[layer], 
                self.dec_filters[max(layer - 1, 0)], 
                self.decoder_type)
        if self.type == 'fpn':
            return self.decoder_block(
                self.filters[layer], 
                self.filters[layer], 
                self.filters[max(layer - 1, 1)], 
                self.decoder_type)

    def get_bottleneck(self, layer):
        layer = max(layer - 1, 0)
        if self.type == 'unet':
            return self.bottleneck_type(
                self.filters[layer] + self.dec_filters[layer], 
                self.filters[layer])
        elif self.type == 'fpn':
            return self.bottleneck_type(
                self.filters[layer], 
                self.filters[layer])

    def get_fpn_block(self, layer):
        if self.type == 'unet':
            return self.fpn_block(
                self.dec_filters[layer], 
                (self.dec_filters[layer] + self.fpn_filters) // 2,
                self.fpn_filters,
                None if layer <= 1 else 2 ** (layer - 1)
            )
        elif self.type == 'fpn':
            return self.fpn_block(
                self.filters[max(layer, 1)],
                self.fpn_filters,
                self.fpn_filters,
                None if layer <= 1 else 2 ** (layer - 1)
            )

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 3, padding=1)
        )


class DPFPNet(DPFPEncoderDecoder):
    def __init__(self, num_classes, num_channels=3, encoder_name='dpn92', 
                 model_type='unet', dropout=.2):
        # self.decoder_block = UnetDoubleDecoderBlock
        self.bottleneck_type = ConvBottleneck
        super().__init__(num_classes, num_channels, encoder_name, model_type, dropout)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.blocks['conv1_1'].conv, #conv
                #encoder.blocks['conv1_1'].bn, #bn
                #encoder.blocks['conv1_1'].act, #relu
                encoder.blocks['conv1_1'].abn, #relu
                encoder.blocks['conv1_1'].pool, #maxpool
            )
        elif layer == 1:
            return nn.Sequential(
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        elif layer == 2:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        elif layer == 3:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        elif layer == 4:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])
