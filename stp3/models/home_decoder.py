from typing import Tuple, Optional

import torch.nn as nn
import torch

from .convolution_block import CNNBlock, TransposeCNNBlock

class HeatmapEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        """
        Encodes rasterized HD map features

        Args:
            input_shape: Raster input shape
        """
        super(HeatmapEncoder, self).__init__()
        self._input_shape = input_shape

        # input_shape[0] = 64
        self._convs = nn.ModuleList([
            # CNNBlock(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=1, padding='same'),
            # CNNBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            CNNBlock(in_channels=input_shape[0], out_channels=128, kernel_size=3, stride=1, padding='same'),
            CNNBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
            CNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same')
        ])
        self._maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        # self._maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Rasterized features

        Returns: extracted features
        """
        assert tuple(x.shape[1:]) == self._input_shape, f'Wrong input shape: Expected {self._input_shape} but got {tuple(x.shape[1:])}'

        
        for conv in self._convs[:-1]:
            x = self._maxpool(conv(x))
        x = self._convs[-1](x)

        return x


class HeatmapOutputDecoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        """
        Estimates probabilities in form of a heatmap for agent end points

        Args:
            input_shape: Input shape
        """
        super(HeatmapOutputDecoder, self).__init__()
        self._input_shape = input_shape

        # input_shape[0] = 64+512

        # self._transpose_convs = nn.ModuleList([
        #     TransposeCNNBlock(in_channels=input_shape[0], out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=0),
        #     TransposeCNNBlock(in_channels=256, out_channels=128, kernel_size=3, stride=2, output_padding=0),
        #     TransposeCNNBlock(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=0),
        #     TransposeCNNBlock(in_channels=64, out_channels=32, kernel_size=3, stride=2, output_padding=1)
        # ])
        # self._conv1 = CNNBlock(in_channels=encoder_input_shape[0]+32, out_channels=16, kernel_size=7, padding='same')
        # self._conv2 = CNNBlock(in_channels=16, out_channels=1, kernel_size=3, padding='same')
        # self._sigmoid = nn.Sigmoid()

        ##### Todos #####
        self._transpose_convs = nn.ModuleList([
            TransposeCNNBlock(in_channels=input_shape[0], out_channels=256, kernel_size=3, stride=1, padding=1, output_padding=0),
            TransposeCNNBlock(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, output_padding=0),
            TransposeCNNBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0),
            TransposeCNNBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0)
        ])

        self._conv1 = CNNBlock(in_channels=32, out_channels=16, kernel_size=7, padding='same')
        self._conv2 = CNNBlock(in_channels=16, out_channels=1, kernel_size=3, padding='same')
        self._sigmoid = nn.Sigmoid()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features

        Returns: Heatmap
        """
        assert tuple(x.shape[1:]) == self._input_shape, f'Wrong input shape: Expected {self._input_shape} but got {tuple(x.shape[1:])}'

        for trans_conv in self._transpose_convs:
            x = trans_conv(x)


        x = self._conv1(x)
        x = self._conv2(x)
        x = self._sigmoid(x)
        
        return {
            'heatmap': x
        }