"""PointNet-style encoder built from Minkowski Engine pointwise layers.

Adapted from the Minkowski Engine PointNet example (Copyright (c) 2020 NVIDIA
CORPORATION; Copyright (c) 2018-2020 Chris Choy; MIT License) and the ALPhA
FRIBMink repository. A shared MLP (64, 64, 64, 128, 512) is applied to every
input point and global max pooling gives the 512-dimensional embedding. The
post-pooling linear layer, dropout and classifier are used only for training
on the source task.
"""

import torch.nn as nn
import MinkowskiEngine as ME


def _pointwise(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiLinear(in_channels, out_channels, bias=False),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(),
    )


class MinkowskiPointNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, embedding_channel=512, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.conv1 = _pointwise(in_channels, 64)
        self.conv2 = _pointwise(64, 64)
        self.conv3 = _pointwise(64, 64)
        self.conv4 = _pointwise(64, 128)
        self.conv5 = _pointwise(128, embedding_channel)
        self.max_pool = ME.MinkowskiGlobalMaxPooling()

        self.linear1 = _pointwise(embedding_channel, 512)
        self.dp1 = ME.MinkowskiDropout()
        self.linear2 = ME.MinkowskiLinear(512, out_channels, bias=True)

    def embed(self, x: ME.TensorField):
        """Return the pooled 512-dimensional embedding (one row per event)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.max_pool(x)

    def forward(self, x: ME.TensorField):
        x = self.linear1(self.embed(x))
        return self.linear2(self.dp1(x))
