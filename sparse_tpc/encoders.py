"""Turn events into network inputs for the two encoder architectures.

ResNet14 sees a sparse voxel grid: hit positions quantized with a 0.05 mm
voxel (GADGET II drift measured from the event's first hit), carrying the
scaled positions and scaled log charge as features. During training, each
event is shifted by a random translation of up to 0.9 mm per axis.

The PointNet-style encoder sees the scaled positions directly as continuous
coordinates of a tensor field, with the same four features.
"""

import numpy as np
import torch
import MinkowskiEngine as ME

from .datasets import scale_points
from .pointnet import MinkowskiPointNet
from .resnet import ResNet14

VOXEL_SIZE = 0.05


class EventEncoder:
    """Featurize, batch and run one architecture."""

    def __init__(self, arch, num_classes, device, translation_range=0.9):
        assert arch in ("resnet", "pointnet")
        self.arch = arch
        self.device = device
        self.translation_range = translation_range
        if arch == "resnet":
            self.net = ResNet14(in_channels=4, out_channels=num_classes, D=3)
        else:
            self.net = MinkowskiPointNet(in_channels=4, out_channels=num_classes, dimension=3)
        self.net.to(device)

    def featurize(self, points, scaling, detector, augment=False):
        xyz, charge = points[:, :3], points[:, 3]
        if self.arch == "pointnet":
            feats = scale_points(xyz, charge, scaling, detector)
            return feats[:, :3].copy(), feats
        if augment and self.translation_range > 0:
            xyz = xyz + np.random.uniform(-self.translation_range, self.translation_range, size=(1, 3))
        coords = np.floor(xyz / VOXEL_SIZE).astype(np.int32)
        if detector == "gadget":
            coords[:, 2] = np.floor((xyz[:, 2] - xyz[:, 2].min()) / VOXEL_SIZE).astype(np.int32)
        return coords, scale_points(xyz, charge, scaling, detector)

    def to_input(self, coords, feats):
        """Batch per-event coordinates and features into a network input."""
        if self.arch == "pointnet":
            c, f = ME.utils.sparse_collate(
                [torch.from_numpy(c) for c in coords],
                [torch.from_numpy(f) for f in feats],
                dtype=torch.float32,
            )
            return ME.TensorField(coordinates=c, features=f, device=self.device)
        return ME.SparseTensor(
            torch.from_numpy(np.vstack(feats)).float(),
            ME.utils.batched_coordinates(coords),
            device=self.device,
        )

    def logits(self, x):
        return self.net(x).F

    def embed(self, x):
        return self.net.embed(x).F
