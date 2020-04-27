# -*- coding: utf-8 -*-
import numpy as np
from dcase_mean import get_mean, get_std


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.ToNormalizedTensor(object),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, feat):
        feat = np.array(feat)
        for t in self.transforms:
            feat = t(feat)
        return feat

    def randomize_parameters(self):
        for t in self.transforms:
            if getattr(t, "randomize_parameters", None):
                t.randomize_parameters()

class ToNormalizedTensor(object):
    def __init__(self):
        self.mean = get_mean()
        self.std = get_std()

    def __call__(self, feat):
        feat = np.asarray(feat, np.float32)
        feat[:,:,0] = (feat[:,:,0] - self.mean[0]) / self.std[0]
        feat[:,:,1] = (feat[:,:,1] - self.mean[1]) / self.std[1]
        feat[:,:,2] = (feat[:,:,2] - self.mean[2]) / self.std[2]
        return feat
