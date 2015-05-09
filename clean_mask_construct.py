
from spectral_cube import SpectralCube
import numpy as np
import scipy.ndimage as nd
from signal_id import RadioMask, Noise
from radio_beam import RadioBeam

'''
Routines for constructing a robust clean mask.

1) Pick two sigma levels, then dilate the higher into the lower.
2) Pick two sigma levels, remove any components in the lower cut if it
   doesn't contain any pixels in the higher cut mask.
'''



class CleanMask(object):
    """docstring for CleanMask"""
    def __init__(self, cube, low_cut, high_cut, method="dilate"):
        super(CleanMask, self).__init__()
        self.cube = cube
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.method = method

        self._low_mask = None
        self._high_mask = None

    def make_initial_masks(self):
        pass


    def to_RadioMask(self):
        return self._radio_mask

