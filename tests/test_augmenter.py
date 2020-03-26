from __future__ import print_function
from __future__ import absolute_import
from ku import generic as gen
from ku import image_augmenter as aug
from munch import Munch
import pandas as pd, numpy as np
import pytest

def test_cropout():
    m = np.zeros((5,5,3))
    c = np.zeros((5,5,3))
    c[1:4,1:4,:] = 255
    
    assert np.array_equal(aug.cropout_patch(m, patch_size=(3,3), patch_position=(0.5,0.5), fill_val=255), c)
