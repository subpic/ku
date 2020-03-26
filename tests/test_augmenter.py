from __future__ import print_function
from __future__ import absolute_import
from ku import generic as gen
from ku import image_augmenter as aug
from munch import Munch
import pandas as pd, numpy as np
import pytest

def test_imageutils_exposes_augmenter():
    from ku import image_utils as iu
    assert isinstance(iu.ImageAugmenter(np.ones(1)), aug.ImageAugmenter)

def test_cropout_and_crop():
    m = np.zeros((5,5,3))
    c = np.zeros((5,5,3))
    c[1:4,1:4,:] = 1
    
    # cropout
    assert np.array_equal(aug.cropout_patch(m, patch_size=(3,3), patch_position=(0.5,0.5), fill_val=1), c)
    assert np.array_equal(aug.ImageAugmenter(c).cropout((3,3), crop_pos=(0.5,0.5), fill_val=1).result, c)
    assert np.array_equal(aug.ImageAugmenter(c).cropout((3,3), crop_pos=(0.5,0.5), fill_val=0).result, m)

    # crop
    assert np.array_equal(aug.ImageAugmenter(c).crop((3,3), crop_pos=(0.5,0.5)).result, np.ones((3,3,3)))

def test_flip():
    m = np.zeros((5,5,3))
    ml, mr = [m]*2
    ml[0:2,0:2,:] = 1
    mr[0:2,-2:,:] = 1

    assert np.array_equal(aug.ImageAugmenter(m).fliplr().result, m)
    assert np.array_equal(aug.ImageAugmenter(ml).fliplr().result, mr)
