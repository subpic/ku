from __future__ import print_function
from __future__ import absolute_import

from ku import generic as gen
from ku import image_utils as iu
from ku import generators as gr

from munch import Munch
import pandas as pd, numpy as np
import pytest, shutil, os

def test_basic_resize_check_save_h5():
    
    # resize
    iu.resize_folder('images/', 'images_temp/',
                     image_size_dst=(50,50), over_write=True)
    image_list = iu.glob_images('images_temp', verbose=False)
    assert image_list
    ims = iu.read_image_batch(image_list)
    assert ims.shape == (4, 50, 50, 3)

    # check
    failed_images, all_images = iu.check_images('images_temp/')
    assert len(failed_images)==0
    assert len(all_images)==4

    # save to h5
    iu.save_images_to_h5('images_temp', 'images.h5', 
                         overwrite=True)
    with gr.H5Helper('images.h5') as h:
        assert list(h.hf.keys()) == sorted(all_images)
        
    # clean-up
    shutil.rmtree('images_temp')
    os.unlink('images.h5')