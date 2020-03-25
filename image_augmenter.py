from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import map
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import math, os, numpy as np, glob
import scipy.ndimage.interpolation
import skimage.transform as transform
from numpy import interp
from numpy.random import rand
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from .generic import *
from ku import image_utils as iu


class ImageAugmenter(object):
    """
    Provides methods to easily transform images.
    Meant for creating custom image augmentation functions for training Keras models.
    e.g. # randomly crop and flip left-right a 224x224 patch out of an image
         process_fn = lambda im: ImageAugmenter(im).crop((224,224)).fliplr().result
         # process_fn can be passed as an argument to keras.utils.Sequence objects (data generators)

    Provides various pre-defined customizable transformations, all randomizable:
    rotate, crop, fliplr, rescale, resize. The transformations can be easily chained.
    """
    def __init__(self, image, remap=True, verbose=False):
        """
        * image: image to be transformed, np.ndarray
        * remap: remap values to [0,1] for easier to apply transformations
                 these are mapped back to the initial range when .result is called
        * verbose: enable verbose prints
        """
        self._rotation_angle = 0
        self._original_range = minmax(image)
        self._remap = remap
        self.image = image if not self._remap else mapmm(image)
        self.verbose = verbose
        
    def rotate(self, angle, random=True):
        """
        Rotate self.image

        * angle: if `random` then rotation angle is a random value between [-`angle`, `angle`]
                 otherwise rotation angle is `angle`
        * random: random or by a fixed amount
        :return: self
        """
        if angle != 0 and random:
            # rotation angle is random between [-angle, angle]
            self._rotation_angle += (rand(1)-0.5)*2*angle
        else:
            self._rotation_angle += angle
            
        self.image = transform.rotate(self.image, self._rotation_angle, 
                                      resize=False, cval=0, 
                                      clip=True, preserve_range=True, 
                                      mode='symmetric')            
        return self
    
    def crop(self, crop_size, crop_pos=None, clip_rotation=False):
        """
        Crop a patch out of self.image. Relies on `extract_patch`.

        * crop_size:     dimensions of the crop (pair of H x W)
        * crop_pos:      if None, then a random crop is taken, otherwise the given `crop_pos` position is used
                         pair of relative coordinates: (0,0) = upper left corner, (1,1) = lower right corner
        * clip_rotation: clip a border around the image, such that the edge resulting from
                         having rotated the image is not visible
        :return:         self
        """
        # equally crop in both dimensions if only one number is provided
        if not isinstance(crop_size, (list, tuple)):
            crop_size = [crop_size, crop_size]
        # if using a ratio crop, compute actual crop size
        crop_size = [np.int32(c*dim) if 0 < c <= (1+1e-6) else c\
                     for c, dim in zip(crop_size, self.image.shape[:2])]
             
        if self.verbose:
            print('image_size:', self.image.shape, 'crop_size:', crop_size)

        if crop_pos is None:
            if crop_size != self.image.shape[:2]:
                if clip_rotation:
                    lrr = largest_rotated_rect(self.image.shape[0], 
                                               self.image.shape[1], 
                                               math.radians(self._rotation_angle))
                    x, y = self.image.shape, lrr
                    border = (old_div((x[0]-y[0]),2), old_div((x[1]-y[1]),2))
                else:
                    border = (0, 0)
                self.image = iu.extract_random_patch(self.image,
                                                     patch_size = crop_size, 
                                                     border     = border)
        else:
            if crop_size != self.image.shape[:2]:
                self.image = iu.extract_patch(self.image,
                                              patch_size     = crop_size,
                                              patch_position = crop_pos)
        return self
    
    def cropout(self, cropout_size, cropout_pos=None, fill_val=0):
        """
        Cropout a patch of self.image and replace it with `fill_val`. Relies on `cropout_patch`.
        
        * cropout_size: dimensions of the cropout (pair of H x W)
        * cropout_pos:  if None, then a random cropout is taken, otherwise the given `cropout_pos` position is used
                        pair of relative coordinates: (0,0) = upper left corner, (1,1) = lower right corner
        * fill_val:     value to fill the cropout with
        :return:        self
        """
        # equally cropout in both dimensions if only one number is provided
        if not isinstance(crop_size, (list, tuple)):
            crop_size = [crop_size, crop_size]
        # if using a ratio cropout, compute actual cropout size
        crop_size = [np.int32(c*dim) if 0 < c <= (1+1e-6) else c\
                     for c, dim in zip(crop_size, self.image.shape[:2])]
             
        if self.verbose:
            print('image_size:', self.image.shape, 'cropout_size:', cropout_size, 'fill_val:', fill_val)

        if cropout_pos is None:
            if cropout_size != self.image.shape[:2]:
                border = (0, 0)
                self.image = iu.cropout_random_patch(self.image,
                                                     patch_size = crop_size,
                                                     border     = border)
        else:
            if crop_size != self.image.shape[:2]:
                self.image = iu.cropout_patch(self.image,
                                              patch_size     = crop_size,
                                              patch_position = crop_pos)
        return self
    
    def fliplr(self, do=None):
        """
        Flip left-right self.image

        * do: if None, random flip, otherwise flip if do=True
        :return: self
        """
        if (do is None and rand(1) > 0.5) or do:
            self._rotation_angle = -self._rotation_angle
            self.image = np.fliplr(self.image)
        return self
    
    def rescale(self, target, proportion=1, min_dim=False):
        """
        Rescale self.image proportionally

        * target: (int) target resolution relative to the reference image resolution
                  taken to be either the height if `min_dim` else min(height, width)
                  (float) zoom level
        * proportion: modulating factor for the zoom
                      when proportion=1 target zoom is unchanged
                      when proportion=0 target zoom=1 (original size)
        * min_dim: bool
        :return: self
        """
        if isinstance(target, int): # target dimensions
            if not min_dim:
                # choose height for zoom
                zoom_target = self.image.shape[0] 
            else:
                # choose minimum dimension
                zoom_target = min(self.image.shape[0],
                                  self.image.shape[1])
            zoom = old_div(1. * target, zoom_target)
        else:
            zoom = target
        zoom = (1-proportion) + proportion*zoom
            
        self.image = transform.rescale(self.image, zoom, 
                                       preserve_range=True,
                                       mode='reflect')
        return self
    
    def resize(self, size, ensure_min=False, fit_frame=False):
        """
        Resize image to target dimensions, exact or fitting inside frame

        * size: (height, width) tuple
        * ensure_min: if true, `size` is the minimum size allowed
                      a dimension is not changed unless it is below the minimum size
        * fit_frame: size concerns the dimensions of the frame that the image is to be 
                     fitted in, while preserving its aspect ratio
        :return: self
        """
        imsz = self.image.shape[:2] # (height, width)
                
        if not fit_frame:
            # resize if needed only
            if (not ensure_min and size != imsz) or\
               (ensure_min and (imsz[0] < size[0] or imsz[1] < size[1])):
                if ensure_min:
                    size = [max(a, b) for a, b in zip(imsz, size)]
                self.image = transform.resize(self.image, size, 
                                              preserve_range=True)
        else:
            image_height, image_width = imsz
            frame_height, frame_width = size
            aspect_image = float(image_width)/image_height
            aspect_frame = float(frame_width)/frame_height
            if aspect_image > aspect_frame: # fit width
                target_width = frame_width
                target_height = frame_width / aspect_image
            else: # fit height
                target_height = frame_height
                target_width = frame_height * aspect_image

            target_width, target_height = int(round(target_width)), int(round(target_height))

            self.image = transform.resize(self.image, (target_height, target_width), 
                                          preserve_range=True)
        return self

    @property
    def result_image(self):
        return array_to_img(self.result)

    @property
    def result(self):
        """
        :return: transformed image
        """
        if self._remap:
            return mapmm(self.image, self._original_range)
        else:
            return self.image
