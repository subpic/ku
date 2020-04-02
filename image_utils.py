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
from .image_augmenter import ImageAugmenter
import random

from keras.preprocessing.image import img_to_array, array_to_img, load_img

def view_stack(ims, figsize=(20, 20), figshape=None, 
               cmap='gray', vrange='all', **kwargs):
    """
    Display a stack or list of images using subplots.

    * ims: single np.ndarray of size [N x H x W x 3/1] or 
           list of np.ndarray(s) of size [H x W x 3/1]
           (if list, np.stack is called first)
    * figsize: plt.figure(figsize=figsize)
    * figshape: (rows, cols) of the figure
                if None, the sizes are inferred
    * cmap:   color map, defaults to 'gray'
    * vrange: remap displayed value range:
              if 'all' set a global display range for the entire stack,
              if 'each' use a different display range for each image
    * kwargs: passed to `imshow` for each image
    """
    # get number of images
    if isinstance(ims, list): n = len(ims)
    else:                     n = ims.shape[0] 
        
    if figshape is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(old_div(1.*n,cols)))
    else:
        rows, cols = figshape
    if vrange == 'all':
        if isinstance(ims, list):
            mm = list(map(minmax, ims))
            vrange = (min([p[0] for p in mm]), 
                      max([p[1] for p in mm]))
        else: 
            vrange = minmax(ims)
    elif vrange == 'each':
        vrange = (None, None)

    fig = plt.figure(figsize=figsize)
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i+1)
        if isinstance(ims, list): im = ims[i]
        else:                     im = np.squeeze(ims[i, ...])            
        ax.imshow(im, cmap=cmap, vmin=vrange[0], 
                  vmax=vrange[1], **kwargs)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
def augment_image(im, augs=None, verbose=False):
    """
    Augments an image by a given augmentation. If no augmentation is provided, 
    a random agumentation is applied within reasonable parameters.
    
    * im:      np.ndarray of size H x W x C
    * aug:     list of augmentations.
               if None, a random augmentation is applied,
               if list of length greater than 1, a random augmentation from the 
               list is applied, otherwise the given augmentation is applied.
    * verbose: enable verbose prints
    :return:   augmented image as np.ndarray of size H x W x C
    """
    augs_dict = {'crop':{'crop_size':(32,32), 'crop_pos':(0,0)},
                 'cropout':{'crop_size':(32,32), 'crop_pos':(0,0) ,'fill_val':0.5},
#                  'gblur':{}
                }
    if augs not in [None, False]:
        if len(augs) > 1:
            # pick random augmentation from given list and call 
            # itself with particular augmentation
            aug = random.choice(augs)
            if verbose:
                print('augmenting by using ', str(aug))
            return augment_image(im, augs=(aug,))
        if augs[0] in augs_dict.keys():
            # perform particular augmentation
            aug = augs[0]
            aug_params = augs_dict[augs[0]]
            
            return ImageAugmenter(im, remap=False).\
                          augment(aug, **aug_params).result
            
        elif verbose:
            print('provided augmentation "', str(aug[0]), '" not a valid augmentation.')

    else:
        # pick random augmentation
        # call itself with particular augmentation
        aug = random.choice(augs)
        if verbose:
            print('augmenting by using ', str(aug,))
        return augment_image(im, augs=(aug))
        
    return im
    
def read_image(image_path, image_size=1):
    """
    Read image from disk

    * image_path: full path to the image
    * image_size: resize image to specified size
                  can be a 2-tuple of (H, W) or a scalar zoom factor
    :return: np.ndarray
    """
    if type(image_size) == tuple:
        im = load_img(image_path, target_size=image_size) 
        x = img_to_array(im)
    else:
        im = load_img(image_path)
        x = img_to_array(im)            
        if not image_size == 1:
            new_size = list(map(int, (x.shape[0]*image_size, x.shape[1]*image_size)))        
            x = transform.resize(x/255., new_size, mode='reflect')*255.
    return x

def read_image_batch(image_paths, image_size=None, as_list=False):
    """
    Reads image array of np.uint8 and shape (num_images, *image_shape)

    * image_paths: list of image paths
    * image_size: if not None, image is resized
    * as_list: if True, return list of images, 
               else return np.ndarray (default)
    :return: np.ndarray or list
    """
    images = None
    for i, image_path in enumerate(image_paths):
        im = load_img(image_path)
        if image_size is not None:
            im = im.resize(image_size, Image.LANCZOS)
        x = img_to_array(im).astype(np.uint8)
        if images is None:
            if not as_list:
                images = np.zeros((len(image_paths),) + x.shape,
                                  dtype=np.uint8)
            else: images = []
        if not as_list: images[i, ...] = x
        else: images.append(x)
    return images

def resize_image(x, size):
    """
    Resize image using skimage.transform.resize even when range is outside [-1,1].

    * x: np.ndarray
    * size: new size (H,W)
    :return: np.ndarray
    """
    if size != x.shape[:2]:
        minx, maxx = minmax(x)
        if maxx > 1 or minx < -1:
            x = mapmm(x)
        x = transform.resize(x, size, mode='reflect')
        if maxx > 1 or minx < -1:
            x = mapmm(x, (minx, maxx))
    return x

def resize_folder(path_src, path_dst, image_size_dst=None,
                  over_write=False, format_dst='jpg', 
                  process_fn=None, jpeg_quality=95):
    """
    Resize an image folder, copying the resized images to a new destination folder.

    * path_src:       source folder path
    * path_dst:       destination folder path, created if does not exist
    * image_size_dst: optionally resize the images
    * over_write:     enable to over-write destination images
    * format_dst:     format type, defaults to 'jpg'
    * process_fn:     apply custom processing function before resizing (optional) and saving
    * jpeg_quality:   quality level if saving as a JPEG image
    :return:          list of file names that triggered an error during read/resize/write
    """
    
    image_types = ('*.jpg', '*.png', '*.bmp', '*.JPG', '*.BMP', '*.PNG')
    # index all `image_types` in source path
    file_list = []
    for imtype in image_types:
        pattern = os.path.join(path_src, imtype)
        file_list.extend(glob.glob(pattern))
    print('Found', len(file_list), 'images')
    
    try:
        os.makedirs(path_dst)
    except: pass

    print('Resizing images from', path_src, 'to', path_dst)
    
    errors = []
    for (i, file_path_src) in enumerate(file_list):
        if i % (old_div(len(file_list),20)) == 0: print(' ',i,end=' ')
        elif i % (old_div(len(file_list),1000)) == 0: print('.',end='')

        try:            
            file_name = os.path.split(file_path_src)[1]
            (file_body, file_ext) = os.path.splitext(file_name)
            
            file_name_dst = file_body + '.' + format_dst.lower()
            file_path_dst = os.path.join(path_dst, file_name_dst)

            # check that image hasn't been already processed
            if over_write or not os.path.isfile(file_path_dst): 
                im = Image.open(file_path_src)
                if process_fn is not None:
                    im = process_fn(im)
                if image_size_dst is not None:
                    if isinstance(image_size_dst, float):
                        actual_size = [int(y*image_size_dst) for y in im.size]
                    else:
                        actual_size = image_size_dst
                    imx = im.resize(actual_size, Image.LANCZOS)
                else:
                    imx = im
                if format_dst.lower() in ('jpg', 'jpeg'):
                    imx.save(file_path_dst, 'JPEG', quality=jpeg_quality)
                else:
                    imx.save(file_path_dst, format_dst.upper())
        
        except Exception as e:
            print('Error saving', file_name)
            print('Exception:', e.message)
            errors.append(file_name)
            
    return errors

def check_images(image_dir, image_types =\
                    ('*.jpg', '*.png', '*.bmp', '*.JPG', '*.BMP', '*.PNG')):
    """
    Check which images from `image_dir` fail to read.

    * image_dir:   the image directory
    * image_types: match patterns for image file extensions, defaults:
                   ('*.jpg', '*.png', '*.bmp', '*.JPG', '*.BMP', '*.PNG')
    :return:       tuple of (list of failed image names, list of all image names)
    """    
    # index all `image_types` in source path
    file_list = []
    for imtype in image_types:
        pattern = os.path.join(image_dir, imtype)
        file_list.extend(glob.glob(pattern))
    print('Found', len(file_list), 'images')
        
    image_names_err = []
    image_names_all = []
    for (i, file_path) in enumerate(file_list):
        if i % (old_div(len(file_list),20)) == 0: print(' ',i,end=' ')
        elif i % (old_div(len(file_list),1000)) == 0: print('.',end='')

        try:            
            file_dir, file_name = os.path.split(file_path)
            file_body, file_ext = os.path.splitext(file_name)
            image_names_all.append(file_name)
            load_img(file_path) # try to load
        except:
            image_names_err.append(file_name)            
    return (image_names_err, image_names_all)

def save_images_to_h5(image_path, h5_path, over_write=False,
                      batch_size=32, image_size_dst=None):
    """
    Save a folder of JPEGs to an HDF5 file. Uses `read_image_batch` and `H5Helper`.

    * image_path: path to the source image folder
    * h5_path:    path to the destination HDF5 file; created if does not exist
    * over_write: true/false
    * batch_size: number of images to read at a time
    * image_size_dst: new size of images, if not None
    """

    file_list = glob.glob(os.path.join(image_path, '*.jpg'))
    print('Found', len(file_list), 'JPG images')  
    make_dirs(h5_path)
    print('Saving images from', image_path, 'to', h5_path)
    
    with H5Helper(h5_path, over_write=over_write) as h:
        for i, batch in enumerate(chunks(file_list, batch_size)):
            if i % 10 == 0:
                print(i*batch_size, end=' ')
            image_names = [str(os.path.basename(path)) for path in batch]
            images = read_image_batch(batch, image_size=image_size_dst)
            h.write_data(images, dataset_names=image_names)            
