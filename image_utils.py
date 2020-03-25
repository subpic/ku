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

def get_patch_dims(im, patch_size, patch_position):
    """
    Returns the dimensions of an image patch of size `patch_size`,
    with its center at `patch_position` expressed as a ratio of the image's H and W
    
    * im:             np.ndarray of size H x W x C
    * patch_size:     2-tuple of patch H x W
    * patch_position: 2-tuple containing patch location
                      (0,0) = upper left corner, (1,1) = lower right corner
    :return:          tuple of (upper left corner X coordinate, 
                                upper left corner Y coordinate,
                                lower right corner X coordinate,
                                lower right corner Y coordinate)
    """
    Py, Px         = patch_position
    H, W, _        = im.shape
    H_crop, W_crop = patch_size
    
    H_crop, W_crop = min(H, H_crop), min(W, W_crop)
    Y_max, X_max   = (H - H_crop, W - W_crop)
    Yc, Xc         = H*Py, W*Px

    X0, Y0 = Xc-old_div(W_crop,2), Yc-old_div(H_crop,2)
    X0, Y0 = min(max(int(X0), 0), X_max),\
             min(max(int(Y0), 0), Y_max)
    
    return (X0, Y0, X0+W_crop, Y0+H_crop)

def get_random_patch_dims(im, patch_size, border):
    """
    Returns the dimensions of a random image patch of size `patch_size`,
    with the center of the patch inside `border`
    
    * im:         np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * border:     2-tuple of border H x W
                  (0,0) = upper left corner, (1,1) = lower right corner
    :return:      tuple of (upper left corner X coordinate, 
                            upper left corner Y coordinate,
                            lower right corner X coordinate,
                            lower right corner Y coordinate)
    """
    H, W, _        = im.shape
    H_crop, W_crop = patch_size
    
    H_crop, W_crop = min(H, H_crop), min(W, W_crop)    
    Y_min, X_min   = border
    Y_max, X_max   = (H - H_crop - Y_min, W - W_crop - X_min)
    
    if Y_max < Y_min: 
        Y_min = old_div((H - H_crop), 2)
        Y_max = Y_min
    
    if X_max < X_min:
        X_min = old_div((W - W_crop), 2)
        X_max = X_min
    
    Y0 = int(rand(1)*(Y_max-Y_min) + Y_min)
    X0 = int(rand(1)*(X_max-X_min) + X_min)  
    
    return (X0, Y0, X0+W_crop, Y0+H_crop)

def extract_random_patch(im, patch_size=(224, 224), border=(0, 0)):
    """
    Extract a random image patch of size `patch_size`,
    with the center of the patch inside `border`

    * im:         np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * border:     2-tuple of border H x W
    :return:      np.ndarray
    """
    (X0, Y0, X1, Y1) = get_random_patch_dims(im, patch_size, border)
    
    patch = im[Y0:Y1, X0:X1, ]
    
    return patch

def extract_patch(im, patch_size=(224, 224), 
                  patch_position=(0.5, 0.5)):
    """
    Extract a patch of size `patch_size`,
    with its center at `patch_position` expressed as a ratio of the image's H and W

    * im:             np.ndarray of size H x W x C
    * patch_size:     2-tuple of patch H x W
    * patch_position: 2-tuple containing patch location
                      (0,0) = upper left corner, (1,1) = lower right corner
    :return:          np.ndarray
    """
    (X0, Y0, X1, Y1) = get_patch_dims(im, patch_size, patch_position)

    patch = im[Y0:Y1, X0:X1, ]
    
    return patch

def cropout_random_patch(im, patch_size=(224, 224), border=(0, 0), fill_val=0):
    """
    Cropout (replace) a random patch of size `patch_size` with `fill_val`,
    with the center of the patch inside `border`

    * im:         np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * border:     2-tuple of border H x W
    * fill_val:   value to fill into the cropout
    :return:      np.ndarray
    """
    (X0, Y0, X1, Y1) = get_random_patch_dims(im, patch_size, border)
    
    im[Y0:Y1, X0:X1, ] = fill_val
    
    return im

def cropout_patch(im, patch_size=(224, 224),
                  patch_position=(0.5, 0.5), fill_val=0):
    """
    Cropout (replace) a patch of size `patch_size` with `fill_val`,
    with its center at `patch_position` expressed as a ratio of the image's H and W

    * im:             np.ndarray of size H x W x C
    * patch_size:     2-tuple of patch H x W
    * patch_position: 2-tuple containing patch location
                      (0,0) = upper left corner, (1,1) = lower right corner
    * fill_val:       value to fill into the cropout
    :return:          np.ndarray
    """
    (X0, Y0, X1, Y1) = get_patch_dims(im, patch_size, patch_position)

    im[Y0:Y1, X0:X1, ] = fill_val
    return im

def resize_image(x, size):
    """
    Resize image using skimage.transform.resize even when range is outside [-1,1].

    * x:     np.ndarray
    * size:  new size (H,W)
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

# modified from stackoverflow
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(old_div(angle, (old_div(math.pi, 2))))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = old_div(d * math.sin(alpha), math.sin(delta))

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )