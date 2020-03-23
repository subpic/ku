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

def extract_random_patch(im, patch_size=(224, 224), border=(0, 0)):
    """
    Extract a random image patch of size `patch_size`,
    with the center of the patch inside `border`

    * im: np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * border: 2-tuple of border H x W
    :return: np.ndarray
    """
    H, W, _ = im.shape
    H_crop, W_crop = patch_size
    H_crop = min(H, H_crop)
    W_crop = min(W, W_crop)    
    Y_min, X_min = border
    Y_max, X_max = (H - H_crop - Y_min, W - W_crop - X_min)
    if Y_max < Y_min: 
        Y_min = old_div((H - H_crop), 2)
        Y_max = Y_min
    if X_max < X_min:
        X_min = old_div((W - W_crop), 2)
        X_max = X_min
    Y0 = int(rand(1)*(Y_max-Y_min) + Y_min)
    X0 = int(rand(1)*(X_max-X_min) + X_min)    
    patch = im[Y0:Y0+H_crop, X0:X0+W_crop, ]
    
    return patch

def extract_patch(im, patch_size=(224, 224), 
                  patch_position=(0.5, 0.5)):
    """
    Extract a patch of size `patch_size`,
    with its center at `patch_position` expressed as a ratio of the image's H and W

    * im: np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * patch_position: 2-tuple containing patch location
                      (0,0) = upper left corner, (1,1) = lower right corner
    :return: np.ndarray
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

    patch = im[Y0:Y0+H_crop, X0:X0+W_crop, ]
    
    return patch

def cropout_random_patch(im, patch_size=(224, 224), border=(0, 0), fill_val=0):
    """
    Cropout (replace) a random patch of size `patch_size` with `fill_val`,
    with the center of the patch inside `border`

    * im: np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * border: 2-tuple of border H x W
    * fill_val: value to fill into the cropout
    :return: np.ndarray
    """
    H, W, _ = im.shape
    H_crop, W_crop = patch_size
    H_crop = min(H, H_crop)
    W_crop = min(W, W_crop)    
    Y_min, X_min = border
    Y_max, X_max = (H - H_crop - Y_min, W - W_crop - X_min)
    if Y_max < Y_min: 
        Y_min = old_div((H - H_crop), 2)
        Y_max = Y_min
    if X_max < X_min:
        X_min = old_div((W - W_crop), 2)
        X_max = X_min
    Y0 = int(rand(1)*(Y_max-Y_min) + Y_min)
    X0 = int(rand(1)*(X_max-X_min) + X_min)    
    im[Y0:Y0+H_crop, X0:X0+W_crop, ] = fill_val
    return im

def cropout_patch(im, patch_size=(224, 224),
                  patch_position=(0.5, 0.5), fill_val=0):
    """
    Cropout (replace) a patch of size `patch_size` with `fill_val`,
    with its center at `patch_position` expressed as a ratio of the image's H and W

    * im: np.ndarray of size H x W x C
    * patch_size: 2-tuple of patch H x W
    * patch_position: 2-tuple containing patch location
                      (0,0) = upper left corner, (1,1) = lower right corner
    * fill_val: value to fill into the cropout
    :return: np.ndarray
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

    im[Y0:Y0+H_crop, X0:X0+W_crop, ] = fill_val
    return im

def imdistort_image(im, dist_fn, **params):
    """
    Distort an image `im` by `dist_fn` with provided parameters.
    Requires the imdistort_python package.

    * im: np.ndarray of size H x W x C
    * dist_fn: String name of the distortion function
    :return: np.ndarray
    """
    try:
        from imdistort_python import distortions
        im = getattr(distortions, dist_fn)(im, **params)
    except ImportError:
        print("Import Error: Couldn't load imdistort_python. Could not perform distortion.")           
    
    return im

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
    
    def crop(self, crop_size, crop_pos=None, clip_rotation=False, cropout=False):
        """
        Crop a patch out of self.image. Relies on `extract_patch`.

        * crop_size: dimensions of the crop (pair of H x W)
        * crop_pos: if None, then a random crop is taken, otherwise the given `crop_pos` position is used
                    pair of relative coordinates: (0,0) = upper left corner, (1,1) = lower right corner
        * clip_rotation: clip a border around the image, such that the edge resulting from
                         having rotated the image is not visible
        :return: self
        """
        # equally crop in both dimensions if only one number is provided
        if not isinstance(crop_size, (list, tuple)):
            crop_size = [crop_size, crop_size]
        # if using a ratio crop, compute actual crop size
        crop_size = [np.int32(c*dim) if 0 < c <= (1+1e-6) else c\
                     for c, dim in zip(crop_size, self.image.shape[:2])]
             
        if self.verbose:
            print('image_size:', self.image.shape, 'crop_size:', crop_size, 'cropping out:', cropout)

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
                if cropout:
                    self.image = cropout_random_patch(self.image,
                                                      patch_size = crop_size, 
                                                      border     = border)
                else:
                    self.image = extract_random_patch(self.image,
                                                      patch_size = crop_size, 
                                                      border     = border)
        else:
            if crop_size != self.image.shape[:2]:
                if cropout:
                    self.image = cropout_patch(self.image,
                                               patch_size     = crop_size,
                                               patch_position = crop_pos)
                else:
                    self.image = extract_patch(self.image, 
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
