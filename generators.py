from __future__ import print_function
from __future__ import absolute_import
from builtins import range
import numpy as np
import pandas as pd
import multiprocessing as mp
from munch import Munch

import keras

from .image_utils import *
from .generic import *
from .tensor_ops import *

# GENERATORS

class DataGeneratorDisk(keras.utils.Sequence):
    """
    Generates data for training Keras models
    - inherits from keras.utils.Sequence
    - reads images from disk and applies `process_fn` to each
    - `process_fn` needs to ensure that processed images are of the same size
    - on __getitem__() returns an ND-array containing `batch_size` images

    ARGUMENTS
    * ids (pandas.dataframe): table containing image names, and output variables
    * data_path  (string):    path of image folder
    * batch_size (int):       how many images to read at a time
    * shuffle (bool):         randomized reading order
    * process_fn (function):  function applied to each image as it is read
    * read_fn (function):     function used to read data from a file (returns numpy.array)
                              if None, image_utils.read_image() is used (default)
    * deterministic (None, int):     random seed for shuffling order
    * inputs (tuple of strings):     column names from `ids` containing image names
    * inputs_df (tuple of strings):  column names from `ids`, returns values from the DataFrame itself
    * outputs (tuple of strings):    column names from `ids`
    * verbose (bool):             logging verbosity
    * fixed_batches (bool):       only return full batches, ignore the last incomplete batch if needed
    * process_args (dictionary):  dict of corresponding `ids` columns for `inputs`
                                  containing arguments to pass to `process_fn`
    """
    def __init__(self, ids, data_path, **args):
        params_defa = Munch(ids           = ids,   data_path = data_path,
                            batch_size    = 32,    shuffle = True,
                            input_shape   = (224, 224, 3), 
                            process_fn    = None,  read_fn = None,
                            deterministic = None,  inputs=('image_name',),
                            inputs_df     = None,  outputs       = ('MOS',), 
                            verbose       = False, fixed_batches = False,
                            process_args  = {})
        check_keys_exist(args, params_defa)
        params = updated_dict(params_defa, **args)  # update only existing
        params.deterministic = {True: 42, False: None}.\
                               get(params.deterministic,
                                   params.deterministic)
        params.process_args = params.process_args or {}
        self.__dict__.update(**params)  # set all as self.<param>        
        if self.verbose>1:
            print('Initialized DataGeneratorDisk')
        self.on_epoch_end()  # initialize indexes

    def __len__(self):
        """Denotes the number of batches per epoch accounting for `fixed_batches`"""
        round_op = np.floor if self.fixed_batches else np.ceil
        return int(round_op(len(self.ids)*1. / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes_batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ids_batch = self.ids.iloc[indexes_batch].reset_index(drop=True)
        if self.verbose:
            if index%10==0: print('.', end='')
        return self._data_generation(ids_batch)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _read_data(self, ids_batch, accessor):        
        X = []
        if accessor:
            assert isinstance(accessor, (tuple, list)) or callable(accessor),\
            'Generator inputs/outputs must be of type list, tuple, or function'

            if callable(accessor):
                X = accessor(ids_batch)
            elif isinstance(accessor, (list, tuple)):
                if all(isinstance(a, (list, tuple)) for a in accessor):
                    X = [np.array(ids_batch.loc[:,a]) for a in accessor]
                else:
                    assert all(not isinstance(a, (list, tuple)) for a in accessor)
                    X = [np.array(ids_batch.loc[:,accessor])]
            else:
                 raise Exception('Wrong generator input/output specifications')

        return X
    
    def _data_generation(self, ids_batch):
        """Generates image-stack + outputs containing batch_size samples"""
        params = self
        np.random.seed(params.deterministic)
        
        y = self._read_data(ids_batch, params.outputs)        
        X_list = self._read_data(ids_batch, params.inputs_df)
            
        assert isinstance(params.inputs, (tuple, list)),\
        'Generator inputs/outputs must be of type list or tuple'

        for input_name in params.inputs:
            data = []
            # read the data from disk into a list
            for i, row in ids_batch.iterrows():
                input_data = row[input_name]
                if params.read_fn is None:
                    file_path = os.path.join(params.data_path, input_data)
                    file_data = read_image(file_path)
                else:
                    file_data = params.read_fn(input_data, params)
                data.append(file_data)

            # column name for the arguments to `process_fn`
            args_name = params.process_args.get(input_name, None)
            
            # if needed, process each image, and add to X_list (inputs list)
            if params.process_fn not in [None, False]:                
                data_list = []
                for i, row in ids_batch.iterrows():                    
                    arg = [] if args_name is None else [row[args_name]]
                    data_i = params.process_fn(data[i], *arg)
                    data_list.append(force_list(data_i))
                    
                # transpose list, sublists become batches
                data_list = zip(*data_list)
                
                # for each sublist of arrays
                data_arrays = []
                for batch_list in data_list:
                    batch_arr = np.float32(np.stack(batch_list))
                    data_arrays.append(batch_arr)
                
                X_list.extend(data_arrays)
            else:
                data_array = np.float32(np.stack(data))
                X_list.append(data_array)

        np.random.seed(None)
        return (X_list, y)


class DataGeneratorHDF5(DataGeneratorDisk):
    """
    Generates data for training Keras models
    - similar to the `DataGeneratorDisk`, but reads data instances from an HDF5 file e.g. images, features
    - inherits from `DataGeneratorDisk`, a child of keras.utils.Sequence
    - applies `process_fn` to each data instance
    - `process_fn` needs to ensure a fixed size for processed data instances
    - on __getitem__() returns an ND-array containing `batch_size` data instances

    ARGUMENTS
    * ids (pandas.dataframe):    table containing data instance names, and output variables
    * data_path  (string):       path of HDF5 file
    * batch_size (int):          how many instances to read at a time
    * shuffle (bool):            randomized reading order
    * process_fn (function):     function applied to each data instance as it is read
    * deterministic (None, int): random seed for shuffling order
    * inputs (strings tuple):    column names from `ids` containing data instance names, read from `data_path`
    * inputs_df (strings tuple): column names from `ids`, returns values from the DataFrame itself
    * outputs (strings tuple):   column names from `ids`, returns values from the DataFrame itself
    * verbose (bool):            logging verbosity
    * fixed_batches (bool):      only return full batches, ignore the last incomplete batch if needed
    * process_args (dictionary): dict of corresponding `ids` columns for `inputs`
                                 containing arguments to pass to `process_fn`
    * group_names (strings tuple): read only from specified groups, or from any if `group_names` is None
                                   `group_names` are randomly sampled from meta-groups
                                   i.e. when group_names = [[group_names_1], [group_names_2]]
    * random_group (bool): read inputs from a random group for every data instance
    """
    def __init__(self, ids, data_path, **args):
        params_defa = Munch(ids         = ids,   data_path     = data_path, deterministic = False,
                            batch_size  = 32,    shuffle       = True,      inputs        = ('image_name',),
                            inputs_df   = None,  outputs       = ('MOS',),  memory_mapped = False,
                            verbose     = False, fixed_batches = False,     random_group  = False,
                            process_fn  = None,  process_args  = None,      group_names   = None,
                            input_shape = None)

        check_keys_exist(args, params_defa)
        params = updated_dict(params_defa, **args) # update only existing       
        params.process_args  = params.process_args or {}
        params.group_names   = params.group_names or [None]
        params.deterministic = {True: 42, False: None}.\
                                get(params.deterministic,
                                    params.deterministic)
        self.__dict__.update(**params)  # set all as self.<param>

        if self.verbose>1:
            print('Initialized DataGeneratorHDF5')
        self.on_epoch_end()  # initialize indexes

    def _data_generation(self, ids_batch):
        """Generates data containing batch_size samples"""
        params = self
        np.random.seed(params.deterministic)        
        
        y = self._read_data(ids_batch, params.outputs)
        X_list = self._read_data(ids_batch, params.inputs_df)

        assert isinstance(params.inputs, (tuple, list)),\
        'Generator inputs/outputs must be of type list or tuple'
        
        if params.data_path:
            with H5Helper(params.data_path, file_mode='r',
                          memory_mapped=params.memory_mapped) as h:
                # group_names are randomly sampled from meta-groups 
                # i.e. when group_names = [[group_names1], [group_names2]]
                group_names = params.group_names
                if isinstance(group_names[0], (list, tuple)):
                    idx = np.random.randint(0, len(group_names))
                    group_names = group_names[idx]

                # get data for each input and add it to X_list
                for group_name in group_names:
                    for input_name in params.inputs:
                        # get data
                        names = ids_batch.loc[:,input_name]
                        if params.random_group:
                            data = h.read_data_random_group(names)
                        elif group_name is None:
                            data = h.read_data(names)
                        else:
                            data = h.read_data(names, group_names=[group_name])[0]
                        if data.dtype != np.float32:
                            data = data.astype(np.float32)

                        # column name for the arguments to `process_fn`
                        args_name = params.process_args.get(input_name, None)
                    
                        # add to X_list
                        if params.process_fn not in [None, False]:                        
                            data_new = None
                            for i, row in ids_batch.iterrows():
                                arg = [] if args_name is None else [row[args_name]]
                                data_i = params.process_fn(data[i,...], *arg)
                                if data_new is None:
                                    data_new = np.zeros((len(data),)+data_i.shape,
                                                       dtype=np.float32)
                                data_new[i,...] = data_i
                            X_list.append(data_new)
                        else:
                            X_list.append(data)

        np.random.seed(None)
        return (X_list, y)


class GeneratorStack(keras.utils.Sequence):
    """
    Creates an aggregator generator that feeds from multiple generators.
    """
    def __init__(self, generator_list):
        self.gens = generator_list
        
    def __len__(self):
        """Number of batches per epoch"""        
        return len(self.gens[0])

    def __getitem__(self, index):
        """Generate one batch of data"""
        X, y = [],[]
        for g in self.gens:
            X_, y_ = g[index]
            X.extend(X_)
            y.extend(y_)
        return (X, y)