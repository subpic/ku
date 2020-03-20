import numpy as np, pandas as pd
import multiprocessing as mp
import os, scipy, h5py, time, sys
from munch import Munch
from sklearn.model_selection import train_test_split
from scipy import stats
import model_helper as mh
import applications as apps
import tensor_ops as ops
import generic as gen
import image_utils as iu
import matplotlib.pyplot as plt

from keras.layers import Input, Dropout
from keras.models import Model

from keras import backend as K

class Ensemble(object):
    """
    Whatever
    """
    def __init__(self, ids, get_helper, ensemble_size, num_epochs=50, verbose=True, statistic='var', splits = (.95,.05), mc_dropout=False):
        """
        description tbd
        """
        self.ids = ids
        self.helper = get_helper()
        self.ensemble_size = ensemble_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.statistic = statistic
        self.splits = splits
        self.mc_dropout = mc_dropout
        
        if self.verbose:
            print("Ensemble of size " + str(self.ensemble_size) + " initiated. Training for " + str(self.num_epochs) + " epochs scheduled.")
        
    def train(self, lr=1e-4, valid_in_memory=False,
             recompile=True, verbose=True):
        """
        description tbd
        """
        ids = self.ids
        ids = ids[ids.set!='test']
        for ens in range(self.ensemble_size):
            helper = self.helper
            helper.model_name.update(ens_n=ens)
            helper.model_name.update(stat=self.statistic)
            helper.model_name.update(splits=self.splits)
            
            train_valid_gen = helper.make_generator(ids, batch_size=len(ids), shuffle=False)
            
            X, y = train_valid_gen[0]
            X, y = X[0],y[0]
            
            itrain, ivalid = train_test_split(list(range(X.shape[0])), 
                                              train_size=self.splits[0], test_size=self.splits[1], random_state=42+ens)   
            X_train, y_train = X[itrain, ...], y[itrain]
            X_valid, y_valid = X[ivalid, ...], y[ivalid]
            
            helper.model_name.splits = self.splits

            helper.train(lr=lr, epochs=self.num_epochs, recompile=recompile, verbose=verbose,
                         train_gen = (X_train, y_train), 
                         valid_gen = (X_valid, y_valid))
            helper.load_model()
            helper.train(lr=lr*0.1, epochs=self.num_epochs, recompile=recompile, verbose=verbose,
                         train_gen = (X_train, y_train), 
                         valid_gen = (X_valid, y_valid))
            
            del helper
            K.clear_session()
        
        
    def predict(self, test_gen=None, splits=False, output='MOS', output_layer=None,
               repeats=1, batch_size=None, remodel=True):
        """
        description tbd
        """
        predictions = []
        for ens in range(self.ensemble_size):
            mcrange = 1
            if self.mc_dropout:
                mcrange = 5
            output='mos_v' + str(ens)
            helper = self.helper
            helper.model_name.update(ens_n = ens)
            helper.model_name.update(stat = self.statistic)
            if helper.load_model(verbose=1):
                for mc in range(mcrange):
                    predictions.append(helper.predict(test_gen=test_gen, output_layer=output_layer,
                                                     repeats=repeats, batch_size=batch_size, remodel=remodel))
        
        K.clear_session()
        
        return np.array(predictions)
        
    def evaluate_performance(self, test_gen=None, output='MOS', output_layer=None,
                             repeats=1, batch_size=None, remodel=True,
                             statistic='var'):
        """
        description tbd
        """
        predictions = self.predict(test_gen=test_gen, output=output, output_layer=output_layer, repeats=repeats,
                              batch_size=batch_size, remodel=remodel)
        predictions = np.array(predictions)
        
        if statistic=='var':
            performance = np.var(np.array(predictions),axis=0)
        
        if self.verbose:
            plt.hist(performance, bins=25);
            
        return performance
    
    def update_splits(self, splits):
        """
        description tbd
        """
        self.splits = splits
    
    def update_ids(self, ids):
        """
        description tbd
        """
        self.ids = ids