from __future__ import print_function
from __future__ import absolute_import
from . import generic, tensor_ops, image_utils
from . import generators, model_helper, applications
    
# from tensorflow.keras import backend as K
# K.tf.compat.v1.logging.set_verbosity(K.tf.compat.v1.logging.ERROR)
# if K.backend()=='tensorflow': 
#     K.set_image_dim_ordering("tf")

# remove tensorflow warning
import logging
class WarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        tf_warning = 'retry (from tensorflow.contrib.learn.python.learn.datasets.base)' in msg
        return not tf_warning           
logger = logging.getLogger('tensorflow')
logger.addFilter(WarningFilter())

# if too many warnings from scikit-image 
import warnings
warnings.filterwarnings("ignore")

print('Loaded keras utilities')