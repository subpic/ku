from __future__ import print_function
from __future__ import absolute_import
from ku import generators as gr
from ku import generic as gen
from munch import Munch
import pandas as pd, numpy as np
import pytest

gen_params = Munch(batch_size    = 2,
                   data_path     = 'images',
                   input_shape   = (224,224,3),
                   inputs        = ['filename'],
                   outputs       = ['score'],
                   shuffle       = False,
                   fixed_batches = True)

ids = pd.read_csv('ids.csv', encoding='latin-1')

def test_correct_df_input():
    assert (np.all(ids.columns == ['filename', 'score']))
    assert (np.all(ids.score == range(1,5)))

def test_init_generator():          
    g = gr.DataGeneratorDisk(ids, **gen_params)
    assert isinstance(g[0], tuple)
    assert isinstance(g[0][0], list)
    assert isinstance(g[0][1], list)
    assert (gen.get_sizes(g[0]) == '([array<2,224,224,3>], [array<2,1>])')
    assert (np.all(g[0][1][0] == np.array([[1],[2]])))

def test_get_sizes():
    x = np.array([[1,2,3]])
    assert gen.get_sizes(([x.T],1,[4,5])) == '([array<3,1>], <1>, [<1>, <1>])'
    assert gen.get_sizes(np.array([[1,[1,2]]])) == 'array<1,2>'
