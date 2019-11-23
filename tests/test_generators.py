from __future__ import print_function
from __future__ import absolute_import
import unittest

from ku import generators as gr
from ku import generic as gen
from munch import Munch
import pandas as pd, numpy as np

gen_params = Munch(batch_size    = 2,
                   data_path     = 'images',
                   input_shape   = (224,224,3),
                   inputs        = ['filename'],
                   outputs       = ['score'],
                   fixed_batches = True)

ids = pd.read_csv('ids.csv', encoding='latin-1')

class TestDataGeneratorDisk(unittest.TestCase):
    
    def test_correct_df_input(self):
        self.assertTrue(np.all(ids.columns == ['filename', 'score']))
        self.assertTrue(np.all(ids.score == range(1,5)))
        
    def test_init_generator(self):          
        g = gr.DataGeneratorDisk(ids, shuffle=False, **gen_params)
        self.assertIsInstance(g[0], tuple)
        self.assertIsInstance(g[0][0], list)
        self.assertIsInstance(g[0][1], list)
        self.assertEqual(gen.get_sizes(g[0]),'([array<2,224,224,3>], [array<2,1>])')
        self.assertTrue(np.all(g[0][1][0] == np.array([[1],[2]])))
        
    def test_get_sizes(self):
        x = np.array([[1,2,3]])
        self.assertEqual(gen.get_sizes(([x.T],1,[4,5])), '([array<3,1>], <1>, [<1>, <1>])')
        self.assertEqual(gen.get_sizes(np.array([[1,[1,2]]])), 'array<1,2>')
    
if __name__ == '__main__':
    unittest.main()