from __future__ import division

import sys
import os

import matplotlib.image as mplimg

EXAMPLES = [
    'powell_butte',
    'so_cal'
]

def load_example(example_name):
    if example_name not in EXAMPLES:
        raise ValueError("`examples_name` must be in {}".format(EXAMPLES))

    filename = os.path.join(sys.prefix, 'watershed_data', 'testing', example_name + '.png')
    return mplimg.imread(filename)

