from __future__ import division

import sys
import os
from pkg_resources import resource_filename

import matplotlib.image as mplimg

EXAMPLES = [
    'powell_butte',
    'so_cal'
]


def load_example(example_name):
    if example_name not in EXAMPLES:
        raise ValueError("`examples_name` must be in {}".format(EXAMPLES))

    filename = resource_filename('watershed.testing.data', example_name + '.png')
    return mplimg.imread(filename)
