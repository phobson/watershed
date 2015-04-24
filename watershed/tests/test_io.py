import numpy

import nose.tools as nt
import numpy.testing as nptest

from watershed import io

class test_load_example(object):
    def test_good(self):
        data = io.load_example('powell_butte')
        nt.assert_true(isinstance(data, numpy.ndarray))

    @nt.raises(ValueError)
    def test_bad(self):
        data = io.load_example('junk')
