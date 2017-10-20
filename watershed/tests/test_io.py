import numpy

import pytest
import numpy.testing as nptest

from watershed import io
from watershed.testing import raises


@pytest.mark.parametrize(('file', 'error'), [
    ('powell_butte', None),
    ('junk', ValueError)
])
def test_load_example(file, error):
    with raises(error):
        data = io.load_example(file)
        assert isinstance(data, numpy.ndarray)
