import numpy

import pytest


## ESRI Example Dataset
# see: http://goo.gl/qadMGm and
# http://goo.gl/57r7SU
@pytest.mark.fixture(scope='module')
def ESRI_TOPO():
    return numpy.array([
        [78., 72., 69., 71., 58., 49.],
        [74., 67., 56., 49., 46., 50.],
        [69., 53., 44., 37., 38., 48.],
        [64., 58., 55., 22., 31., 24.],
        [68., 61., 47., 21., 16., 19.],
        [74., 53., 34., 12., 11., 12.],
    ])

@pytest.mark.fixture(scope='module')
def ESRI_FLOW_DIR_D8():
    return numpy.array([
        [  2,   2,   2,   4,   4,   8],
        [  2,   2,   2,   4,   4,   8],
        [  1,   1,   2,   4,   8,   4],
        [128, 128,   1,   2,   4,   8],
        [  2,   2,   1,   4,   4,   4],
        [  1,   1,   1,   1,   4,  16]
    ])

@pytest.mark.fixture(scope='module')
def ESRI_FLOW_ACC():
    return numpy.array([
        [ 0,  0,  0,  0,  0,  0],
        [ 0,  1,  1,  2,  2,  0],
        [ 0,  3,  7,  5,  4,  0],
        [ 0,  0,  0, 20,  0,  1],
        [ 0,  0,  0,  1, 24,  0],
        [ 0,  2,  4,  7, 35,  1],
    ])


@pytest.mark.fixture(scope='module')
def ESRI_UPSTREAM_HIGH():
    return numpy.array([
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])


@pytest.mark.fixture(scope='module')
def ESRI_UPSTREAM_LOW():
    return numpy.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])



## sinks and stuff
@pytest.mark.fixture(scope='module')
def basic_slope_single_sink():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 10, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.mark.fixture(scope='module')
def basic_slope_quad_sink():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 10, 10, 12, 11],
        [20, 18, 10, 10, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.mark.fixture(scope='module')
def basic_slope_filled():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.mark.fixture(scope='module')
def basic_slope_big_sink():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 10, 10, 10, 10, 11],
        [20, 10, 10, 10, 10, 11],
        [20, 10, 10, 10, 10, 11],
        [20, 10, 10, 10, 10, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.mark.fixture(scope='module')
def basic_slope_filled_big_sink():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 20, 16, 14, 12, 11],
        [20, 20, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
    ])

