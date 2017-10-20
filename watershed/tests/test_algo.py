import numpy
from scipy import ndimage

import pytest
import numpy.testing as nptest

from watershed import algo
from watershed.testing import raises


@pytest.fixture
def expected_stacks():
    expected = {
        (1, 'edge'): numpy.array([
            [
                [10, 10, 11, 10, 10, 11, 13, 13, 14],
                [10, 11, 12, 10, 11, 12, 13, 14, 15],
                [11, 12, 12, 11, 12, 12, 14, 15, 15]
            ], [
                [10, 10, 11, 13, 13, 14, 13, 13, 14],
                [10, 11, 12, 13, 14, 15, 13, 14, 15],
                [11, 12, 12, 14, 15, 15, 14, 15, 15]
            ]
        ]),
        (2, 'edge'): numpy.array([
            [
                [10, 10, 10, 11, 12, 10, 10, 10, 11, 12, 10, 10, 10, 11, 12, 13, 13, 13, 14, 15, 13, 13, 13, 14, 15],
                [10, 10, 11, 12, 12, 10, 10, 11, 12, 12, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 13, 13, 14, 15, 15],
                [10, 11, 12, 12, 12, 10, 11, 12, 12, 12, 10, 11, 12, 12, 12, 13, 14, 15, 15, 15, 13, 14, 15, 15, 15]
            ], [
                [10, 10, 10, 11, 12, 10, 10, 10, 11, 12, 13, 13, 13, 14, 15, 13, 13, 13, 14, 15, 13, 13, 13, 14, 15],
                [10, 10, 11, 12, 12, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 13, 13, 14, 15, 15, 13, 13, 14, 15, 15],
                [10, 11, 12, 12, 12, 10, 11, 12, 12, 12, 13, 14, 15, 15, 15, 13, 14, 15, 15, 15, 13, 14, 15, 15, 15]
            ]
        ]),
        (1, 'constant'): numpy.array([
            [
                [ 0,  0,  0,  0, 10, 11,  0, 13, 14],
                [ 0,  0,  0, 10, 11, 12, 13, 14, 15],
                [ 0,  0,  0, 11, 12,  0, 14, 15,  0]
            ], [
                [ 0, 10, 11,  0, 13, 14,  0,  0,  0],
                [10, 11, 12, 13, 14, 15,  0,  0,  0],
                [11, 12,  0, 14, 15,  0,  0,  0,  0]
            ]
        ]),
        (2, 'constant'): numpy.array([
            [
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0]
            ], [
                [ 0,  0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0]
            ]
        ]),
    }
    return expected


# sinks and stuff
@pytest.fixture
def basic_slope_single_sink():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 10, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.fixture
def basic_slope_quad_sink():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 10, 10, 12, 11],
        [20, 18, 10, 10, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.fixture
def basic_slope_filled():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.fixture
def basic_slope_big_sink():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 10, 10, 10, 10, 11],
        [20, 10, 10, 10, 10, 11],
        [20, 10, 10, 10, 10, 11],
        [20, 10, 10, 10, 10, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.fixture
def basic_slope_filled_big_sink():
    return numpy.array([
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 20, 16, 14, 12, 11],
        [20, 20, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
        [20, 18, 16, 14, 12, 11],
    ])


@pytest.mark.parametrize('radius', [1, 2])
@pytest.mark.parametrize(('mode', 'error'), [
    ('edge', None),
    ('constant', None),
    ('junk', NotImplementedError)
])
def test__stack_neighbors(radius, mode, expected_stacks, error):
    arr = numpy.arange(10, 10 + 6).reshape(2, 3)
    with raises(error):
        blocks = algo._stack_neighbors(arr, radius=radius, mode=mode)
        nptest.assert_array_equal(blocks, expected_stacks[(radius, mode)])


def test__adjacent_slopes():
    known_slope_data = numpy.array([
        [
            [ 0.   ,  0.   ,  1.414,  0.   ,  0.   ,  2.   , -1.414, -2.   , 1.414],
            [-1.414,  0.   ,  0.707, -2.   ,  0.   ,  1.   , -2.828,  0.   ,-2.121],
            [-0.707,  0.   ,  0.707, -1.   ,  0.   ,  1.   , -0.707, -4.   ,-1.414],
            [-0.707,  0.   ,  0.   , -1.   ,  0.   ,  0.   , -3.536, -3.   ,-2.121]
        ], [
            [ 1.414,  2.   ,  2.828,  0.   ,  0.   ,  4.   , -1.414, -2.   , 1.414],
            [-1.414,  0.   ,  0.707, -4.   ,  0.   , -3.   , -4.243, -2.   ,-1.414],
            [ 2.121,  4.   ,  3.536,  3.   ,  0.   ,  2.   ,  0.707,  1.   , 2.121],
            [ 1.414,  3.   ,  2.121, -2.   ,  0.   ,  0.   , -0.707,  1.   , 0.707]
        ], [
            [ 1.414,  2.   ,  4.243,  0.   ,  0.   ,  4.   , -1.414, -2.   , 0.   ],
            [-1.414,  2.   , -0.707, -4.   ,  0.   ,  0.   , -4.243, -4.   ,-1.414],
            [ 1.414, -1.   ,  0.707,  0.   ,  0.   ,  2.   , -2.828, -2.   , 0.   ],
            [-2.121, -1.   , -0.707, -2.   ,  0.   ,  0.   , -2.828, -2.   ,-1.414]
        ], [
            [ 1.414,  2.   ,  4.243,  0.   ,  0.   ,  2.   ,  0.   ,  0.   , 1.414],
            [ 0.   ,  4.   ,  2.828, -2.   ,  0.   ,  2.   , -1.414,  0.   , 1.414],
            [ 1.414,  2.   ,  2.828, -2.   ,  0.   ,  2.   , -1.414,  0.   , 1.414],
            [ 0.   ,  2.   ,  1.414, -2.   ,  0.   ,  0.   , -1.414,  0.   , 0.   ]
        ]
    ])

    known_slope_mask = numpy.array([
        [
            [ True,  True, False,  True,  True, False,  True,  True, False],
            [ True,  True, False,  True,  True, False,  True,  True,  True],
            [ True,  True, False,  True,  True, False,  True,  True,  True],
            [ True,  True,  True,  True,  True,  True,  True,  True,  True]
        ], [
            [False, False, False,  True,  True, False,  True,  True, False],
            [ True,  True, False,  True,  True,  True,  True,  True,  True],
            [False, False, False, False,  True, False, False, False, False],
            [False, False, False,  True,  True,  True,  True, False, False]
        ], [
            [False, False, False,  True,  True, False,  True,  True,  True],
            [ True, False,  True,  True,  True,  True,  True,  True,  True],
            [False,  True, False,  True,  True, False,  True,  True,  True],
            [ True,  True,  True,  True,  True,  True,  True,  True,  True]
        ], [
            [False, False, False,  True,  True, False,  True,  True, False],
            [ True, False, False,  True,  True, False,  True,  True, False],
            [False, False, False,  True,  True, False,  True,  True, False],
            [ True, False, False,  True,  True,  True,  True,  True,  True]
        ]
    ])

    known_slopes = numpy.ma.masked_array(data=known_slope_data, mask=known_slope_mask)

    topo = numpy.array([
        [14, 12, 11, 10],
        [16, 12, 15, 13],
        [18, 14, 14, 12],
        [20, 18, 16, 14]
    ])

    slopes = algo._adjacent_slopes(topo)
    nptest.assert_array_almost_equal(slopes, known_slopes, decimal=3)


# Mark and Fill sink tests
@pytest.mark.parametrize('topo', [
    basic_slope_single_sink(),
    basic_slope_quad_sink(),
    basic_slope_big_sink(),
])
def test__mark_sinks(topo):
    sinks = algo._mark_sinks(topo)
    expected = topo == topo.min()
    nptest.assert_array_equal(sinks, expected)


@pytest.mark.parametrize(('topo', 'expected'), [
    (basic_slope_single_sink(), basic_slope_filled()),
    (basic_slope_quad_sink(), basic_slope_filled()),
    (basic_slope_big_sink(), basic_slope_filled_big_sink())
])
def test_fill_sinks_filler(topo, expected):
    filled = algo.fill_sinks(topo, copy=True)
    nptest.assert_array_equal(filled, expected)


@pytest.mark.parametrize('topo', [
    basic_slope_single_sink(),
    basic_slope_quad_sink(),
    basic_slope_big_sink()
])
def test_fille_sinks_no_copy(topo):
    topo = topo.copy()
    filled = algo.fill_sinks(topo, copy=False)
    nptest.assert_array_equal(topo, filled)


# Process edges
@pytest.mark.parametrize(('sink', 'error'), [
    (False, None),
    (True, ValueError)
])
def test__process_edges(sink, error):
    slope = numpy.ones((4, 7, 1))
    mask = numpy.zeros_like(slope)

    rows = numpy.array([0, 0, 0, 1, 2, 3, 3, 3, 3])
    cols = numpy.array([0, 3, 6, 0, 6, 0, 2, 3, 6])
    mask[rows, cols, :] = 1

    if sink:
        mask[2, 4] = 1

    slope = numpy.ma.masked_array(slope, mask)

    raw_direction = numpy.zeros_like(slope).sum(axis=2)
    expected_direction = numpy.array([
        [32, 0, 0, 64, 0, 0, 128],
        [16, 0, 0,  0, 0, 0,   0],
        [ 0, 0, 0,  0, 0, 0,   1],
        [ 8, 0, 4,  4, 0, 0,   2]
    ])
    with raises(error):
        direction = algo._process_edges(slope, raw_direction)
        nptest.assert_array_equal(direction, expected_direction)


# Flow Direction tests
def test_flow_direction(ESRI_TOPO, ESRI_FLOW_DIR_D8):
    flow_dir = algo.flow_direction_d8(ESRI_TOPO)
    nptest.assert_array_equal(flow_dir, ESRI_FLOW_DIR_D8)


# Trace Upstream Tests
@pytest.mark.parametrize(('row', 'col'), [(2, 3), (3, 3)])
def test_trace_upstream(row, col, ESRI_FLOW_DIR_D8, ESRI_UPSTREAM_LOW, ESRI_UPSTREAM_HIGH):
    expected_upstream = {
        (2, 3): ESRI_UPSTREAM_HIGH,
        (3, 3): ESRI_UPSTREAM_LOW,
    }
    upstream = algo.trace_upstream(ESRI_FLOW_DIR_D8, row, col)
    expected = expected_upstream[(row, col)]
    nptest.assert_array_equal(upstream, expected)


# Masking Topo tests
@pytest.mark.parametrize('upstream', [True, False])
@pytest.mark.parametrize(('row', 'col'), [(2, 3), (3, 3)])
@pytest.mark.parametrize('factor', [1, 2])
def test_mask_upstream(row, col, upstream, factor, ESRI_TOPO,
                       ESRI_UPSTREAM_LOW, ESRI_UPSTREAM_HIGH):
    expected_upstream = {
        (2, 3): ESRI_UPSTREAM_HIGH,
        (3, 3): ESRI_UPSTREAM_LOW,
    }

    masked_topo = algo.mask_topo(ESRI_TOPO, row // factor, col // factor,
                                 zoom_factor=1. / factor,
                                 mask_upstream=upstream)

    expected_mask = expected_upstream[(row, col)]
    if not upstream:
        expected_mask = numpy.logical_not(expected_mask)

    expected = numpy.ma.masked_array(data=ESRI_TOPO, mask=expected_mask)
    nptest.assert_array_equal(masked_topo, expected)


# Flow Accumulation tests
def test_flow_accumulation_arcgis(ESRI_FLOW_DIR_D8, ESRI_FLOW_ACC):
    flow_acc = algo.flow_accumulation(ESRI_FLOW_DIR_D8)
    nptest.assert_array_equal(flow_acc, ESRI_FLOW_ACC)
