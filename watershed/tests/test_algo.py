import numpy
from scipy import ndimage

import nose.tools as nt
import numpy.testing as nptest

from watershed import algo
from watershed.testing import testing

class test__stack_neighbors(object):
    def setup(self):
        self.arr = numpy.arange(10, 10+6).reshape(2, 3)

        self.known_edge_blocks1 = numpy.array([
            [
                [10, 10, 11, 10, 10, 11, 13, 13, 14],
                [10, 11, 12, 10, 11, 12, 13, 14, 15],
                [11, 12, 12, 11, 12, 12, 14, 15, 15]
            ], [
                [10, 10, 11, 13, 13, 14, 13, 13, 14],
                [10, 11, 12, 13, 14, 15, 13, 14, 15],
                [11, 12, 12, 14, 15, 15, 14, 15, 15]
            ]
        ])

        self.known_edge_blocks2 = numpy.array([
            [
                [10, 10, 10, 11, 12, 10, 10, 10, 11, 12, 10, 10, 10, 11, 12, 13, 13, 13, 14, 15, 13, 13, 13, 14, 15],
                [10, 10, 11, 12, 12, 10, 10, 11, 12, 12, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 13, 13, 14, 15, 15],
                [10, 11, 12, 12, 12, 10, 11, 12, 12, 12, 10, 11, 12, 12, 12, 13, 14, 15, 15, 15, 13, 14, 15, 15, 15]
            ], [
                [10, 10, 10, 11, 12, 10, 10, 10, 11, 12, 13, 13, 13, 14, 15, 13, 13, 13, 14, 15, 13, 13, 13, 14, 15],
                [10, 10, 11, 12, 12, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 13, 13, 14, 15, 15, 13, 13, 14, 15, 15],
                [10, 11, 12, 12, 12, 10, 11, 12, 12, 12, 13, 14, 15, 15, 15, 13, 14, 15, 15, 15, 13, 14, 15, 15, 15]
            ]
        ])

        self.known_constant_blocks1 = numpy.array([
            [
                [ 0,  0,  0,  0, 10, 11,  0, 13, 14],
                [ 0,  0,  0, 10, 11, 12, 13, 14, 15],
                [ 0,  0,  0, 11, 12,  0, 14, 15,  0]
            ], [
                [ 0, 10, 11,  0, 13, 14,  0,  0,  0],
                [10, 11, 12, 13, 14, 15,  0,  0,  0],
                [11, 12,  0, 14, 15,  0,  0,  0,  0]
            ]
        ])

        self.known_constant_blocks2 = numpy.array([
            [
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0]
            ], [
                [ 0,  0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0, 10, 11, 12,  0,  0, 13, 14, 15,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0]
            ]
        ])

    def test_edge_blocks1(self):
        blocks = algo._stack_neighbors(self.arr, radius=1, mode='edge')
        nptest.assert_array_equal(blocks, self.known_edge_blocks1)

    def test_edge_blocks2(self):
        blocks = algo._stack_neighbors(self.arr, radius=2, mode='edge')
        nptest.assert_array_equal(blocks, self.known_edge_blocks2)

    def test_constant_blocks1(self):
        blocks = algo._stack_neighbors(self.arr, radius=1, mode='constant')
        nptest.assert_array_equal(blocks, self.known_constant_blocks1)

    def test_constant_blocks2(self):
        blocks = algo._stack_neighbors(self.arr, radius=2, mode='constant')
        nptest.assert_array_equal(blocks, self.known_constant_blocks2)

    @nt.raises(NotImplementedError)
    def test_bad_mode(self):
        blocks = algo._stack_neighbors(self.arr, radius=2, mode='junk')


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


##
## Fill sink tests
class baseFillSink_Mixin(object):
    def test_filler(self):
        filled = algo.fill_sinks(self.topo, copy=True)
        nptest.assert_array_equal(filled, self.known_filled)

    def test_filler_no_copy(self):
        topo = self.topo.copy()
        filled = algo.fill_sinks(topo, copy=False)
        nptest.assert_array_equal(topo, filled)

    def test_marker(self):
        sinks = algo._mark_sinks(self.topo)
        nptest.assert_array_equal(sinks, self.known_sinks)


class test_fill_sinks_single(baseFillSink_Mixin):
    def setup(self):
        self.topo = testing.basic_slope_single_sink.copy()
        self.known_sinks = self.topo == self.topo.min()
        self.known_filled = testing.basic_slope_filled.copy()


class test_fill_sinks_quad(baseFillSink_Mixin):
    def setup(self):
        self.topo = testing.basic_slope_quad_sink.copy()
        self.known_sinks = self.topo == self.topo.min()
        self.known_filled = testing.basic_slope_filled.copy()


class test_fill_sinks_big(baseFillSink_Mixin):
    def setup(self):
        self.topo = testing.basic_slope_big_sink.copy()
        self.known_sinks = self.topo == self.topo.min()
        self.known_filled = testing.basic_slope_filled_big_sink.copy()


##
## Process edges
class test__process_edges(object):
    def setup(self):
        self._slope = numpy.ones((4, 7, 1))
        self._mask = numpy.zeros_like(self._slope)

        rows = numpy.array([0, 0, 0, 1, 2, 3, 3, 3, 3])
        cols = numpy.array([0, 3, 6, 0, 6, 0, 2, 3, 6])
        self._mask[rows, cols, :] = 1

        self.direction = numpy.zeros_like(self._slope).sum(axis=2)

        self.known_direction = numpy.array([
            [32, 0, 0, 64, 0, 0, 128],
            [16, 0, 0,  0, 0, 0,   0],
            [ 0, 0, 0,  0, 0, 0,   1],
            [ 8, 0, 4,  4, 0, 0,   2]
        ])

    def test_nosink(self):
        slope = numpy.ma.masked_array(self._slope, self._mask)
        direction = algo._process_edges(slope, self.direction)
        nptest.assert_array_equal(direction, self.known_direction)

    @nt.raises(ValueError)
    def test_sink(self):
        mask = self._mask.copy()
        mask[2, 4] = 1
        slope = numpy.ma.masked_array(self._slope, mask)
        direction = algo._process_edges(slope, self.direction)


##
## Flow Direction tests
class baseFlowDir_Mixin(object):
    def test_flow_dir_d8(self):
        flow_dir = algo.flow_direction_d8(self.topo)
        nptest.assert_array_equal(flow_dir, self.known_flow_dir_d8)


class test_flow_direction_arcgis(baseFlowDir_Mixin):
    def setup(self):
        self.topo = testing.ESRI_TOPO.copy()
        self.known_flow_dir_d8 = testing.ESRI_FLOW_DIR_D8.copy()


##
## Trace Upstream Tests
class baseTraceUpstream_Mixin(object):
    def test_trace_upstream_high(self):
        upstream = algo.trace_upstream(self.flow_dir, self.row_high, self.col_high)
        nptest.assert_array_equal(upstream, self.known_upstream_high)

    def test_trace_upstream_low(self):
        upstream = algo.trace_upstream(self.flow_dir, self.row_low, self.col_low)
        nptest.assert_array_equal(upstream, self.known_upstream_low)


class test_trace_upstream_arcgis(baseTraceUpstream_Mixin):
    def setup(self):
        self.flow_dir = testing.ESRI_FLOW_DIR_D8.copy()
        self.row_high, self.col_high = (2, 3)
        self.known_upstream_high = testing.ESRI_UPSTREAM_HIGH.copy()

        self.row_low, self.col_low = (3, 3)
        self.known_upstream_low = testing.ESRI_UPSTREAM_LOW.copy()


##
## Masking Topo tests
class baseMaskUpstream_Mixin(object):
    def test_mask_topo_upstream(self):
        masked_topo = algo.mask_topo(self.topo, self.row_high, self.col_high,
                                     zoom_factor=1./self.factor, mask_upstream=True)
        expected = numpy.ma.masked_array(data=self.topo, mask=self.known_upstream_high)
        nptest.assert_array_equal(masked_topo, expected)

    def test_mask_topo_not_upstream(self):
        masked_topo = algo.mask_topo(self.topo, self.row_high, self.col_high,
                                     zoom_factor=1./self.factor, mask_upstream=False)
        mask = numpy.logical_not(self.known_upstream_high)
        expected = numpy.ma.masked_array(data=self.topo, mask=mask)
        nptest.assert_array_equal(masked_topo, expected)


class test_mask_upstream_arcgis(baseMaskUpstream_Mixin):
    def setup(self):
        self.factor = 1.
        self.order = 0
        self.topo = testing.ESRI_TOPO.copy()
        self.row_high, self.col_high = (2, 3)
        self.known_upstream_high = testing.ESRI_UPSTREAM_HIGH.copy()

        self.row_low, self.col_low = (3, 3)
        self.known_upstream_low = testing.ESRI_UPSTREAM_LOW.copy()


class test_mask_upstream_arcgis_zoomed(baseMaskUpstream_Mixin):
    def setup(self):
        self.factor = 2.
        self.order = 0
        self.topo = ndimage.zoom(testing.ESRI_TOPO.copy(), self.factor, order=self.order)
        self.row_high, self.col_high = (2*self.factor, 3*self.factor)
        self.known_upstream_high = ndimage.zoom(testing.ESRI_UPSTREAM_HIGH.copy(),
                                                self.factor, order=self.order)

        self.row_low, self.col_low = (3*self.factor, 3*self.factor)
        self.known_upstream_low = ndimage.zoom(testing.ESRI_UPSTREAM_LOW.copy(),
                                               self.factor, order=self.order)


##
## Flow Accumulation tests
class baseFlowAccumulation_Mixin(object):
    def test_flow_accumulation(self):
        flow_acc = algo.flow_accumulation(self.flow_dir)
        nptest.assert_array_equal(flow_acc, self.known_flow_acc)


class test_flow_accumulation_arcgis(baseFlowAccumulation_Mixin):
    def setup(self):
        self.flow_dir = testing.ESRI_FLOW_DIR_D8.copy()
        self.known_flow_acc = testing.ESRI_FLOW_ACC.copy()


