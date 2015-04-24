import numpy

import nose.tools as nt
import numpy.testing as nptest
from scipy import ndimage

from watershed import algo


## ESRI Example Datase
## see: http://goo.gl/UeiaCi
ESRI_TOPO = numpy.array([
    [78., 72., 69., 71., 58., 49.],
    [74., 67., 56., 49., 46., 50.],
    [69., 53., 44., 37., 38., 48.],
    [64., 58., 55., 22., 31., 24.],
    [68., 61., 47., 21., 16., 19.],
    [74., 53., 34., 12., 11., 12.],
])

ESRI_FLOW_DIR_D8 = numpy.array([
    [  2,   2,   2,   4,   4,   8],
    [  2,   2,   2,   4,   4,   8],
    [  1,   1,   2,   4,   8,   4],
    [128, 128,   1,   2,   4,   8],
    [  2,   2,   1,   4,   4,   4],
    [  1,   1,   1,   1,  32,  16]
])

ESRI_UPSTREAM_HIGH = numpy.array([
    [0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

ESRI_UPSTREAM_LOW = numpy.array([
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

ESRI_FLOW_ACC = numpy.array([
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  2,  2,  0],
    [ 0,  3,  7,  5,  4,  0],
    [ 0,  0,  0, 20,  0,  1],
    [ 0,  0,  0,  1, 24,  0],
    [ 0,  2,  4,  6, 35,  2],
])


def test__stack_neighbors():
    arr = numpy.arange(6).reshape(2, 3)
    blocks = algo._stack_neighbors(arr, mode='edge', pad_width=1)

    known_blocks = numpy.array([
        [
            [0, 0, 1, 0, 0, 1, 3, 3, 4],
            [0, 1, 2, 0, 1, 2, 3, 4, 5],
            [1, 2, 2, 1, 2, 2, 4, 5, 5]
        ], [
            [0, 0, 1, 3, 3, 4, 3, 3, 4],
            [0, 1, 2, 3, 4, 5, 3, 4, 5],
            [1, 2, 2, 4, 5, 5, 4, 5, 5]
        ]
    ])

    nptest.assert_array_equal(blocks, known_blocks)


##
## Flow Direction tests
class baseFlowDir_Mixin(object):
    def test_flow_dir_d8(self):
        flow_dir = algo.flow_direction_d8(self.topo)
        nptest.assert_array_equal(flow_dir, self.known_flow_dir_d8)


class test_flow_direction_arcgis(baseFlowDir_Mixin):
    def setup(self):
        self.topo = ESRI_TOPO.copy()
        self.known_flow_dir_d8 = ESRI_FLOW_DIR_D8.copy()


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
        self.flow_dir = ESRI_FLOW_DIR_D8.copy()
        self.row_high, self.col_high = (2, 3)
        self.known_upstream_high = ESRI_UPSTREAM_HIGH.copy()

        self.row_low, self.col_low = (3, 3)
        self.known_upstream_low = ESRI_UPSTREAM_LOW.copy()


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
        self.topo = ESRI_TOPO.copy()
        self.row_high, self.col_high = (2, 3)
        self.known_upstream_high = ESRI_UPSTREAM_HIGH.copy()

        self.row_low, self.col_low = (3, 3)
        self.known_upstream_low = ESRI_UPSTREAM_LOW.copy()


class test_mask_upstream_arcgis_zoomed(baseMaskUpstream_Mixin):
    def setup(self):
        self.factor = 2.
        self.order = 0
        self.topo = ndimage.zoom(ESRI_TOPO.copy(), self.factor, order=self.order)
        self.row_high, self.col_high = (2*self.factor, 3*self.factor)
        self.known_upstream_high = ndimage.zoom(ESRI_UPSTREAM_HIGH.copy(),
                                                self.factor, order=self.order)

        self.row_low, self.col_low = (3*self.factor, 3*self.factor)
        self.known_upstream_low = ndimage.zoom(ESRI_UPSTREAM_LOW.copy(),
                                               self.factor, order=self.order)
