import numpy

import nose.tools as nt
import numpy.testing as nptest

from watershed import algo


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


class base_WatershedMixin(object):
    def test_flow_dir_d8(self):
        flow_dir = algo.flow_direction_d8(self.topo)
        nptest.assert_array_equal(flow_dir, self.known_flow_dir)

    def test_trace_upstream_high(self):
        upstream = algo.trace_upstream(self.known_flow_dir, self.row_high, self.col_high)
        nptest.assert_array_equal(upstream, self.known_upstream_high)

    def test_trace_upstream_low(self):
        upstream = algo.trace_upstream(self.known_flow_dir, self.row_low, self.col_low)
        nptest.assert_array_equal(upstream, self.known_upstream_low)

    def test_mask_topo_upstream(self):
        masked_topo = algo.mask_topo(self.topo, self.row_high, self.col_high, mask_upstream=True)
        expected = numpy.ma.masked_array(data=self.topo, mask=self.known_upstream_high)
        nptest.assert_array_equal(masked_topo, expected)

    def test_mask_topo_not_upstream(self):
        masked_topo = algo.mask_topo(self.topo, self.row_high, self.col_high, mask_upstream=False)
        mask = numpy.logical_not(self.known_upstream_high)
        expected = numpy.ma.masked_array(data=self.topo, mask=mask)
        nptest.assert_array_equal(masked_topo, expected)


class test_arcgis_example(base_WatershedMixin):
    def setup(self):
        self.topo = numpy.array([
            [78., 72., 69., 71., 58., 49.],
            [74., 67., 56., 49., 46., 50.],
            [69., 53., 44., 37., 38., 48.],
            [64., 58., 55., 22., 31., 24.],
            [68., 61., 47., 21., 16., 19.],
            [74., 53., 34., 12., 11., 12.],
        ])

        self.known_flow_dir = numpy.array([
            [  2,   2,   2,   4,   4,   8],
            [  2,   2,   2,   4,   4,   8],
            [  1,   1,   2,   4,   8,   4],
            [128, 128,   1,   2,   4,   8],
            [  2,   2,   1,   4,   4,   4],
            [  1,   1,   1,   1,  32,  16]
        ])


        self.row_high, self.col_high = (2, 3)
        self.known_upstream_high = numpy.array([
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])

        self.row_low, self.col_low = (3, 3)
        self.known_upstream_low = numpy.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])

        self.known_flow_acc = numpy.array([
            [ 0,  0,  0,  0,  0,  0],
            [ 0,  1,  1,  2,  2,  0],
            [ 0,  3,  7,  5,  4,  0],
            [ 0,  0,  0, 20,  0,  1],
            [ 0,  0,  0,  1, 24,  0],
            [ 0,  2,  4,  6, 35,  2],
        ])
