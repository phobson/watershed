from __future__ import division

from collections import defaultdict

import numpy
from scipy import ndimage

DISTANCE = numpy.sqrt([
    2., 1., 2.,
    1., 1., 1.,
    2., 1., 2.
])
DIR_MAP = dict(zip(range(9), [32, 64, 128, 16, -1, 1, 8, 4, 2]))
FLOWS_IN = numpy.array([2, 4, 8, 1, numpy.nan, 16, 128, 64, 32])


def _stack_neighbors(topo, *padargs, **padkwargs):
    '''Create a MxNx9 array of neighbors

    Creates a MxNx9 array where each layer represents all of the
    adjacent values at a give row/col. Input array is edge padded to
    handle the blocks on edges and corners.

    Parameters
    ----------
    topo : numpy array (MxN)
        An array represeting a digital elevation model (DEM)
    *padargs, **padkwargs : optional parameters
        Positional and keyword arguments padded to numpy.pad

    Returns
    -------
    blocked : numpy array (MxNx9)
        Array of neighboros where `blocked[row, col, :].reshape(3, 3)`
        returns the block cells adjacent to and including
        `topo[row, col]`


    '''
    padded = numpy.pad(topo, *padargs, **padkwargs)
    blocked = numpy.dstack([
        padded[0:-2, 0:-2], # upper left
        padded[0:-2, 1:-1], # upper center...
        padded[0:-2, 2:],
        padded[1:-1, 0:-2], # middle left...
        padded[1:-1, 1:-1],
        padded[1:-1, 2:],
        padded[2:, 0:-2],   # lower left ...
        padded[2:, 1:-1],
        padded[2:, 2:],
    ])

    return blocked


def _adjacent_slopes(topo):
    '''Compute the slope from each to cell to all of its neighbors

    Parameters
    ----------
    topo : numpy array (MxN)
        Elevation data

    Returns
    -------
    slope : numpy array (MxNx9)
        3-D array where the z-axis is the unraveled slope array of
        each cell's neighbors.

    Notes
    -----
    Downward slopes are represented with positive numbers.

    See Also
    --------
    watershed._stack_neighbors

    '''

    # initial array shape
    rows, cols = topo.shape

    # make 3-D array of each cell's neighbors
    blocks = _stack_neighbors(topo, mode='edge', pad_width=1)

    # change in elevation (dz/dx)
    drop = topo.reshape(rows, cols, 1) - blocks

    # Slope (dz/dx) masked to exclude uphill directions
    slope = numpy.ma.masked_less_equal(drop / DISTANCE, 0)

    return slope


def _mark_sinks(topo):
    '''Marks sink areas in a DEM

    Parameters
    ----------
    topo : numpy array
        Eelvation data

    Returns
    -------
    sink : numpy array
        Bool array. True value indicate cell is (in) a sink.

    '''

    # compute the slopes in every direction at each cell
    slope = _adjacent_slopes(topo)

    # find where this is no downward slope
    sinks = slope.data.max(axis=2) == 0

    # remove 'sinks' on the edges of the array
    sinks[(0, -1), :] = False
    sinks[:, (0, -1)] = False

    return sinks


def fill_sinks(topo, copy=True):
    '''
    Fills sink areas in a DEM with the lowest adjacent elevation

    Parameters
    ----------
    topo : numpy array
        Elevation data
    copy : bool, optional
        When True, operates on a copy of the `topo` array. Set to
        False if memory is a concern.

    Returns
    -------
    filled : numpy array
        Numpy array with all the sinks filled

    See Also
    --------
    watershed.flow_direction_d8
    watershed.trace_upstream
    watershed.flow_accumulation

    '''

    if copy:
        _topo = topo.copy()
    else:
        _topo = topo

    sinks = _mark_sinks(_topo)
    blocks = _stack_neighbors(topo, mode='edge', pad_width=1)

    # return if there are no sinks to fill
    if sinks.sum() == 0:
        return _topo
    else:
        # loop through each sink and set its elevation to that of
        # its lowest neighbor
        rows, cols = numpy.where(sinks)
        for r, c in zip(rows, cols):
            neighbors = blocks[r, c, :]
            _topo[r, c] = neighbors[neighbors > _topo[r, c]].min()

        # recursively go back and check that all the
        # sinks were filled
        return fill_sinks(_topo, copy=copy)


def _process_edges(slope, direction):
    '''Handles edges and corners of the a flow-direction array.

    When edges and corners do not flow into the interior of the
    array, they need to flow out of the array.
    '''

    # shape of the raster
    rows, cols = direction.shape

    # where no flow direction could be computed
    _r, _c =  numpy.where(slope.mask.all(axis=2))

    # no direction cells on the top row flow up
    toprow = defaultdict(lambda: 64)

    # top-row corners flow out at angles
    toprow.update({0: 32, cols-1: 128})

    # no direction cells on the bottom flow down
    bottomrow = defaultdict(lambda: 4)

    # bottom row corners
    bottomrow.update({0: 8, cols-1: 2})

    # set up the main look-up dictionary
    # non-top or bottom cells flow left or right
    missing_directions = defaultdict(lambda: {0: 16, cols-1: 1})

    # add top/bottom rows to the main dictionary
    missing_directions.update({0: toprow, rows-1: bottomrow})

    # loop through all of the cells w/o flow direction
    for r, c in zip(_r, _c):
        if r in [0, rows-1] or c in [0, cols-1]:
            direction[r, c] = missing_directions[r][c]
        else:
            # raise an error if we didn't clean up an internal sink
            msg = "internal sink at ({}, {})".format(int(r), int(c))
            raise ValueError(msg)

    return direction


def flow_direction_d8(topo):
    '''Compute the flow direction of topographic data

    Flow Directions from cell X:
     32 64 128
     16  X  1
      8  4  2

    Parameters
    ----------
    topo : numpy array
        Elevation data

    Returns
    -------
    direction : numpy array
        Flow directions as defined in the references below.

    See Also
    --------
    watershed.fill_sinks
    watershed.trace_upstream
    watershed.flow_accumulation

    References
    --------
    http://onlinelibrary.wiley.com/doi/10.1029/96WR03137/pdf

    '''
    # inital array shape
    rows, cols = topo.shape

    slope = _adjacent_slopes(topo)

    # location of the steepes slope
    index = slope.argmax(axis=2)

    direction = numpy.array([
        DIR_MAP.get(x, -1) for x in index.flat
    ]).reshape(rows, cols)
    return _process_edges(slope, direction)


def _trace_upstream(flow_dir, blocks, is_upstream, row, col):
    '''Recursively traces all cells upstream from the specified cell

    Parameters
    ----------
    flow_dir : numpy array (MxM)
        Array defining the flow direction of each cell
    blocks : numpy array (MxNx9)
        Layers of arrays for each neighbor for each cell
        (see _stack_neighbors)
    is_upstream : numpy array (MxN)
        Bool-like array where cells set to 1 (True) are upstream of the
        in the flow network
    row, col : int
        Indices of the cells from which the upstream network should be
        traced.

    Returns
    -------
    None

    Notes
    -----
     - Acts in-place on `is_upstream`
     - called by the public function `trace_upstream`

    See Also
    --------
    flow_direction_d8
    _stack_neighbors
    trace_upstream

    '''

    if is_upstream[row, col] == 0:

        # we consider a cell to be upstream of itself
        is_upstream[row, col] = 1

        # flow direction of a cell's neighbors
        neighbors = blocks[row, col, :].reshape(3, 3)

        # indices of neighbors that flow into this cell
        matches = numpy.where(neighbors == FLOWS_IN.reshape(3, 3))

        # recurse on all of the neighbors
        for rn, cn in zip(*matches):
            _trace_upstream(flow_dir, blocks, is_upstream, row+rn-1, col+cn-1)


def trace_upstream(flow_dir, row, col):
    '''Trace the upstream network from a cell based on flow direction

    Parameters
    ----------
    flow_dir : numpy array (MxM)
        Array defining the flow direction of each cell.
    row, col : int
        Indices of the cells from which the upstream network should be
        traced.

    Returns
    -------
    is_upstream : numpy array (MxN)
        Bool-like array where cells set to 1 (True) are upstream of the
        in the flow network.

    See Also
    --------
    watershed.fill_sinks
    watershed.flow_direction_d8
    watershed.flow_accumulation

    '''

    is_upstream = numpy.zeros_like(flow_dir)

    # create the neighborhoods
    blocks = _stack_neighbors(flow_dir, ((1, 1), (1, 1)), mode='constant')

    _trace_upstream(flow_dir, blocks, is_upstream, row, col)

    return is_upstream


def mask_topo(topo, row, col, zoom_factor=1, mask_upstream=False):
    '''Block out all cells that are not upstream from a specific cell

    Parameters
    ----------
    topo : numpy array (MxN)
        An array represeting a digital elevation model (DEM)
    row, col : int
        Indices of the cells from which the upstream network should be
        traced.
    zoom_factor : float, optional
        Factor by which the image should be zoomed. Should typically be
        less than 1 to effectively coursen very high resolution DEMs
        so that flat areas or depressions don't truncate the trace.
    mask_upstream : bool, optional
        If False (default) all cell *not* upstream of `topo[row, col]`
        will be maked. Otherwise, the upstream cells will be masked

    Returns
    -------
    topo_masked : numpy masked array (MxN)
        Masked array where all cells not upstream of `topo[row, col]`
        are masked out.

    '''
    # apply the zoom_factor
    _topo = ndimage.zoom(topo, zoom_factor, order=0)
    _row, _col = map(lambda x: numpy.floor(x * zoom_factor), (row, col))

    # determine the flow direction
    flow_dir = flow_direction_d8(_topo)

    # trace upstream on the zoomed DEM
    _upstream = trace_upstream(flow_dir, _row, _col)

    # unzoom the upstream mask
    upstream = ndimage.zoom(_upstream, zoom_factor**-1, order=0)

    # apply the mask
    if mask_upstream:
        return numpy.ma.masked_array(data=topo, mask=upstream)
    else:
        return numpy.ma.masked_array(data=topo, mask=numpy.logical_not(upstream))


def flow_accumulation(flow_dir):
    '''Compute the flow accumulation from flow directions

    Determines the number of cells flowing into every cell
    in an array represeting flow direction.

    Parameters
    ----------
    flow_dir : numpy array
        Array representing flow direction.

    Returns
    -------
    flow_acc : numpy array
        Array representing the flow accumulation for each
        cell in the input `flow_dir` array.

    See Also
    --------
    watershed.fill_sinks
    watershed.flow_direction_d8
    watershed.trace_upstream

    References
    ----------
    http://goo.gl/57r7SU

    '''

    # initial the output array
    flow_acc = numpy.zeros_like(flow_dir)
    for row in range(flow_acc.shape[0]):
        for col in range(flow_acc.shape[1]):
            flow_acc[row, col] = trace_upstream(flow_dir, row, col).sum() - 1

    return flow_acc
