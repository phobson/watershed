from __future__ import division

import numpy
from scipy import ndimage

DISTANCE = numpy.sqrt([
    2., 1., 2.,
    1., 1., 1.,
    2., 1., 2.
])

DIR_MAP = dict(zip(range(9), [32, 64, 128, 16, 0, 1, 8, 4, 2]))
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


def flow_direction_d8(topo):
    '''Compute the flow direction of topographic data

    Parameters
    ----------
    topo : numpy array
        Elevation data

    Returns
    -------
    direction : numpy array
        Flow directions as defined in
        http://hydrology.usu.edu/taudem/taudem5/help/D8FlowDirections.html

    See Also
    --------
    http://onlinelibrary.wiley.com/doi/10.1029/96WR03137/pdf

    '''
    # inital array shape
    rows, cols = topo.shape

    blocks = _stack_neighbors(topo, mode='edge', pad_width=1)

    # change in elevation (dz/dx)
    drop = topo.reshape(rows, cols, 1) - blocks

    # Slope (dz/dx) masked to exclude uphill directions
    slope = numpy.ma.masked_less_equal(drop / DISTANCE, 0)

    # location of the steepes slope
    index = slope.argmax(axis=2)

    # map in the values to the flow directions
    direction = numpy.array([DIR_MAP.get(x) for x in index.flat]).reshape(rows, cols)
    return direction


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

        # we consider a cell tp be upstream of itself
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
    flow_direction_d8

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
