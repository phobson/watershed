import matplotlib.pyplot as plt
import matplotlib.image as image

def plot_topo(topo, **kwargs):
    '''Create a basic plot of a DEM

    Parameters
    ----------
    topo : numpy array
        Raster/DEM of topography

    kwargs : keyword arguments
        Options to be passed to plt.imshow

    Returns
    -------
    fig : matplotlib figure instance

    '''

    # create the figure and plot area
    fig, ax =  plt.subplots(figsize=(6.5, 6.5))

    # set the default color map if one isn't spec'd
    cmap = kwargs.pop('cmap', plt.cm.Blues_r)

    # plot the data and add the colorbar
    img = ax.imshow(topo, cmap=cmap, **kwargs)
    fig.colobar(img)

    return fig


#def plot_flow_direction(flow_dir, **kwargs):
