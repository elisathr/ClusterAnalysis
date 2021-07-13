import matplotlib.pyplot as plt
import hyperspy.api as hs
import numpy as np
from scipy import ceil


def make_rgb(image, color, *args, **kwargs):
    """
    Make RGB(A) image from grayscale image with given color-scheme.

    Function to transform a grayscale image to a RGB(A) image with a certain color tint. If for instance, color is given as a red color, the returned image will be an RGB(A) image with only values in the red channel.
    :param image: Image of size (NxM). It is expected to be a numpy array, but it may also be given as a hyperspy signal.
    :param color: The color tuple to tint the image with. It should be given as either (r, g, b) or (r, g, b, a). The maximum values of the image array will be given this color, while any other value will be a scaled down version of this color
    :param args: Optional argument list.
    :param kwargs: Keyword argument list.
    :return: rgb_image. Numpy ndarray with size (NxMxC), where C is either 3 or 4 if RGB or RGBA image.

    Allowed *args are:
    `rgb` or `rgba`. If `rgb` (default), an rgb image is returned. if rgba, a rgba image is returned instead.
    `debug`. Toggles print output along the way.

    Allowed **kwargs are:
    `threshold`. A threshold value (between 0 and 1). Only values above this value is treated, and values below this value is set to numpy.nan.
    """
    if not isinstance(image, np.ndarray):
        try:
            rgb_image = make_rgb(image.data, *args, **kwargs)
        except AttributeError as e:
            if not isinstance(image, (hs.signals.Signal1D, hs.signals.Signal2D, hs.signals.BaseSignal)):
                print(
                    'Got image of type `{}` (not expected `np.ndarray`). Tried to treat it as a hyperspy signal and failed.'.format(
                        type(image)))
                raise e

    else:
        image = (image - np.nanmin(image)) / np.nanmax(image)  # normalize
        if 'rgb' in args:
            rgb = True
            rgba = False
        else:
            rgb = False
            rgba = True

        if 'threshold' in kwargs:
            threshold = kwargs.pop('threshold')
            if not isinstance(threshold, (int, float)):
                raise NotImplementedError(
                    'Value for threshold recieved as {} (`{}`). Only inputs of type `{}` or `{} are implemented'.format(
                        threshold, type(threshold), int, float))
            image = np.ma.masked_where(image < threshold, image)
            image.data[image.mask] = np.nan

        if 'debug' in args:
            debug = True
            print('This is `make_rgb`.\n\trgb: {rgb}\n\trgba: {rgba}'.format(rgb=rgb, rgba=rgba))
            print('Got image:\n\tshape: {shape}\n\tmin: {minimum}\n\t{maximum}\n\t# NaN: {nNan}.'.format(
                shape=np.shape(image), minimum=np.nanmin(image), maximum=np.nanmax(image),
                nNan=len(np.where(np.isnan(image))[1])))
        else:
            debug = False

        nx, ny = np.shape(image)  # get image size
        if rgb:
            colors = color[:-1]
        elif rgba:
            colors = color

        rgb_image = np.zeros((nx, ny, len(colors)))  # Create empty image array
        if debug:
            print('Color: {}'.format(colors))
        for i, channel in enumerate(colors):  # Loop over color channels
            if debug:
                print('\tChannel {i:.0f}: {channel}'.format(i=i, channel=channel))
            rgb_image[:, :, i] = image * channel  # Multiply normalized image with color channel
    return rgb_image


def decomp_as_rgb(factors, loadings, *args, **kwargs):
    if len(factors) != len(loadings):
        raise ValueError(
            'Number of `factors` ({N_fac}) must match number of `loadings` ({N_loa})'.format(N_fac=len(factors),
                                                                                             N_loa=len(loadings)))

    if 'rgba' in args:
        rgb_mode = 'rgba'#'rgba'
    else:
        rgb_mode = 'rgb'

    if 'components' in kwargs:
        components = kwargs.pop('components')
    else:
        components = np.arange(0, len(factors), 1)
    if np.max(components) > len(factors):
        raise ValueError(
            'Maximum component ({comp}) exceeds number of decoposed components ({n})'.format(comp=np.max(components),
                                                                                             n=len(components)))

    if np.min(components) < 0:
        raise ValueError('Minimum component ({comp}) must be positive'.format(comp=np.min(components)))

    if 'colors' in kwargs:
        colors = kwargs.pop('colors')
        if colors.N < len(components):
            raise ValueError('Number of colors ({n_colors}) is lower than number of components ({n_comp})'.format(
                n_colors=colors.N, n_comp=len(components)))
        elif colors.N > len(components):
            warn.warn('Number of colors ({n_colors}) is larger than number of components ({n_comp})'.format(
                n_colors=colors.N, n_comp=len(components)))
    else:
        colors = plt.get_cmap('jet', len(components))

    rgb_factors = [make_rgb(factors.inav[components[i]].data, colors(i), rgb_mode) * (2 ** 16 - 1) for i in
                   range(len(components))]
    rgb_loadings = [make_rgb(loadings.inav[components[i]].data, colors(i), rgb_mode) * (2 ** 16 - 1) for i in
                    range(len(components))]

    rgb_factors = hs.signals.Signal1D(rgb_factors)
    rgb_loadings = hs.signals.Signal1D(rgb_loadings)

    rgb_factors.change_dtype('uint16')
    rgb_loadings.change_dtype('uint16')

    rgb_factors.change_dtype('rgba16')
    rgb_loadings.change_dtype('rgba16')

    rgb_factors.axes_manager[0].name = 'channels'
    rgb_factors.axes_manager[1].name = factors.axes_manager[1].name
    rgb_factors.axes_manager[1].scale = factors.axes_manager[1].scale
    rgb_factors.axes_manager[1].units = factors.axes_manager[1].units

    rgb_factors.axes_manager[2].name = factors.axes_manager[2].name
    rgb_factors.axes_manager[2].scale = factors.axes_manager[2].scale
    rgb_factors.axes_manager[2].units = factors.axes_manager[2].units

    rgb_loadings.axes_manager[0].name = 'channels'
    rgb_loadings.axes_manager[1].name = loadings.axes_manager[1].name
    rgb_loadings.axes_manager[1].scale = loadings.axes_manager[1].scale
    rgb_loadings.axes_manager[1].units = loadings.axes_manager[1].units

    rgb_loadings.axes_manager[2].name = loadings.axes_manager[2].name
    rgb_loadings.axes_manager[2].scale = loadings.axes_manager[2].scale
    rgb_loadings.axes_manager[2].units = loadings.axes_manager[2].units

    return rgb_factors, rgb_loadings, components, colors



def save_component_maps(signal_decomp, factors, loadings, algorithm, output_directory, num_components,scalebar=False, saveFactorsLoadings=False):
    '''
    Saves each component and navigation map from the decomposition.
    Parameters:
        signal_decomp: The decomposed dataset
        factors: The decomposition factors = signal_decomp.get_decomposition_factors()
        loadings: The decomposition loadings = signal_decomp.get_decomposition_loadings()
        algorithm: The algorithm used
        output_directory: The directory where the images will be saved
        num_components: The number of components used in the decomposition
        scalebar: False. If True, scalebars will be included in the images. Requires matplotlib.scalebar
        saveFactorsLoadings: False. If True, the factors and loadings will be saved as hyperspy signals.
    '''
    import os
    if scalebar: # Only works if the matplotlib-scalebar package is installed.
        from matplotlib_scalebar.scalebar import ScaleBar #For adding scalebars to imshow objects
        from matplotlib_scalebar.dimension import _Dimension #For defining dimensions for scalebars
        import traits

        reciprocal_unit = '1/Å'
        reciprocal_dimension = _Dimension(reciprocal_unit)
        if type(signal_decomp.axes_manager[2].units) == traits.trait_base._Undefined:
            DP_scale = signal_decomp.metadata.Signal.calibration.diffraction
            xy_scale = signal_decomp.metadata.Signal.calibration.position
            signal_decomp.set_diffraction_calibration(DP_scale)
            signal_decomp.axes_manager[0].scale = xy_scale
            signal_decomp.axes_manager[0].units = 'nm'
                     
    try:
        os.mkdir(output_directory + '/{}_{}_components'.format(algorithm, num_components))
        output_directory_img = output_directory + '/{}_{}_components'.format(algorithm, num_components)
        print('Saving images in new directory: ', output_directory + '/{}_{}_components'.format(algorithm, num_components))
    except:
        output_directory_img = output_directory + '/{}_{}_components'.format(algorithm, num_components)
        print('Saving images in: ', output_directory + '/{}_{}_components'.format(algorithm, num_components))
    if saveFactorsLoadings:
        factors.save(output_directory_img + '/factors_{}_{}comp.hdf5'.format(algorithm, num_components))
        loadings.save(output_directory_img + '/loadings_{}_{}comp.hdf5'.format(algorithm, num_components))
    
    plt.close('all')
    pt_to_in = 1.0/72.0 #There are 72 points in one inch
    fig_height = 250 #pt
    fig_width = 2 * fig_height
    fig_height *= pt_to_in #Convert to inches
    fig_width *= pt_to_in #Convert to inches
    dpi = 600 #Resolution in output images
    font_size = 4 #Font size of plot titles in pts

    plt.ioff() #dont show plots as they are plotted

    #Go through all factors and loadings, and plot corresponding pairs into the two different axes.    
    for i, (factor, loading) in enumerate(zip(factors, loadings)):
        plt.close('all')

        #For every factor and loading, create a new figure. I tried making only one figure, and then updating the data in it, but this causes a problem with the figure sizes. This way is more robust.
        figure, axes = plt.subplots(nrows=1, ncols = 2, figsize=[fig_width, fig_height], num=i) #Create figure with two subplot axes
        axes[1].set_title(r'Factor {:.0f}'.format(i), size = font_size) #Add Title to the factor plot
        axes[0].set_title(r'Loading {:.0f}'.format(i), size = font_size) #Add Title to the loading plot

        #Remove the ticks from the axes
        for ax in axes:
            ax.set_xticks([]) 
            ax.set_yticks([])

        factor_data = factor.data #Extract the factor data (as a numpy array)
        loading_data = loading.data #Extract the loading data (as a numpy array)

        factor_data[np.isnan(factor_data)] = 0 #Set the factor data to 0 where it is NaN (helps with vizualising results better)
        loading_data[np.isnan(loading_data)] = 0 #Set the loading data to 0 where it is NaN (helps with vizualising results better)

        factor_img = axes[1].imshow(factor_data, interpolation = 'nearest') #Plot the factor
        loading_img =axes[0].imshow(loading_data, interpolation = 'nearest') #Plot the loading

        #Create scalebars. They need to be recreated everytime.
        if scalebar:
            #print(reciprocal_dimension)
            scale_bar_realSpace = ScaleBar(signal_decomp.axes_manager[0].scale, units=signal_decomp.axes_manager[0].units, location='lower left', length_fraction=0.2)
            scale_bar_DP = ScaleBar(signal_decomp.axes_manager[2].scale, dimension=reciprocal_dimension, units=reciprocal_unit, location='lower left', fixed_value=0.1)

            #Add the scalebars to the axes
            axes[1].add_artist(scale_bar_DP)
            axes[0].add_artist(scale_bar_realSpace)

        plt.tight_layout() #Adjust axes size

        print('Figure dimensions:\n\twidth: {:.3f} in\n\theight:{:.3f} in.\n'.format(figure.get_figwidth(), figure.get_figheight()))

        figure.savefig(output_directory_img + '/{}_comp_no{}.tiff'.format(algorithm, i+1), bbox_inches = 'tight', dpi = dpi) #Save the figure with minimal "deadspace", and with specified resolution

