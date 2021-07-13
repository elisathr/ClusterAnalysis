from scipy.ndimage import gaussian_filter
from skimage.morphology import reconstruction
from skimage.feature import  blob_log
from matplotlib.patches import Circle, Wedge, Polygon, CirclePolygon
import numpy as np
from matplotlib import pyplot as plt
import hyperspy.api as hs

def filterLocalMaxima(img_temp, threshold, sigma):
    '''
    Method for background subtraction. Inspired by the following example from scikit-image: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html
    Parameters:
        img_temp: The image to be background subtracted
        threshold: Threshold for the background subtraction routine
        sigma: Standard deviation for the Gaussian kernel
    Returns:
        hdome: The background subtracted image
    '''

    img_temp = gaussian_filter(img_temp, sigma)
    mask = img_temp

    ########### Different seed ##############
    seed = img_temp - threshold
    dilated = reconstruction(seed, mask, method='dilation')
    hdome = img_temp - dilated
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    
    yslice = 2
    
    ax1.plot(mask[yslice], '0.5', label='mask')
    ax1.plot(seed[yslice], 'k', label='seed')
    ax1.plot(dilated[yslice], 'r', label='dilated')
    #ax1.set_ylim(-0.2, 2)
    ax1.set_title('image slice')
    ax1.set_xticks([])
    ax1.legend()
    
    ax2.imshow(dilated, vmin=img_temp.min(), vmax=img_temp.max())
    ax2.axhline(yslice, color='r', alpha=0.4)
    ax2.set_title('dilated')
    ax2.axis('off')
    
    ax3.imshow(hdome)
    ax3.axhline(yslice, color='r', alpha=0.4)
    ax3.set_title('image - dilated')
    ax3.axis('off')
    
    fig.tight_layout()
    plt.show()
    ###################
    return hdome
####################################################


