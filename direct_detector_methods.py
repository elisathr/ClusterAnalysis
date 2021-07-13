import numpy as np
def remove_negative_pixels(dp, detector_size=128*128):
    # When using the EMPAD, we subtract the background because it contains dark current. Unfortunately, when doing this
    # some of the pixels become negative, which is unphysical. Thus we force those pixels to take a nonzero positive number.
    # Do not set the negative pixels to 0 because it will cause trouble with the binary mask created at a later stage.
    mask = np.nonzero(dp.inav[:].data <= 0)
    dp.inav[:].data[mask] = 0.1 
    
    number_of_negatives = np.count_nonzero(mask)
    number_of_pixels = len(dp.inav[:])
    number_of_pixels=number_of_pixels*detector_size

    print('Succesfully removed {} negative pixels. This corresponds to {}% of the total number of pixels'.format(number_of_negatives,number_of_negatives/number_of_pixels))