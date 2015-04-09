
import scipy.ndimage as nd
from warnings import warn
import numpy as np

'''
Regrid an image in the laziest way possible (ie. no transformations)
'''


def lazy_2D_regrid(image, new_dim, hdr, channel_slice=None,
                   append_dim=True, *args):
    '''
    '''

    # If a cube is inputted, specify the channel to extract
    if len(image.shape) > 2:
        if channel_slice is None:
            raise TypeError('Must specify a slice if inputting a cube!')

        img_slice = image[channel_slice]
        img_slice = img_slice.squeeze()
    else:
        img_slice = image

    # Calculate factors to 'zoom' by
    old_dim = img_slice.shape

    zoom_factor = [float(i)/float(j) for i, j in zip(new_dim, old_dim)]

    cntr_pix = [dim/2. for dim in new_dim]

    new_img = nd.zoom(img_slice, zoom_factor, *args)

    new_hdr = _3D_to_2D_hdr(hdr, zoom_factor, cntr_pix, specaxis=4, polaxis=3)

    if append_dim:
        if len(image.shape) == 2:
            pass
        elif len(image.shape) == 3:
            new_img = new_img[np.newaxis, :, :]
        elif len(image.shape) == 4:
            new_img = new_img[np.newaxis, np.newaxis, :, :]

    return new_img, new_hdr


def _3D_to_2D_hdr(header, zoom_factor, cntr_pix, specaxis=3, polaxis=None,
                  rm_axes=False):
    '''

    '''

    if isinstance(specaxis, int):
        specaxis = str(specaxis)

    if header['NAXIS'] == 4:
        if polaxis is None:
            warn("You should specify the polarization axis!"
                 "Assuming it is 4...")
            polaxis = '4'
        else:
            polaxis = str(polaxis)

    if polaxis is not None:
        del_axes = [specaxis, polaxis]
    else:
        del_axes = [specaxis]

    del_keys = ['CTYPE', 'CRVAL', 'CRPIX', 'CDELT', 'CUNIT', 'NAXIS']

    twoD_header = header.copy()

    if rm_axes:
        for key in del_keys:
            for axis in del_axes:
                del twoD_header[key+axis]
    else:
        for axis in del_axes:
            twoD_header['NAXIS'+axis] = 1

    if rm_axes:
        twoD_header['NAXIS'] = 2
        twoD_header['WCSAXES'] = 2

    twoD_header['CRPIX1'] = cntr_pix[0]
    twoD_header['CRPIX2'] = cntr_pix[1]

    twoD_header['CDELT1'] /= float(zoom_factor[0])
    twoD_header['CDELT2'] /= float(zoom_factor[1])

    return twoD_header
