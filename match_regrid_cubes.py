
'''
Script to match coords and image shapes of 2 cubes.
Specifically, this is for creating model images of single dish data for
cleaning interferometric data.
'''

import FITS_tools as ft
from astropy.io import fits
import numpy as np
from reproject import reproject
from astropy.wcs import WCS


def match_regrid(filename1, filename2, reappend_dim=True,
                 degrade_factor=(1, 2, 2),
                 remove_hist=True, save_output=False, save_name='new_img'):
    '''
    Input two fits filenames. The output will be the projection of file 1
    onto file 2
    '''

    fits1 = fits.open(filename1)
    fits2 = fits.open(filename2)

    hdr1 = fits1[0].header.copy()
    hdr2 = fits2[0].header.copy()

    hdr2["CUNIT4"] = 'km/s    '
    hdr2["CRVAL4"] = -48.1391
    hdr2["CTYPE4"] = 'VELO-LSR'
    hdr2["CDELT4"] = -1.288141

    if remove_hist:
        # Remove the huge CASA history
        del hdr2["HISTORY"]

    # Make sure slices match axes
    if hdr2['NAXIS'] != len(degrade_factor):
        raise ValueError('len of degrade_factor must match number of \
                          dimensions in '+filename2)

    slices = [slice(None, None, i) for i in degrade_factor]

    # Assume numpy convention for axes
    new_wcs = WCS(hdr2).slice(slices)

    shape1 = fits1[0].data.shape
    shape2 = fits2[0].data.shape

    # Change shape to match degradation
    shape2 = tuple(i/j for i, j in zip(shape2, degrade_factor))

    fits2.close()

    # Do the matching
    if len(shape1) == 2:
        regrid_img = ft.hcongrid.hcongrid(fits1[0].data, fits1[0].header, hdr2)
    else:
        # regrid_img = ft.regrid_cube(fits1[0].data, fits1[0].header, hdr2, specaxes=(3, 3))
        regrid_img = reproject(fits1[0], hdr2, shape_out=shape2)[0]

    if save_output:
        hdu = fits.PrimaryHDU(regrid_img, header=hdr2)
        hdu.writeto(save_name+".fits")

    else:
        return fits.PrimaryHDU(regrid_img, header=hdr2)


if __name__ == '__main__':

    import sys

    file1 = str(sys.argv[1])
    file2 = str(sys.argv[2])
    save_name = str(sys.argv[3])

    match_regrid(file1, file2, save_output=True, save_name=save_name)
