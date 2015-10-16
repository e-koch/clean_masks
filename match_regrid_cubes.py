
'''
Script to match coords and image shapes of 2 cubes.
Specifically, this is for creating model images of single dish data for
cleaning interferometric data.
'''

import FITS_tools as ft
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import zoom


def match_regrid(filename1, filename2, reappend_dim=True, spec_axis=None,
                 spec_slice=None, degrade_factor=(1, 1, 8, 8),
                 is_binary_mask=False, remove_hist=True,
                 save_output=False, save_name='new_img', is_huge=True):
    '''
    Input two fits filenames. The output will be the projection of file 1
    onto file 2

    Parameters
    ----------
    filename1 : str
        FITS file to regrid
    filename2 : str
        FITS file to regrid to
    reappend_dim : bool, optional
        If there's extra dimensions in the data (ie. Stokes I, etc..), add
        them back into the regridded version.
    spec_axis : int, optional
        Specify which axis is the spectral axis. Tries to find it
        automatically if none is specified.
    spec_slice : slice, optional
        Apply a slice in the spectral dimension.
    degrade_factor : tuple, optional
        Apply factor to reduce dimension by. Requires the same amount of
        elements as number of axes in the data. Uses numpy convention,
        NOT WCS.
    is_binary_mask : bool, optional
        Enable is regridding a mask.
    remove_hist : bool, optional
        Remove HISTORY from the output header.
    save_output : bool, optional
        Saves as a FITS file. When disabled, returns an hdu with the regridded
        cube and header.
    save_name : str, optional
        Name of outputted FITS file. Defaults to 'new_img'.
    is_huge : bool, optional
        Uses huge_fits.write_huge_fits to save larger-than-memory arrays.

    '''

    fits1 = fits.open(filename1)
    fits2 = fits.open(filename2)

    hdr1 = fits1[0].header
    hdr2 = fits2[0].header

    # Check that they are in the same velocity frame.
    # There currently isn't a non-CASA python method to do this, so the task
    # just raises an error.
    if hdr1['SPECSYS'] != hdr2['SPECSYS']:
        raise ValueError('Data are in different spectral reference frames.')

    if remove_hist:
        # Remove the huge CASA history
        del hdr2["HISTORY"]

    # Try finding the spectral axis
    wcs1 = WCS(hdr1)
    naxes = wcs1.wcs.naxis
    if spec_axis is None:
        spec_axis = naxes - 1 - wcs1.wcs.spec

    # Make sure slices match axes
    if hdr2['NAXIS'] != len(degrade_factor):
        raise ValueError('len of degrade_factor must match number of \
                          dimensions in '+filename2)

    slices = [slice(None, None, i) for i in degrade_factor]

    if spec_slice is not None:
        assert len(spec_slice) == 2

        step = slices[spec_axis].step

        slices[spec_axis] = \
            slice(spec_slice[0], spec_slice[1], step)

    # Assume numpy convention for axes
    new_wcs = WCS(hdr2).slice(slices)

    shape1 = fits1[0].shape
    shape2 = fits2[0].data[slices].shape
    full_shape = fits2[0].shape

    fits2.close()

    new_hdr2 = new_wcs.to_header()

    for i, s in enumerate(shape2[::-1]):
        new_hdr2['NAXIS'+str(i+1)] = s
    new_hdr2['NAXIS'] = len(shape2)

    # Do the matching
    if len(shape1) == 2:
        regrid_img = ft.hcongrid.hcongrid(fits1[0].data, fits1[0].header, hdr2)
    else:
        if is_binary_mask:
            data = fits1[0].data.astype('bool')
        else:
            data = fits1[0].data
        regrid_img = ft.regrid_cube(data, hdr1, new_hdr2)
        regrid_img = regrid_img.reshape((1,)+regrid_img.shape)

    if is_binary_mask:
        # Regridding can cause holes in the mask.
        from scipy import ndimage as nd

        vel_slice = [slice(None)] * naxes

        vel_dim = shape2[spec_axis]

        for i in range(vel_dim):
            vel_slice[spec_axis] = slice(i, i+1)

            regrid_img[vel_slice] = \
                nd.binary_closing(regrid_img[vel_slice],
                                  structure=np.ones((1, 1, 3, 3)))
            regrid_img[vel_slice] = \
                nd.binary_opening(regrid_img[vel_slice],
                                  structure=np.ones((1, 1, 3, 3)))

            regrid_img[vel_slice] = \
                nd.median_filter(regrid_img[vel_slice], 5)

        # For restoring full size if needed
        order = 0
    else:
        order = 3

    # If is_huge is enabled, create the empty FITS file
    if is_huge:
        from huge_fits.write_huge_fits import create_huge_fits
        create_huge_fits(full_shape, save_name.rstrip(".fits")+".fits")

    regrid_hdr = _regrid_header(hdr1, hdr2)

    naxis = len(regrid_img.shape)

    vel_shape = regrid_img.shape[spec_axis]

    vel_slice = [slice(None)] * naxis

    for v in np.arange(vel_shape):
        print 'Channel %s/%s' % (v+1, vel_shape)

        vel_slice[spec_axis] = slice(v, v+1)

        plane = regrid_img[vel_slice]

        restored_plane = _restore_shape(plane, degrade_factor)

        # If it's a binary mask, force to dtype '>i2' to save space
        if is_binary_mask:
            restored_plane = restored_plane.astype('>i2')

        if is_huge:
            output_fits = fits.open(save_name.rstrip(".fits")+".fits")
            output_fits[0].data[vel_slice] = restored_plane

        else:
            # Create the cube in memory
            if v == 0:
                full_cube = restored_plane
            else:
                full_cube = np.hstack((full_cube, restored_plane))

    if not is_huge:
        if save_output:
            hdu = fits.PrimaryHDU(full_cube, header=regrid_hdr)
            hdu.writeto(save_name.rstrip(".fits")+".fits")

        else:
            return fits.PrimaryHDU(full_cube, header=regrid_hdr)


def _restore_shape(plane, zoom_factor, order=3):
    '''
    Interpolates the array to the given shape.
    '''

    if ~np.isfinite(plane).any():
        bad_pix = ~np.isfinite(plane)
        plane[bad_pix] = 0

    zoom_plane = zoom(plane, zoom_factor, order=order)

    if ~np.isfinite(plane).any():
        zoom_bad_pix = zoom(bad_pix, zoom_factor, order=0)
        zoom_plane[zoom_bad_pix] = np.NaN

    return zoom_plane


def _regrid_header(header1, header2):
    '''
    Make a header for the regridded image.
    '''

    new_hdr = header1.copy()

    naxes = header1['NAXIS']

    for i in range(1, naxes+1):
        new_hdr['CRPIX'+str(i)] = header2['CRPIX'+str(i)]
        new_hdr['CTYPE'+str(i)] = header2['CTYPE'+str(i)]
        new_hdr['CRVAL'+str(i)] = header2['CRVAL'+str(i)]
        new_hdr['CDELT'+str(i)] = header2['CDELT'+str(i)]
        new_hdr['NAXIS'+str(i)] = header2['NAXIS'+str(i)]

    return new_hdr

if __name__ == '__main__':

    import sys

    file1 = str(sys.argv[1])
    file2 = str(sys.argv[2])
    save_name = str(sys.argv[3])

    is_binary_mask = sys.argv[4]
    if is_binary_mask == 'T':
        is_binary_mask = True
    else:
        is_binary_mask = False

    match_regrid(file1, file2, save_output=True, save_name=save_name,
                 is_binary_mask=is_binary_mask)  # ,spec_slice=[670, 1901])
