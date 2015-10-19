
'''
Save massive FITS files.

I keep another version in https://github.com/e-koch/ewky_scripts, but the gist
avoids you having to download a bunch of other stuff.
'''

from astropy.io import fits
import numpy as np
import os


def create_huge_fits(shape, filename, header=None, nblocks=4):
    '''
    Creates a massive empty FITS file that can be then written to
    in slice (or something that doesn't require reading it all in).

    Code from the handy astropy FAQ:
    http://astropy.readthedocs.org/en/latest/io/fits/appendix/faq.html#how-can-i-create-a-very-large-fits-file-from-scratch

    Parameters
    ----------
    shape : tuple
        Shape of the data.
    filename : str
        Outputted FITS file name.
    header: FITS header, optional
        Provide a header for the FITS file. If given, the shape in the header
        will be used (if it is given) before the given shape is used.
    '''

    if header is not None:
        try:
            naxis = header["NAXIS"]

            shape = []

            for axis in range(naxis, 0, -1):
                shape.append(header["NAXIS"+str(axis)])

            shape = tuple(shape)
        except KeyError:
            Warning("Cannot extract shape info from header. Using given"
                    " shape instead.")

    naxis = len(shape)

    # Create an array with the right number of dimensions.

    inp_data = np.zeros((100, ) * naxis, dtype=np.float64)

    # Make hdu and pad header with enough header blocks
    hdu = fits.PrimaryHDU(data=inp_data)

    if header is not None:
        hdr = hdu.header
        while len(hdr) < (36 * nblocks - 1):
            hdr.append()
    else:
        hdu.header = header

    # Set the actual shape in the header
    hdr["NAXIS"] = naxis

    for axis in range(1, naxis+1):
        hdr["NAXIS"+str(axis)] = shape[naxis-axis]

    # Save the header
    filename = os.path.splitext(filename)[0] + ".fits"
    hdr.tofile(filename)

    # Now stream some zeros into it!
    nelements = np.prod(shape)
    with open(filename, 'rb+') as fobj:
        fobj.seek(len(hdr.tostring()) + (8 * nelements) - 1)
        fobj.write('\0')
