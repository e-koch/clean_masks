
import numpy as np
import scipy.ndimage as nd
from signal_id import RadioMask, Noise
from radio_beam import Beam
import astropy.units as u
from astropy.io import fits
from astropy.extern import six
import astropy
from skimage.morphology import reconstruction

'''
Routines for constructing a robust clean mask.

1) Pick two sigma levels, then dilate the higher into the lower.
2) Pick two sigma levels, remove any components in the lower cut if it
   doesn't contain any pixels in the higher cut mask.
'''


class CleanMask(object):
    """
    Creates a robust CLEAN mask.

    Parameters
    ----------
    cube : numpy.ndarray or astropy PrimaryHDU

    low_cut : float or int
        Lower sigma cut.
    high_cut : float or int
        Higher sigma cut.
    beam : Beam
        Object defining the beam.
    pbcoverage : numpy.ndarray
        Defines the beam coverage over the image for mosaics.
    pb_thresh : float
        Defines a threshold between 0 and 1 to remove regions with low beam
        coverage in the image.

    """
    def __init__(self, cube, low_cut, high_cut, beam=None, pbcoverage=None,
                 pb_thresh=0.7, iteraxis=0):
        super(CleanMask, self).__init__()
        self._cube = cube
        self.low_cut = low_cut
        self.high_cut = high_cut

        if isinstance(beam, Beam):
            self.beam = beam
        elif beam is None:
            self.beam = None
        else:
            raise TypeError("beam must be a Beam object or None.")

        if pbcoverage is not None:
            if isinstance(pbcoverage, six.string_types):
                pbcoverage = fits.getdata(pbcoverage)

            if not isinstance(pbcoverage, np.ndarray):
                raise TypeError("pbcoverage must be a numpy array.")

            if pb_thresh < 0.0 or pb_thresh > 1.0:
                raise Warning("pb_thresh must be between 0 and 1.")

            self.pb_mask = pbcoverage > pb_thresh
            self.pb_flag = True
        else:
            self.pb_mask = np.ones_like(cube, dtype=bool)
            self.pb_flag = False

        if iteraxis > len(self.cube.shape):
            raise IndexError(str(iteraxis)+"is greater than the total number"
                             " of axes.")
        self.iteraxis = iteraxis

        self.restor_dims = [np.newaxis if i == 1 else slice(None)
                            for i in self.cube.shape]
        self.restor_dims.pop(self.iteraxis)

        self._low_mask = None
        self._high_mask = None

        self._mask = None

        self._pb_applied = False
        self._smoothed = False
        self._method = "None"
        self._pb_thresh = pb_thresh

    @property
    def cube(self):
        return Cube(self._cube)

    def make_initial_masks(self, compute_slicewise=False):
        '''
        Calculate the initial masks.
        '''

        if compute_slicewise or self.cube.huge_flag:
            sums = 0.0
            num_finite = 0

            for plane in self.cube.generate_slice(self.iteraxis):

                sums += np.nansum(plane)

                num_finite += np.isfinite(plane).sum()

            mean = sums / num_finite

            var = 0.0

            for plane in self.cube.generate_slice(self.iteraxis):
                var += np.nansum(np.power(plane - mean, 2), axis=None)

            std = np.sqrt(var / (num_finite - 1))

            print "Slice"
            print mean
            print std

            low_thresh = mean + self.low_cut * std
            high_thresh = mean + self.high_cut * std

            self._low_mask = np.zeros(self.cube.shape, dtype=bool)
            self._high_mask = np.zeros(self.cube.shape, dtype=bool)

            for slices in self.cube.generate_slice(self.iteraxis,
                                                   return_slice=False):
                self._low_mask[slices] = self.cube[slices] > low_thresh
                self._high_mask[slices] = self.cube[slices] > high_thresh

        else:
            mean = np.nanmean(self.cube[:])
            std = np.nanstd(self.cube[:])

            print "Full"
            print mean
            print std

            low_thresh = mean + self.low_cut * std
            high_thresh = mean + self.high_cut * std

            self._low_mask = self.cube > low_thresh
            self._high_mask = self.cube > high_thresh

    @property
    def low_mask(self):
        return self._low_mask

    @property
    def high_mask(self):
        return self._high_mask

    @property
    def mask(self):
        return self._mask

    @property
    def method(self):
        return self._method

    def to_RadioMask(self, which_mask='final'):

        if which_mask is 'final':
            return RadioMask(self._mask, wcs=None)  # Load in WCS somehow

        elif which_mask is 'low':
            return RadioMask(self._low_mask, wcs=None)  # Load in WCS somehow

        elif which_mask is 'high':
            return RadioMask(self._high_mask, wcs=None)  # Load in WCS somehow

        else:
            raise TypeError("which_mask must be 'final', 'low', or 'high'.")

    def dilate_into_low(self, verbose=False):
        '''
        Dilates the high mask into the low using morphological reconstruction.
        '''

        dilate_struct = nd.generate_binary_structure(2, 3)

        for i, slices in enumerate(self.cube.generate_slice(self.iteraxis,
                                                            return_slice=False)):

            # Skip empty channels
            if self._high_mask[slices].max() is False:
                continue

            if verbose:
                print "Iteration %s of %s" % (str(i+1),
                                              self.cube.shape[self.iteraxis])

            self.high_mask[slices] = \
                reconstruction(self.high_mask[slices].squeeze(),
                               self.low_mask[slices].squeeze(),
                               selem=dilate_struct)[self.restor_dims]

        self._mask = self._high_mask
        self._method = "dilate"

    def remove_high_components(self, min_pix=10, beam_check=False,
                               pixscale=None, verbose=False):
        '''
        Remove components in the low mask which are not
        contained in the high mask.

        The criteria is set by min_pix, or is based off of the beam area.
        Note that if min_pix < beam area, min_pix has no effect.
        '''

        # 8-connectivity
        connect = np.ones((3, 3))

        # Objects must be at least the beam area to be kept.
        if beam_check:

            # Remove this when WCS object is added.
            if pixscale is None:
                raise TypeError("pixscale must be specified to use beamarea")

            major = self.major.to(u.deg).value/pixscale
            minor = self.minor.to(u.deg).value/pixscale

            # Round down by default?
            # Should this be made into an optional input?
            beam_pix_area = np.floor(np.pi * major * minor)

        else:
            beam_pix_area = 0

        for i, slices in enumerate(self.cube.generate_slice(self.iteraxis,
                                                            return_slice=False)):

            if verbose:
                print "Iteration %s of %s" % (str(i+1),
                                              self.cube.shape[self.iteraxis])

            # Skip empty channels
            if self.high_mask[slices].max() is False:
                continue

            low_labels, low_num = nd.label(self._low_mask[slices], connect)

            for j in range(1, low_num+1):

                low_pix = zip(*np.where(low_labels == j))

                high_pix = zip(*np.where(self._high_mask[slices] > 0))

                # Now check for overlap

                matches = list(set(low_pix) & set(high_pix))

                if len(matches) >= min_pix:
                    continue

                if len(matches) > beam_pix_area:
                    continue

                x_pos = [x for x, y in low_pix]
                y_pos = [y for x, y in low_pix]

                # If less than match threshold, remove region in the low mask
                self._low_mask[slices][x_pos, y_pos] = 0

        self._mask = self._low_mask
        self._method = "remove small"

    def _smooth_it(self, kern_size='beam', pixscale=None):
        '''
        Apply median filter to smooth the edges of the mask.
        '''

        if kern_size is 'beam':
            if pixscale is None:
                raise TypeError("pixscale must be specified to use beamarea")

            footprint = self.beam.as_tophat_kernel(pixscale)

        elif isinstance(kern_size, float) or isinstance(kern_size, int):
            major = kern_size
            minor = kern_size
            footprint = np.ones((major, minor))
        else:
            Warning("kern_size must be 'beam', or a float or integer.")

        from scipy.ndimage import median_filter

        for i, slices in enumerate(self.cube.generate_slice(self.iteraxis,
                                                            return_slice=False)):
            self._mask[slices] = \
                median_filter(self._mask[slices],
                              footprint=footprint)[self.restor_dims]

        self._smoothed = True

    def apply_pbmask(self):
        '''
        Apply the given primary beam coverage mask.
        '''
        if self.pb_flag:
            self._mask *= self.pb_mask
            self._pb_applied = True

    def save_to_fits(self, filename, header=None, append_comments=True):
        '''
        Save the final mask as a FITS file. Optionally append the parameters
        used to create the mask.
        '''

        if header is not None and append_comments:
            header["COMMENT"] = "Settings used in CleanMask: "
            header["COMMENT"] = "Mask created with method "+self.method
            if self._smoothed:
                header["COMMENT"] = "Mask smoothed with beam kernel."
            if self.pb_flag:
                header["COMMENT"] = \
                    "Mask corrected for pb coverage with a threshold of " + \
                    str(self._pb_thresh)

        # Set BITPIX to 8 (unsigned integer)
        header["BITPIX"] = 8

        hdu = fits.PrimaryHDU(self.mask.astype(">i2"), header=header)
        hdu.writeto(filename)

    def make_mask(self, method="dilate", compute_slicewise=False,
                  smooth=False, kern_size='beam', pixscale=None,
                  verbose=False):

        self.make_initial_masks(compute_slicewise=compute_slicewise)

        if method == "dilate":
            self.dilate_into_low(verbose=verbose)
        elif method == "remove small":
            self.remove_high_components(pixscale=pixscale, verbose=verbose)
        else:
            raise TypeError("method must be 'dilate' or 'remove small'.")

        if smooth:
            self._smooth_it(kern_size=kern_size, pixscale=pixscale)

        self.apply_pbmask()


class Cube(object):
    """
    Cube attempts to handle numpy arrays and FITS HDUs transparently. This
    is useful for massive datasets, in particular. The data is loaded in only
    for the requested slice.

    It is certainly *NOT* robust or complete, but handles what is needed for
    creating CLEAN masks.
    """
    def __init__(self, cube, huge_flag=None, huge_thresh=5e9,
                 squeeze=True):

        self.cube = cube

        if huge_flag is not None:
            self.huge_flag = huge_flag
        else:
            self.huge_flag = self.size > huge_thresh

    @property
    def cube(self):
        return self._cube

    @cube.setter
    def cube(self, input_cube):

        if isinstance(input_cube, six.string_types):
            input_cube = self._load_fits(input_cube)

        is_array = isinstance(input_cube, np.ndarray)
        is_hdu = isinstance(input_cube, astropy.io.fits.hdu.image.PrimaryHDU)

        if not is_array and not is_hdu:
            raise TypeError("cube must be a numpy array or an astropy "
                            "PrimaryHDU. Input was of type " +
                            str(type(input_cube)))

        self._cube = input_cube

    def __getitem__(self, view):
        if self.is_hdu:
            return self.cube.data[view]
        else:
            return self.cube[view]

    def _load_fits(self, fitsfile, ext=0):
        return fits.open(fitsfile)[ext]

    def _is_hdu(self):
        if hasattr(self.cube, 'header'):
            return True
        return False

    @property
    def is_hdu(self):
        return self._is_hdu()

    @property
    def shape(self):
        return self.cube.shape

    @property
    def size(self):
        return self.cube.size

    def close(self):
        '''
        If an HDU, close it.
        '''
        if self.is_hdu:
            self.cube.close()

    def generate_slice(self, iteraxis, return_slice=True):
        slices = [slice(None)] * len(self.shape)
        for i in xrange(self.shape[iteraxis]):
            slices[iteraxis] = i
            if return_slice:
                yield self[slices]
            else:
                yield slices

    def __gt__(self, value):
        return self[:] > value

    def __lt__(self, value):
        return self[:] < value

    def __ge__(self, value):
        return self[:] >= value

    def __le__(self, value):
        return self[:] <= value
