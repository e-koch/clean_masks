
from spectral_cube import SpectralCube
import numpy as np
import scipy.ndimage as nd
from signal_id import RadioMask, Noise
from radio_beam import Beam
import astropy.units as u

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
    cube : numpy.ndarray

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
                 pb_thresh=0.7):
        super(CleanMask, self).__init__()
        self.cube = cube
        self.low_cut = low_cut
        self.high_cut = high_cut

        if isinstance(beam, Beam):
            self.beam = beam
        elif beam is None:
            self.beam = None
        else:
            raise TypeError("beam must be a Beam object or None.")

        if pbcoverage is not None:
            if not isinstance(pbcoverage, np.ndarray):
                raise TypeError("pbcoverage must be a numpy array.")

            if pb_thresh < 0.0 or pb_thresh > 1.0:
                raise Warning("pb_thresh must be between 0 and 1.")

            self.pb_mask = pbcoverage > pb_thresh
            self.pb_flag = True
        else:
            self.pb_mask = np.ones_like(cube)
            self.pb_flag = False

        self.vel_slices = self.cube.shape[0]  # Generalize with WCS object

        self._low_mask = None
        self._high_mask = None

        self._mask = None

    def make_initial_masks(self, compute_slicewise=False):
        '''
        Calculate the initial masks.
        '''

        if compute_slicewise:
            sums = 0.0
            num_finite = 0

            for i in range(self.vel_slices):

                sums += np.nansum(self.cube[i, :, :])

                num_finite += np.isfinite(self.cube[i, :, :]).sum()

            mean = sums / num_finite

            var = 0.0

            for val in self.cube[np.isfinite(self.cube)]:
                var += np.np.power(val - mean, 2)

            std = np.sqrt(var) / (num_finite - 1)

            low_thresh = mean + self.low_cut * std
            high_thresh = mean + self.high_cut * std

            self._low_mask = np.zeros_like(self.cube)
            self._high_mask = np.zeros_like(self.cube)

            for i in range(self.vel_slices):
                self._low_mask[i, :, :] = self.cube[i, :, :] > low_thresh
                self._high_mask[i, :, :] = self.cube[i, :, :] > high_thresh

        else:
            mean = np.nanmean(self.cube)
            std = np.nanstd(self.cube)

            low_thresh = mean + self.low_cut * std
            high_thresh = mean + self.high_cut * std

            self._low_mask = self.cube > low_thresh
            self._high_mask = self.cube > high_thresh

        return self

    @property
    def low_mask(self):
        return self._low_mask

    @property
    def high_mask(self):
        return self._high_mask

    @property
    def mask(self):
        return self._mask

    def to_RadioMask(self, which_mask='final'):

        if which_mask is 'final':
            return RadioMask(self._mask, wcs=None)  ## Load in WCS somehow

        elif which_mask is 'low':
            return RadioMask(self._low_mask, wcs=None)  ## Load in WCS somehow

        elif which_mask is 'high':
            return RadioMask(self._high_mask, wcs=None)  ## Load in WCS somehow

        else:
            raise TypeError("which_mask must be 'final', 'low', or 'high'.")


    def dilate_into_low(self, max_iter=100, verbose=False):
        '''
        Dilates the high mask into the low.
        The stopping criterion is when the higher mask crosses lower the one
        '''

        dilate_struct = nd.generate_binary_structure(2, 3)

        for i in range(self.vel_slices):

            # Skip empty channels
            if self._high_mask[i, :, :].max() is False:
                continue

            if verbose:
                print "Iteration %s of %s" % (str(i+1), self.vel_slices)

            iters = 0
            while True:
                self._high_mask[i, :, :] = \
                    nd.binary_dilation(self._high_mask[i, :, :],
                                       structure=dilate_struct)

                posns = np.where(self._high_mask[i, :, :] > 0)

                if np.any(self._low_mask[i, :, :][posns] == 0):

                    # Go back one dilation
                    self._high_mask[i, :, :] = \
                        nd.binary_erosion(self._high_mask[i, :, :],
                                          structure=dilate_struct)

                    break

                iters += 1

                if iters == max_iter:
                    print "Reached max iterations."
                    break

        self._mask = self._high_mask

        return self

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
            #Should this be made into an optional input?
            beam_pix_area = np.floor(np.pi * major * minor)

        else:
            beam_pix_area = 0

        for i in range(self.vel_slices):

            if verbose:
                print "Iteration %s of %s" % (str(i+1), self.vel_slices)

            # Skip empty channels
            if self.high_mask[i, :, :].max is False:
                continue

            low_labels, low_num = nd.label(self._low_mask[i, :, :], connect)

            for j in range(1, low_num+1):

                low_pix = zip(*np.where(low_labels == j))

                high_pix = zip(*np.where(self._high_mask[i, :, :] > 0))

                # Now check for overlap

                matches = list(set(low_pix) & set(high_pix))

                # Add in some check to make sure region is at least the beam size.
                if len(matches) >= min_pix:
                    continue

                if len(matches) > beam_pix_area:
                    continue

                x_pos = [x for x, y in low_pix]
                y_pos = [y for x, y in low_pix]

                # If less than match threshold, remove region in the low mask
                self._low_mask[i, :, :][x_pos, y_pos] = 0

        self._mask = self._low_mask

        return self

    def _smooth_it(self, kern_size='beam', pixscale=None):
        '''
        Apply median filter to smooth the edges of the mask.
        '''

        if kern_size is 'beam':
            if pixscale is None:
                raise TypeError("pixscale must be specified to use beamarea")

            major = self.major.to(u.deg).value/pixscale
            minor = self.minor.to(u.deg).value/pixscale

        elif isinstance(kern_size, float) or isinstance(kern_size, int):
            major = kern_size
            minor = kern_size
        else:
            Warning("kern_size must be 'beam', or a float or integer.")

        from scipy.ndimage import median_filter


        for i in range(self.vel_slices):
            self._mask[i, :, :] = \
                median_filter(self._mask[i, :, :],
                              footprint=np.ones((major, minor)))

        return self

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

        if self.pb_flag:
            self._mask *= self.pb_mask

        return self
