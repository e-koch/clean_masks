
import numpy as np
from astropy.io import fits

from lazy_img_regrid import lazy_2D_regrid

'''
Create some single channel maps using lazy regrid
'''

# Load in the already regridded model

model, model_hdr = \
    fits.getdata("/Users/eric/RAIDers_of_the_lost_ark/M33/Arecibo/M33_model.fits",
                 header=True)


# Regrid the 100 and 145th channels in the cube

slice_100 = [slice(None)] * 4
slice_100[0] = slice(100, 101)

mod_100, mod_100_hdr = lazy_2D_regrid(model, (8096, 8096), model_hdr,
                                      channel_slice=slice_100, append_dim=True)

# The shape had a couple extra dimensions appended on to match the input cube.

mod_100_hdu = fits.PrimaryHDU(mod_100, header=mod_100_hdr)
mod_100_hdu.writeto('/Users/eric/RAIDers_of_the_lost_ark/M33/Arecibo/M33_model_channel_100.fits')

slice_145 = [slice(None)] * 4
slice_145[0] = slice(145, 146)

mod_145, mod_145_hdr = lazy_2D_regrid(model, (8096, 8096), model_hdr,
                                      channel_slice=slice_145, append_dim=True)

mod_145_hdu = fits.PrimaryHDU(mod_145, header=mod_145_hdr)
mod_145_hdu.writeto('/Users/eric/RAIDers_of_the_lost_ark/M33/Arecibo/M33_model_channel_145.fits')

# Now do the same for the masks.

mask, mask_hdr = \
    fits.getdata("/Users/eric/RAIDers_of_the_lost_ark/M33/Arecibo/M33_mask.fits",
                 header=True)


# Regrid the 100 and 145th channels in the cube

mask_100, mask_100_hdr = lazy_2D_regrid(mask, (8096, 8096), mask_hdr,
                                        channel_slice=slice_100, append_dim=True)

# The shape had a couple extra dimensions appended on to match the input cube.

mask_100_hdu = fits.PrimaryHDU(mask_100, header=mask_100_hdr)
mask_100_hdu.writeto('/Users/eric/RAIDers_of_the_lost_ark/M33/Arecibo/M33_mask_channel_100.fits')

mask_145, mask_145_hdr = lazy_2D_regrid(mask, (8096, 8096), mask_hdr,
                                        channel_slice=slice_145, append_dim=True)

mask_145_hdu = fits.PrimaryHDU(mask_145, header=mask_145_hdr)
mask_145_hdu.writeto('/Users/eric/RAIDers_of_the_lost_ark/M33/Arecibo/M33_mask_channel_145.fits')

