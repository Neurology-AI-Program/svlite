import numpy as np
import nibabel
import nilearn.image

SPACEFARER_COLORS = [
    '#9CF0B5',
    '#0011F0',
    '#F09D9C',
    '#F0BE00',
    '#D7F09C',
    '#00D3F0',
    '#D99CF0',
    '#F06900',
    '#F0DE9C',
    '#00F043',
    '#9CA4F0',
    '#F00002',
    '#F0C09C',
    '#AAF000',
    '#9CE6F0',
    '#AC00F0'
]

def suvr_normalize(img, suvr_mask):
	
	img_data = img.get_fdata().squeeze()
	suvr_mask_data = suvr_mask.get_fdata().squeeze()
	suvr_values_ma = np.ma.masked_equal(img_data*suvr_mask_data, 0)
	suvr_median = np.median(suvr_values_ma.compressed())
	img_data_suvr = img_data/suvr_median

	return nibabel.Nifti1Image(img_data_suvr, affine = img.affine, header = img.header)


def mask_data(img, mask):

	img_data = img.get_fdata().squeeze()
	mask_data = mask.get_fdata().squeeze()
	mask_ma = np.ma.masked_equal(mask_data, 0)

	return img_data*mask_ma

def smooth_img(img, fwhm, mask = None):

	smooth_img = nilearn.image.smooth_img(img, fwhm)
	if mask is not None:
		smooth_data = smooth_img.get_fdata().squeeze()*mask.get_fdata()
		return nibabel.Nifti1Image(smooth_data, header = img.header, affine = img.affine)
	else:
		return smooth_img

