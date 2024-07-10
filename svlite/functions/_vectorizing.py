import numpy as np
import nibabel
import nilearn.image

from ._common import suvr_normalize, mask_data

def check_img_space(img, brain_mask):
	if not np.allclose(img.affine, brain_mask.affine):
		raise ValueError(f'img has wrong affine matrix.  Model is expecting {brain_mask.affine}, this img has {img.affine}')

	if img.shape[:3] != brain_mask.shape:
		raise ValueError(f'img has wrong shape.  Model is expecting {brain_mask.shape}, this img has {img.shape}')
		

def check_vector_length(vec, brain_mask):

	mask_length = brain_mask.get_fdata().sum().astype('int')
	if vec.size != mask_length:
            raise ValueError(f'img has wrong number of non-zero elements, expected {mask_length}, found {vec.size}')


def brain_to_vector(img, brain_mask, suvr_mask = None, resample = False):

	if resample:
		img_resample = nilearn.image.resample_to_img(img, target_img = brain_mask)
	else:
		img_resample = img

	check_img_space(img_resample, brain_mask)

	if suvr_mask:
		img_norm = suvr_normalize(img_resample, suvr_mask)
	else:
		img_norm = img_resample

	img_ma = mask_data(img_norm, brain_mask)
	img_vector = img_ma.compressed()

	check_vector_length(img_vector, brain_mask)

	return img_vector


def vector_to_brain(vec, brain_mask, nz_indices = None):

	img_data = np.zeros(brain_mask.shape)
	check_vector_length(vec, brain_mask)
	if nz_indices is None:
		nz_indices = np.ma.nonzero(brain_mask.get_fdata())

	img_data[nz_indices] = vec

	return nibabel.Nifti1Image(img_data, header = brain_mask.header, affine = brain_mask.affine)
