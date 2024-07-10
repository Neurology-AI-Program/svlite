import pandas as pd
import numpy as np
import nilearn.image
from joblib import Parallel, delayed
from pathlib import Path

from svlite.functions import *

from ._vector_table import VectorTable

class BrainTable(VectorTable):

    def __init__(
                self,
                brain_mask,
                suvr_mask = None,
                df = None,
                path = None,
                img_loader = None,
                index_col = None,
                subject_col = None,
                incident_date_col = None,
                feature_col = None,
                smoothing_fwhm = None, 
                max_workers = 4,
                resample_img = None,
                resample_affine = None,
                serializable = True
                ):

        def _mask_resampler(img):

            if img is None:
                return None
            elif not self.resample:
                return img
            elif self.resample_img is not None:
                return nilearn.image.resample_to_img(img, target_img = self.resample_img, interpolation = 'nearest')
            elif self.resample_affine is not None:
                return nilearn.image.resample_img(img, target_affine = self.resample_affine, interpolation = 'nearest')
            else:
                raise Exception(f'Either resample_image or resample_affine must be defined for resampling')

        self.resample_img = resample_img
        self.resample_affine = resample_affine

        if self.resample_img is None and self.resample_affine is None:
            self.resample = False
        else:
            self.resample = True
            
        self.brain_mask = _mask_resampler(brain_mask)
        self.suvr_mask = _mask_resampler(suvr_mask)
        self.nz_indices = np.ma.nonzero(self.brain_mask.get_fdata())
        self.serializable = serializable
        self.max_workers = max_workers
        self.smoothing_fwhm = smoothing_fwhm

        self.loaded = False

        if path is not None:
            if Path(path).suffix == '.parquet':
                df = pd.read_parquet(path)
            elif Path(path).suffix == '.csv':
                df = pd.read_csv(path)
            else:
                raise ValueError(f'can only read parquet or csv files')
            
        if index_col is not None:
            df.set_index(index_col, inplace = True)
            
        if img_loader is not None:
            self._image_vector_name = 'image_vector'
            df = self._vectorize_brains(
                img_loader = img_loader,
                df = df,
                smoothing_fwhm = self.smoothing_fwhm, 
                max_workers = self.max_workers, 
                serializable = self.serializable
                )
            feature_col = self._image_vector_name

        if df is not None:

            if feature_col is None:
                raise ValueError(f'No feature column was specified. Available columns : {df.columns.tolist()}')
            
            super().__init__(df, feature_col = feature_col, subject_col = subject_col, incident_date_col = incident_date_col)
            vec = self.data.head(1).iloc[0][self.feature_col]
            check_vector_length(vec, self.brain_mask)
            self.loaded = True


    def brain_to_vector(self, img, suvr = True):

        if suvr:
            return brain_to_vector(img, brain_mask = self.brain_mask, suvr_mask = self.suvr_mask, resample = self.resample)
        else:
            return brain_to_vector(img, brain_mask = self.brain_mask, suvr_mask = None, resample = self.resample)


    def vector_to_brain(self, vec = None, label = None):

        if vec is not None:
            img = vector_to_brain(vec, self.brain_mask, nz_indices = self.nz_indices)
        elif label is not None:
            vec = self.data.loc[label][self.feature_col]
            img = vector_to_brain(vec, self.brain_mask, nz_indices = self.nz_indices)
        else:
            raise ValueError('need a vector or an index to a vector, nothing was provided')

        return img


    def smooth_data(self, fwhm, img = None, vec = None):
        
        if vec is None and img is None:
            ValueError('Need either an image or a vector argument, but neither were provided')
        elif vec is None and img is not None:
            return self._smooth_img(img, fwhm)
        elif vec is not None and img is None:
            return self._smooth_vector(vec, fwhm)
        else:
            raise ValueError('Need either an image or a vector argument, but both were provided')


    def subset(self, labels):

        df_subset = self.data.loc[labels].copy()

        B_subset = BrainTable(brain_mask = self.brain_mask, suvr_mask = self.suvr_mask, df = df_subset)

        return B_subset


    def _smooth_img(self, img, fwhm, apply_mask = True):

        if apply_mask:
            return smooth_img(img, fwhm, mask = self.brain_mask)
        else:
            return smooth_img(img, fwhm)


    def _smooth_vector(self, vec, fwhm):

        img = self.vector_to_brain(vec = vec)
        img_smooth = self._smooth_img(img, fwhm)
        return self.brain_to_vector(img_smooth, suvr = False)


    def _vectorize_brains(self, img_loader, df, smoothing_fwhm, max_workers, serializable):

        if serializable: 
            prefer = 'processes'
        else:
            prefer = 'threads'

        def _task(row, smoothing_fwhm):
            label, img = img_loader(row)
            if smoothing_fwhm is not None:
                img =  self._smooth_img(img, fwhm = smoothing_fwhm, apply_mask = False)
            vec = brain_to_vector(img, self.brain_mask, self.suvr_mask, self.resample)
            return label, vec

        if max_workers is not None and max_workers > 1:
            labeled_vectors = Parallel(n_jobs = max_workers, prefer = prefer)(delayed(_task)(row, smoothing_fwhm) for row in df.iterrows())
        else:
            labeled_vectors = [_task(row, smoothing_fwhm) for row in df.iterrows()]

        X = np.row_stack([v for _, v in labeled_vectors])
        labels = [l for l, _ in labeled_vectors]

        return df.join(pd.DataFrame({'label' : labels, self._image_vector_name : list(X)}).set_index('label'))
     

