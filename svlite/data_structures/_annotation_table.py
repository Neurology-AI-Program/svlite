"""
AnnotationTable class


Author(s):
Last Modified:
"""

# REQUIRED MODULES
# built-in
from dataclasses import dataclass
import datetime
import os
from typing import List, Tuple, Iterable, Optional, Union
import warnings

# third-party
import pandas as pd
import numpy as np

# local
from ._base_table import _BaseTable


class AnnotationTable(_BaseTable):
    """
    Annotation table to load data, validate and down sample annotation columns
    
    1. loads data by Initializing _BaseTable with index_col, subject_col and incident_date_col

    2. Validates annotation cols by identifying annotation columns specified in `annotation_cols` parameter and keeps track of the col names in `self.annotation_cols`
    
    3. Identifies binary annotation column names and continuous annotation column names in `self.binary_annotation_cols` and `self.continuous_annotation_cols`
    
    4. Raises error if there are Missing values in the annotation columns

    5. Down sample annotation data to columns `subject_col`, `incident_date_col`, `annotation_cols`, and `supplemental_cols`

    Parameters
    ----------
    data : pd.Dataframe/str/np.ndarray
        initialize data with either dataframe, str, or numpy array

    index_col :  List/str/np.ndarray, default None
        Any index to use for the dataframe.

    subject_col : List/str/np.ndarray, default None
        String to identify column with subject ids

    incident_date_col : List/str, default None
       String to identify column with incident dates

    annotation_cols: List [str], default None
        names of the annotation columns

    supplemental_cols: List [str], default None
        names of additional columns to keep but not validate 
    """

    def __init__(self,
                 data,
                 index_col: Optional[Union[Iterable, str]] = None,
                 subject_col: Optional[Union[Iterable, str]] = None,
                 incident_date_col: Optional[Union[Iterable, str]] = None,
                 annotation_cols: Iterable = None,
                 supplemental_cols: Iterable = None
                 ):
        
        super().__init__(data,
                         index_col=index_col,
                         subject_col=subject_col,
                         incident_date_col=incident_date_col)

        # binary annotation columns
        self.binary_annotation_cols = []

        # continuous annotation columns
        self.continuous_annotation_cols = []

        #extra columns we want to keep but have no restrictions on types, etc
        if supplemental_cols is None:
            self.supplemental_cols = []
        else:
            self.supplemental_cols = supplemental_cols

        # get the list of annotation column names
        self.annotation_cols = self._get_annotation_cols(annotation_cols)
        
        # downsample columns
        downsample_cols = [self.subject_col] + [self.incident_date_col] + self.annotation_cols + self.supplemental_cols
        downsample_cols = [col for col in downsample_cols if col is not None]
        
        # copy annotation columns, subject column, and incident date column
        self.data = self.data[downsample_cols].copy()
        
        # check if dataframe has no columns
        if not self.data.columns.tolist():
            raise ValueError('Annotation table has no columns')

        # check if dataframe has no rows
        if not self.data.index.tolist():
            raise ValueError('Annotation table has no rows')

        # can inherit _BaseTable public variables

    def _get_annotation_cols(self, annotation_cols: List[str] = None) -> List[str]:
        """
        annotations columns should include all other columns except subject_col, incident_date_col, and supplemental_cols
            1. They should not contain missing values
            2. They should be binary if the values are 0/1
            3. Else floats

        Parameters
        ----------
        annotation_cols: List [str], default None
            names of the annotation columns

        Returns
        -------
        annotation_cols: List [str]
            names of the annotation columns
        """

        if annotation_cols is None:
            # all columns other than annotation subject and incident_date_col are annotation cols
            non_annotation_cols = [self.subject_col, self.incident_date_col] + self.supplemental_cols
            annotation_cols = self.data.columns.difference(non_annotation_cols).tolist()


        # check if annotation cols in data
        if not set(annotation_cols).issubset(set(self.data.columns)):
            raise ValueError(f'Annotation columns {annotation_cols} are not in the dataframe with columns '
                             f'{self.data.columns}')

        data = self.data[annotation_cols].copy()

        # check for missing values
        col_names_with_missing_values = data.columns[data.isnull().any()].tolist()
        if col_names_with_missing_values:
            raise ValueError(f'There are missing values in the dataframe with columns: {col_names_with_missing_values}')

        # check columns that are binary, if they are make them of type int
        self.binary_annotation_cols = data.columns[data.isin([0, 1]).all()].tolist()
        self.data[self.binary_annotation_cols] = self.data[self.binary_annotation_cols].astype(int)

        # check if the remaining annotation_cols are all continuous
        remaining_cols = list(set(annotation_cols) - set(self.binary_annotation_cols))
        if remaining_cols:
            # forcefully convert them to float, if they fail raise an error
            try:
                self.data[remaining_cols] = self.data[remaining_cols].apply(pd.to_numeric)
            except ValueError:
                raise ValueError(f'These dataframe columns are not float/binary : {remaining_cols}')

            self.continuous_annotation_cols = remaining_cols

        return annotation_cols

    def copy(self) -> object:
        """
        Returns a copy of the AnnotationTable object
        
        Returns
        -------
        AnnotationTable
            copy of the AnnotationTable table object
        """
        if self.annotation_cols:
            data_copy = self.data.copy()
            return AnnotationTable(data=data_copy,
                                   index_col=None,
                                   subject_col=self.subject_col,
                                   incident_date_col=self.incident_date_col,
                                   annotation_cols=self.annotation_cols,
                                   supplemental_cols=self.supplemental_cols
                                   )

        else:
            raise ValueError("Data is not initialized yet")






