"""
Transformer class
    

Author(s): Daniela Wiepert
Last Modified: 11/13/2023
"""

#REQUIRED MODULES
#built-in
from typing import List, Tuple, Union, Optional, Iterable, Callable
import warnings

#third-party
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder

#local
from svlite.data_structures._vector_table import VectorTable
from svlite.data_structures._annotation_table import AnnotationTable


class VectorTransformer():
    """
    Class for transforming the feature column of a VectorTable

    Parameters
    ----------
    base_model: str or BaseEstimator w TransformerMixin
        input for initializing the base transformer for the class
    **kwargs
        arguments for the base model if it is being initialized from a str

    """
    def __init__(self, 
                  base_model: Union[str, Tuple[BaseEstimator, TransformerMixin]],
                  **kwargs):
        
        self.isfit = False 
        if isinstance(base_model, str):
            if base_model not in ['StandardScaler']:
                raise ValueError(f'{base_model} is not an implemented transform using string input. Give an initialized model as input or use StandardScaler as the string input.')
            
            if base_model == 'StandardScaler':
                self.base_model = StandardScaler(**kwargs)

        else:
            self.base_model = base_model
    
    def fit(self, 
            table: VectorTable,
            **kwargs):
        """
        Fit a transformer

        Parameters
        ----------
        table: VectorTable
            table for fitting the base model
        **kwargs
            any additional arguments for the fit function of the base model
       
        """
        assert isinstance(table, VectorTable), 'Input must be a VectorTable'

        X = table.to_numpy() 
        self.base_model.fit(X, **kwargs)
        self.isfit = True
    
    def transform(self,
                  table: VectorTable,
                  **kwargs) -> VectorTable:
        """
        Transform a table
        Parameters
        ----------
        table: VectorTable/AnnotationTable
            table to transform
        **kwargs
            any additional arguments for the transform function of the base model

        Returns
        ----------
        new_t: VectorTable
            transformed VectorTable
        """
        assert self.isfit, 'Transformer has not been fit'
        assert isinstance(table, VectorTable), 'Input must be a VectorTable'
        x = table.to_numpy() 
        x_tf = self.base_model.transform(x, **kwargs)

        new_data = table.data.copy()
        new_data[table.feature_col] = list(x_tf)
        new_t = table.copy()
        new_t.data = new_data

        return new_t
    

class AnnotationTransformer():
    """
    Class for transforming columns from an AnnotationTable

    Parameters
    ----------
    base_model: str or BaseEstimator w TransformerMixin
        input for initializing the base transformer for the class. Note that if there is an option for sparse_output, it should be set to False
    transform_cols: list of str
        list of string column names for columns to transform
    **kwargs
        arguments for the base model if it is being initialized from a str

    """
    def __init__(self, 
                base_model: Union[str, Tuple[BaseEstimator, TransformerMixin]],
                transform_cols: List[str],
                **kwargs):

        self.isfit = False 
        self.transform_cols = transform_cols
        if isinstance(base_model, str):
            if base_model not in ['OneHotEncoder']:
                raise ValueError(f'{base_model} is not an implemented transform using string input. Give an initialized model as input or use OneHotEncoder as the string input.')
            
            if base_model == 'OneHotEncoder':
                self.base_model = OneHotEncoder(sparse_output=False, **kwargs)

        else:
            self.base_model = base_model

        self.base_model.set_output(transform="pandas")
        

    def fit(self, 
            table: AnnotationTable,
            **kwargs):
        """
        Fit a transformer

        Parameters
        ----------
        table: AnnotationTable
            table for fitting the base model
    
        """
        assert isinstance(table, AnnotationTable), 'Input must be a AnnotationTable'
        assert set(self.transform_cols).issubset(set(table.data.columns)) #TODO

        # ignore_cols = table.binary_annotation_cols + table.supplemental_cols #NOTE: the annotation table can't take categorical data as strings, only ints. Can't assume continuous != categorical
        
        # assert not any([t in ignore_cols for t in self.transform_cols])
        
        X = table.data[self.transform_cols]

        self.base_model.fit(X, **kwargs)
        self.isfit = True
    
    def transform(self,
                table: AnnotationTable,
                **kwargs) -> AnnotationTable:
        """
        Transform a table
        Parameters
        ----------
        table: AnnotationTable
            table to transform

        Returns
        ----------
        new_t: AnnotationTable
            transformed AnnotationTable
        """
        assert self.isfit, 'Transformer has not been fit'
        assert isinstance(table, AnnotationTable), 'Input must be a VectorTable'
        assert set(self.transform_cols).issubset(set(table.data.columns)) 


        x = table.data[self.transform_cols]
        x_df = self.base_model.transform(x, **kwargs)
        
        new_data = table.data.copy().join(x_df)
   

        #TODO check annotation_cols
        new_t = AnnotationTable(new_data, index_col=table.index_col, subject_col=table.subject_col, 
                                incident_date_col=table.incident_date_col, 
                                annotation_cols=table.annotation_cols+x_df.columns.to_list(), supplemental_cols=table.supplemental_cols)
        return new_t
