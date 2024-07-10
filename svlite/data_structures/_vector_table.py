"""
VectorTable class
    

Author(s): Daniela Wiepert
Last Modified: 03/09/2023
"""

#REQUIRED MODULES
#built-in
import datetime
import os
from typing import List, Tuple, Union, Optional, Iterable, Callable
import warnings

#third-party
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist, _METRIC_ALIAS, _METRICS

from scipy.optimize import minimize

#local
from ._base_table import _BaseTable
from ._annotation_table import AnnotationTable

class VectorTable(_BaseTable):
    """
    VectorTable class to store feature vectors. Can load data through a pandas DataFrame, numpy array, or a 
    string path to a parquet file

    Inherits from the _BaseTable class with some added functionality for handling features
    
    Parameters
    ----------
    data : pd.Dataframe/str/np.ndarray
        initialize data with either dataframe, str, or numpy array
    index_col :  List/str, default None
        Any index to use for the dataframe.
    subject_col : List/str, default None
        String to identify column with subject ids
    incident_date_col : List/str, default None
        String to identify column with incident dates
    feature_col : str, default None
        String to identify column with features
    annotation_cols : List/str, default = None
        List of strings to use for generating an annotations table
    check_missing: str, default = 'ignore'
        string that stores 'ignore', 'warn', 'raise' to decide how to handle missing values in feature vectors, defaults to 'ignore'
    missing_value: np.float64, default = np.nan
        value to consider as a missing value in a feature 
    features_to_keep: int, str, list[int], list[str]
        value to downsample features by - can give either an int/str of the number of elements to keep (first <features_to_keep> elements),
        or a list of the indices for features to keep
        if list[str] is provided, and the list contains column names in the data; feature vectors will be computed from those list of columns
    feature_names: list[str]
        record feature names as a data variable, should be equal to length of the feature vector
    """
    
    def __init__(self,
                data: Union[pd.DataFrame, str, np.ndarray],
                index_col: Optional[Union[Iterable, str]] = None,
                subject_col: Optional[Union[Iterable, str]] = None,
                incident_date_col: Optional[Union[Iterable, str]] = None,
                feature_col: Optional[str]= None,
                annotation_cols:Optional[Iterable[str]] = None,
                check_missing: Optional[str] = 'ignore',
                missing_value: Optional[np.float64] = np.nan,
                features_to_keep: Optional[Union[int, str, List[int],List[str]]] = None,
                feature_names: Optional[List[str]] = None
                ):
        """
        Initialize and load - inheriting from _BaseTable
        """
        
        #initialize public instance variable(s)
        self.feature_col = feature_col
        self.annotation_table = None 
        self.feature_names = None
        
        #initialize private instance variable(s)
        self._annotation_cols = annotation_cols
        self._check_missing = check_missing
        self._missing_value = missing_value
        
        self._features_to_keep = features_to_keep

        super().__init__(data,
                         index_col=index_col,
                         subject_col=subject_col,
                         incident_date_col=incident_date_col)

        #downsample features
        if self._features_to_keep is not None:
            self.downsample_features(self._features_to_keep)

        #check if provided annotations, if yes, generate AnnotationsTable object
        if self._annotation_cols is not None:
            self._generate_annotations()
            
        # check if feature names is of the same length as the feature vector
        if not feature_names is None:
            length_of_feature_vector = len(self.data[self.feature_col].iloc[0])
            if len(feature_names) == length_of_feature_vector:
                self.feature_names = feature_names
            
            else:
                raise ValueError(f"Feature names {feature_names} donot match the length of the feature vectore {length_of_feature_vector}")
            
    
    def _from_df(self,
                 df: pd.DataFrame,
                 index: Optional[Union[Iterable, str]] = None) -> None:
        """
        Overwritten version of _BaseTable _from_df function. Includes all same functionality,
        but addes checks for dataframe having a valid feature_col and
        generates an annotations table if specificed

        Checks if the dataframe has valid
            1. index
            2. subject_col and/or incident_date col (if provided)
            3. feature_col 
            4. annotations_cols

        Parameters
        ----------
        df : pd.Dataframe
            dataframe to check validity
        index :  List/str, default None
            index to use for the dataframe.

        """
        # load into data object
        self.data = df.copy()

        # Index: unique string (most likely sub-<subject_id>_ses-<YYYMMDDHHMM>_<free_text_eg_protocol>)
        # check for type (coerce to string) and uniqueness
        self._check_set_index(index)

        # check for duplicates in subject and incident date col if provided
        self._check_subject_incident_date()

        #check that the features are correct
        self._check_features()

        # set loaded to true once finished
        self.loaded = True


    def _check_features(self):
        """
        Check for feature column and handle missing values
        Uses instance variables so no input or return paramaters, alters variables in place
        """
        cols = self.data.columns.tolist()
        if self.feature_col is not None:             
            try:
                assert self.feature_col in cols
            except:
                raise ValueError(f'{self.feature_col} column not in input data')
        else:
            # if _features_to_keep is a list of strings, construct a feature vector from those columns
            # just for info: _features_to_keep can also be used to downsample feature_vectors so make sure the columns are in the dataframe
            # if they are in the dataframe, then we can construct a feature vector from them and set the _features_to_keep to None so that it is not downsampled
            if isinstance(self._features_to_keep, list) and \
                all(isinstance(f, str) for f in self._features_to_keep) and \
                all(f in cols for f in self._features_to_keep):
                    # given a list of feature columns, generate a new column with the feature vector
                    self.data[self._default_feature_col_name] = list(self.data[self._features_to_keep].values)
                    self.feature_col = self._default_feature_col_name
                    self._features_to_keep = None
            
            else:
                #if not given a feature column and , use the first column in the dataframe
                self.data.rename(columns={cols[0]: self._default_feature_col_name}, inplace=True)
                self.feature_col = self._default_feature_col_name
        
        #check that feature vector is same length for each row 
        try:
            assert all(len(v) == len(self.data[self.feature_col].values[0]) for v in self.data[self.feature_col].values)
        except:
                raise ValueError('Feature vectors not the same length')
        
        #handle missing values in each feature vector
        self._missing_feat()


    def _missing_feat(self):
        """
        Handle missing values in a feature vector
        Uses instance variables so no input or return paramaters
        """ 
        try:
            assert self._check_missing in ['ignore', 'warn','raise']
        except:
            raise ValueError(f"Expected one of [ignore, warn, raise] for check_missing, but was given {self._check_missing}")
    
        if self._check_missing != 'ignore':
            features = self.data[self.feature_col].values

            for f in features:
                if np.isnan(self._missing_value): #TODO: are there any other edge cases like this we would expect?
                    missing = np.where(np.isnan(f))
                else:
                    missing = np.where(f == self._missing_value)
                if np.any(missing):
                    if self._check_missing == 'warn':
                        warnings.warn('Missing values are present in feature vectors') #is this the proper way to throw a warning?
                        break
                    elif self._check_missing == 'raise':
                        raise ValueError(f'Missing values of type {self._missing_value} are present in feature vectors')


    def _generate_annotations(self):
        """
        Generates an AnnotationsTable object if columns are specified
        """

        #check if given None, that there are no annotations columns present 
        if self._annotation_cols == []:
            cols = self.data.columns.tolist()
            cols = [c for c in cols if c not in [self.feature_col, self.subject_col, self.incident_date_col]]

            self._annotation_cols = cols
            
        if len(self._annotation_cols) != 0:
            #make sure feature vector is excluded
            if self.feature_col in self._annotation_cols:
                self._annotation_cols.remove(self.feature_col)
            #generate annotation table
            annotations = AnnotationTable(self.data, self.index_col, self.subject_col, self.incident_date_col, self._annotation_cols)
            self.annotation_table = annotations.data 
        
            #drop the data from the vector table
            self.data.drop(self._annotation_cols, axis=1, inplace=True)

    def downsample_features(self, features_to_keep: Union[int, str, List[int],List[str]] = None):
        """
        Downsample feature vectors in self.data in place

        Parameters
        ----------
        feature_to_keep: int, str, list[int], list[str]
            value to downsample features by - can give either an int/str of the number of elements to keep (first <features_to_keep> elements),
            or a list of the indices for features to keep
        """
        if isinstance(features_to_keep, list):
            # check if the list is a list of ints
            features_to_keep = [int(f) for f in features_to_keep]
            self.data[self.feature_col] = self.data[self.feature_col].map(lambda x: [f for f in x if f in features_to_keep])
        else:
            features_to_keep = int(features_to_keep)
            self.data[self.feature_col] = self.data[self.feature_col].map(lambda x: x[:features_to_keep])
    

    def calc_group_centers(self,
                           groups: Union[List[List[int]],List[List[str]]],
                           metric: Union[str,Callable]='euclidean',
                           agg_function: bool=False,
                           group_names: List[str] = None) -> object:
        """
        Calculate a vector that minimizes the distance from feature vectors in self.data where groups
        as specified by the input groups.

        Parameters
        ----------
        groups: list[list[string]], list[list[int]]
        metric: (string) of a valid distance metric as specified in scipy.cdist or 
                (callable distance function) that takes as an input a 1D and ND array and returns a float or
                (callable aggregator function) that can be applied to the vector table
            If metric is a string, must equal one of the following distances scipy uses:
                [braycurtis, canberra, chebyshev, cityblock, correlation, 
                cosine, dice, euclidean, hamming, jaccard, jensenshannon, kulczynski1, 
                mahalanobis, matching, minkowski, rogerstanimoto, russellrao, seuclidean, 
                sokalmichener, sokalsneath, sqeuclidean, yule]
        agg_function: boolean if True specifies that the user-provided function in metric is to be used
            as an aggregate for each of the groups, and not find the group center based on a metric, default False
        group_names: list[string], default = None
            list of names to assign to the output groups, assumed to be in same order as groups. If provided,  must 
            be same length as groups.
        
        Returns
        ----------
        group_centers: VectorTable with the calculated group center based on the provided distance metric (or callable)
            or the output of the aggregate function
        group_dict: Dictionary mapping the group names to the feature names
        """

        # updating the metric check with the scipy code
        # check if the metric is manhattan and if so change to cityblock (as used by scipy)
        if isinstance(metric,str):
            metric = metric.lower() # ensure metric is all lowercase
            if (metric=='manhattan'):  # if entered manhattan, scipy treats this as cityblock
                metric = 'cityblock'
            metric_info = _METRIC_ALIAS.get(metric, None)
            if metric_info is None:
                raise ValueError(f'metric {metric} is not a valid distance metric that scipy uses. Must enter one of the following: {list(_METRICS.keys())}')

            # # metrics used by scipy
            # validmetrics = ('braycurtis',
            #                 'canberra',
            #                 'chebyshev',
            #                 'cityblock',
            #                 'correlation',
            #                 'cosine',
            #                 'dice',
            #                 'euclidean', 
            #                 'hamming', 
            #                 'jaccard', 
            #                 'jensenshannon', 
            #                 'kulczynski1', 
            #                 'mahalanobis', 
            #                 'matching', 
            #                 'minkowski', 
            #                 'rogerstanimoto', 
            #                 'russellrao', 
            #                 'seuclidean', 
            #                 'sokalmichener', 
            #                 'sokalsneath', 
            #                 'sqeuclidean', 
            #                 'yule')
            # if metric not in validmetrics:
            #     raise ValueError(f'metric {metric} is not a valid distance metric that scipy uses. Must enter one of the following: {_METRICS.keys()}')

        # initialize as empty list (since might not know the size and converting to list later)
        group_cent_vals = []

        # for each group, find the group center
        group_dict = {}
        if group_names is not None:
            if len(group_names) != len(groups):
                raise ValueError(f'Number of groups and group names must be equal, recieved {len(groups)} groups but {len(group_names)} group_names')
            else:
                idx = group_names
        else:
            idx = [f'group_{n}' for n,_ in enumerate(groups)]

        for n,thisgroup in enumerate(groups):
            # get the features
            X = self.to_numpy(thisgroup)

            # use the mean of the group as an initial guess
            cent_0 = np.mean(X,axis=0)  # needs to be size (1,n) but my_dist_fun reshapes to (n,1) for cdist

            if not agg_function:
                if callable(metric):
                    fit = minimize(metric,x0=cent_0,args=(X))
                else:
                    fit = minimize(self._dist_fun,x0=cent_0,args=(X,metric))
                group_cent_vals.append(fit.x)
            else:
                if callable(metric):
                    val = metric(X)
                    if val.size==1:
                        group_cent_vals.append([val])
                    else:
                        group_cent_vals.append(val)
                else:
                    raise ValueError('Provided aggregate function not callable, if you are providing a string distance metric, set the agg_function input to False')
            
            group_dict[idx[n]] = list(thisgroup)

        group_center_df = pd.DataFrame({'group_centers':group_cent_vals},index=idx)  # group_cent_vals is now a list of lists so don't need to convert from numpy array

        return VectorTable(group_center_df), group_dict
    
    def _dist_fun(self,x,xy,metric):
        # for the fit function, x needs to be size (n,) but for cdist it needs to be size (1,n)
        return cdist(x.reshape(1,-1),xy,metric=metric).sum()

    def to_numpy(self, indices: Union[List[int],List[str],np.ndarray]=[]) -> np.ndarray:
        """
         Generate an np array from feature vector column in a dataframe 
         
        Parameters
        ----------
        indices: list[int], list[string], ndarray: if indices are provided, the output
            array will be shaped [indices, n_features]. If no indices are provided or 
            the indices are empty ([]), the output array will be shaped [n_samples, n_features]
        
        Returns
        ----------
        feature_np, numpy array with feature vector data (n_samples, n_features) or (indices, n_features)
        """

        if len(indices)==0: # if not indices are sent, then return the full feature as a numpy array
            feature_np = np.stack(self.data[self.feature_col])
        else:
            if (isinstance(indices,list) and bool(indices)) or (isinstance(indices,np.ndarray) and indices.size>0):
                if all(isinstance(elem,int) for elem in indices): # if the indices are provided as integers, use the iloc indexing
                    # feature_np = np.array(self.data.iloc[indices,self.feature_col].tolist())
                    feature_np = np.stack(self.data.iloc[indices,self.feature_col])
                elif all(isinstance(elem,str) for elem in indices): # if the indices are provided as strings, use the loc indexing
                    # feature_np = np.array(self.data.loc[indices,self.feature_col].tolist())
                    feature_np = np.stack(self.data.loc[indices,self.feature_col])
                else:
                    feature_np = []
                    ValueError(f'The indices provided were not valid')        
            else:
                feature_np = []
                ValueError(f'The indices provided were not valid')
        return feature_np
    
    def copy(self) -> object:
        """
        Returns a copy of the VectorTable object
        
        Returns
        -------
        VectorTable
            copy of the vector table object
        """
        if self.loaded:
            data_copy = self.data.copy()
            vector_table_copy =  VectorTable(data=data_copy,
                                             index_col=None,
                                             subject_col=self.subject_col,
                                             incident_date_col=self.incident_date_col,
                                             feature_col=self.feature_col,
                                             check_missing=self._check_missing,
                                             missing_value=self._missing_value,
                                             )
            if self.annotation_table is not None:
                vector_table_copy.annotation_table = self.annotation_table.copy()
                vector_table_copy._annotation_cols = self._annotation_cols.copy()
            
            return vector_table_copy
            
        else:
            raise ValueError("Data is not initialized yet")

