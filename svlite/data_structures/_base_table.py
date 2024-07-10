"""
BaseTable class


Author(s):
Last Modified:
"""

# REQUIRED MODULES
# built-in
from typing import List, Tuple, Union, Optional, Iterable

# third-party
import pathlib
import datetime
import pandas as pd
import numpy as np

# local


class _BaseTable:
    """
    BaseTable class to load, check and save data
    """

    def __init__(self,
                 data: Union[pd.DataFrame, str, np.ndarray],
                 index_col: Optional[Union[Iterable, str]] = None,
                 subject_col: Optional[Union[Iterable, str]] = None,
                 incident_date_col: Optional[Union[Iterable, str]] = None,
                 ):
        """
        Initializes and loads the dataframe

        Parameters
        ----------
        data : pd.Dataframe/str/np.ndarray
            initialize data with either dataframe, str, or numpy array
        index_col :  List/str, default None
            Any index to use for the dataframe.
        subject_col : str, default None
            String to identify column with subject ids
        incident_date_col : str, default None
           String to identify column with incident dates

        """
        # initialize public instance variables
        self.data = None
        self.index_col = None
        self.subject_col = subject_col
        self.incident_date_col = incident_date_col

        # Flag for data loading
        self.loaded = False
        # check for duplicate subjects (allowed, but flag)
        self.duplicate_subject = []
        # check for duplicate incident_date (allowed, but flag)
        self.duplicate_date = []
        # check for duplicate subject-date pairs (allowed, but flag)
        self.duplicate_subject_date = []

        # private variables
        # private variable for naming feature col when not provided 
        self._default_feature_col_name = 'features'
        # private variable for naming index col
        self._default_index_col = 'uid'
        self._default_subject_col = 'subject'
        self._default_incident_date_col = 'incident_date'

        # load and check data
        self._load_data(data, index_col)

    def _load_data(self,
                   data: Union[pd.DataFrame, str, np.ndarray],
                   index: Optional[Union[Iterable, str]] = None,
                   ) -> None:
        """
        Initializes/loads the dataframe to self.data

        Parameters
        ----------
        data : pd.Dataframe/str/np.ndarray
            initialize data with either dataframe, str, or numpy array
        index :  List/str, default None
            index to use for the dataframe.

        """

        # type of the data can be a dataframe/str/nd array
        # check its type and load the data by calling appropriate function

        if isinstance(data, pd.DataFrame):
            self._from_df(df=data, index=index)

        elif isinstance(data, str):
            extension = pathlib.PurePath(data).suffix

            if extension == ".parquet":
                df = pd.read_parquet(data)
                self._from_df(df=df, index=index)

            elif extension == ".csv":
                df = pd.read_csv(data)
                self._from_df(df=df, index=index)

            elif extension == ".xlsx":
                df = pd.read_excel(data)
                self._from_df(df=df, index=index)

            else:
                raise TypeError(f'Nightingale does not support the given {extension} file, '
                                f'accepted file formats include .parquet/.csv/.xlsx')

        elif isinstance(data, np.ndarray):
            self._from_np(np_array=data, np_index=index)

        else:
            raise TypeError(f'Not a valid data format, accepted formats includes'
                            f' pd.DataFrame, parquet_path and np.ndarray')

    def _check_set_index(self,
                         index: Optional[Union[Iterable, str]] = None) -> None:
        """
        Checks the validity of index and set self.index_col variable to index column name

        1. if the given index is a string
            Checks if the given index is a column on data, if it is, set as index, else raise error
        2. if the given index is a list
            Use the list as the index for the dataframe
        3. if the given index is None
            if the existing index is string type already use it
            else autogenerate index like 'sample_0, sample_1 ... sample_n'
        4. check uniqueness of index

        Parameters
        ----------
        index : List/str, default None
            index to use for the dataframe

        """
        # 1. checks if index is string
        if isinstance(index, str):
            cols = self.data.columns.tolist()
            if self.data.index.name == index:
                self.index_col = index
            # Checks if the given index is a column on data, if it is, set as index, else raise error
            elif index in cols:
                self.index_col = index
                self.data.set_index(self.index_col, inplace=True)
            else:
                raise ValueError(f'The index column {index} does not exist in the given dataframe')

        # 2. if index is list / series / numpy array
        elif isinstance(index, (list, pd.core.series.Series, np.ndarray)):
            # Use the list as the index for the dataframe
            if len(index) == len(self.data):
                self.data[self._default_index_col] = index
                self.data.set_index(self._default_index_col, inplace=True)
            else:
                raise ValueError(f'The length of the index {len(index)} is not equal to '
                                 f'length of the dataframe {len(self.data)}')

        # 3. index is None
        elif index is None:
            # if the index is string type already use it else generate index
            if not all(isinstance(i, str) for i in self.data.index.values):
                index = [f'sample_{i}' for i in range(len(self.data))]
                self.data[self._default_index_col] = index
                self.data.set_index(self._default_index_col, inplace=True)

        # 4. check uniqueness of index
        try:
            assert len(set(list(self.data.index.values))) == len(list(self.data.index.values))
        except AssertionError:
            raise ValueError('Indices for data are not unique')

        self.index_col = self.data.index.name

    def _check_subject_incident_date(self) -> None:
        """
        Checks the validity of subject and incident columns
        If provided
            Check if they are in the dataframe columns
            Check for individual duplicates
            Check for duplicate_pairs

        Parameters
        ----------

        """

        cols = self.data.columns.tolist()

        if not (self.subject_col or self.incident_date_col):
            return None

        # Check if they are in the dataframe columns
        if self.subject_col is not None:
            
            if isinstance(self.subject_col, (list, pd.core.series.Series, np.ndarray)):
                # Use the list as the index for the dataframe
                if len(self.subject_col) == len(self.data):
                    self.data[self._default_subject_col] = self.subject_col
                    self.subject_col = self._default_subject_col
                else:
                    raise ValueError(f'The length of the subject_col {len(self.subject_col)} is not equal to '
                                    f'length of the dataframe {len(self.data)}')

            elif self.subject_col not in cols:
                raise ValueError(f'Given subject column {self.subject_col} is not in the dataframe')

            elif not all(isinstance(i, str) for i in self.data[self.subject_col].values):
                raise TypeError(f'Given subject column {self.subject_col} is not of type string')

        if self.incident_date_col is not None:
            
            if isinstance(self.incident_date_col, (list, pd.core.series.Series, np.ndarray)):
                # Use the list as the index for the dataframe
                if len(self.incident_date_col) == len(self.data):
                    self.data[self._default_incident_date_col] = self.incident_date_col
                    self.incident_date_col = self._default_incident_date_col
                else:
                    raise ValueError(f'The length of the incident_date_col {len(self.incident_date_col)} is not equal to '
                                    f'length of the dataframe {len(self.data)}')
                    
            elif self.incident_date_col not in cols:
                raise ValueError(f'Given Incident date column {self.incident_date_col} is not in the dataframe')

            elif not all(isinstance(i, (datetime.datetime, np.datetime64, np.ndarray))
                         for i in self.data[self.incident_date_col].values):
                try:
                    self.data[self.incident_date_col] = pd.to_datetime(self.data[self.incident_date_col])
                except:
                    raise TypeError(f'Given Incident date column {self.incident_date_col} is not of type datetime')

        # Check for individual duplicates
        # self._duplicate_cols returns list if len(cols) is one
        # else returns list[Tuple]; length of tuple will be the length of cols
        self.duplicate_subject = self._duplicate_cols(cols=[self.subject_col])
        self.duplicate_date = self._duplicate_cols(cols=[self.incident_date_col])

        # Check for duplicate subject_date pairs
        self.duplicate_subject_date = self._duplicate_cols(cols=[self.subject_col,
                                                                 self.incident_date_col])

    def _from_df(self,
                 df: pd.DataFrame,
                 index: Optional[Union[Iterable, str]] = None) -> None:
        """
        Checks if the dataframe has valid
            1. index
            2. subject_col and/or incident_date col (if provided)

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

        # set loaded to true once finished
        self.loaded = True


    def _from_np(self,
                 np_array: np.ndarray,
                 np_index: List[str]):
        """
        Load VectorTable from an np array

        :param np_array: numpy array with data to load
        :type np_array: np.ndarray
        :param np_index: index for numpy data, defaults to None
        :type np_index: List[str]
        """

        try:
            assert np_index is not None
        except AssertionError:
            raise ValueError('No index list given for np.ndarray data')

        try:
            assert np_array.shape[0] == len(np_index)
        except AssertionError:
            raise ValueError(
                f'Feature array and index list must be the same lengths (= # of samples). '
                f'The feature array had a length of {np_array.shape[0]} while index had a length of {len(np_index)}')

        # create dataframe from numpy
        np_array = [n for n in np_array]
        np_index = [n for n in np_index]

        df = pd.DataFrame({self._default_feature_col_name: np_array, self._default_index_col: np_index})
        df = df.set_index(self._default_index_col)

        self._from_df(df=df, index=None)

    def _duplicate_cols(self, cols: List[str]) -> Union[List, List[Tuple]]:
        """
        Checks the validity of subject and incident columns
        If provided
            Check if they are in the dataframe columns
            Check for individual duplicates
            Check for duplicate_pairs

        Parameters
        ----------
        cols : str
            Name of the columns to check for duplicates

        Returns
        -------
        duplicates: List/List[tuple]
            if single column return list of duplicates
            if multiple columns return duplicates as list of tuples in the given `cols` order
        """

        # if any of the col is None return empty list
        for col in cols:
            if not col:
                return []

        cols_to_check_duplicates = cols

        # mask duplicates
        mask_duplicate_cols = self.data.duplicated(subset=cols_to_check_duplicates)

        # get col values that were duplicated
        duplicate_col_values_df = self.data[cols_to_check_duplicates][mask_duplicate_cols]

        # Group by cols to get unique pairs of duplicates
        unique_duplicate_df = duplicate_col_values_df.groupby(cols_to_check_duplicates).first().reset_index()

        if len(cols) == 1:
            column = cols_to_check_duplicates[0]
            duplicates = unique_duplicate_df[column].tolist()
        else:
            # Convert to a list of tuples
            duplicates = list(unique_duplicate_df[cols_to_check_duplicates].to_records(index=False))

        return duplicates

    def copy(self) -> object:
        """
        Returns a copy of the _BaseTable object
        
        Returns
        -------
        _BaseTable
            copy of the base table object
        """
        if self.loaded:
            data_copy = self.data.copy()
            return _BaseTable(data=data_copy,
                              index_col=None,
                              subject_col=self.subject_col,
                              incident_date_col=self.incident_date_col)

        else:
            raise ValueError("Data is not initialized yet")

