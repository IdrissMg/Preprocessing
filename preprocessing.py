# -*- coding: utf-8 -*-
"""
@author: idriss
"""

import pandas as pd
import numpy as np
import itertools 

from sklearn.base import TransformerMixin

def drop_zero_var(d):
    """ Droping columns where the variance is null - also for categorical columns"""
    def robust_var(col):
        if col.dtype.name == 'float64':
            return col.var()
        else:
            return col.nunique() - 1.0
    v = d.apply(robust_var, axis = 0)
    names = v.where(v==0).dropna().index
    return d.drop(labels = names, axis = 1)

class Discretizer(TransformerMixin):
    """
        Discretize numerical columns 
        
        Parameters : (mandatory for discretization of training set)
            names : columns on which to perform discretization
            bins : bins associated with column names
                
    """
    def fit(self, df, names = None, threshold = 50):        
        if names is None :
            self.cols = [c for c in df.columns if df[c].nunique() > threshold ]
        else :  
            self.cols= list(set(df.columns).intersection(names))
#            self.cols = list(set(effective_names).union([c for c in df.columns if df[c].nunique() > threshold ]))
        #equalsize
        self.equalsize = [ c for c in self.cols if df[c].value_counts(normalize=True).head(10).sum() > 0.8]
        #equalfrequency        
        self.equalquantile = list(set(self.cols).difference(self.equalsize))
        return self
    
    def equalsizeDiscretizer(self, col) :
        """ Bins of equal size
            Returns the new column and save bins in self.bins_output
        """
        if self.bins_input is None:
            nbins = max(1, 2*np.log(col.nunique()))
            newcol, self.bins_output[col.name] = pd.cut(col, nbins, retbins = True)
            #manually force infinite bounds 
            self.bins_output[col.name][0] = - np.inf
            self.bins_output[col.name][-1] = np.inf 
            newcol = pd.cut(col, self.bins_output[col.name], labels = False)
        else:
            newcol, self.bins_output[col.name] = pd.cut(col, self.bins_input[col.name], retbins = True, labels = False) 
        return newcol
    
    def equalquantileDiscretizer(self, col) :
        """ Bins based on quantiles (either decile or quartiles)
            Returns the new column withs numerical labels corresponding to the decile (resp. quartile)            
        """
        newcol, self.bins_output[col.name] = pd.qcut(col, 10, labels = False, retbins = True, duplicates = 'drop') 
        
        return newcol   
    
    def transform(self, df, bins = None):
        self.bins_input = None if bins is None else bins
        self.bins_output = {}
        df[self.equalsize] = df[self.equalsize].apply(self.equalsizeDiscretizer, axis = 0)
        df[self.equalquantile] = df[self.equalquantile].apply(self.equalquantileDiscretizer, axis = 0)   
        return df, self.bins_output

class DurationConverter(TransformerMixin):
    """
        Transforms all column dates into durations and sort by specified column
    """

    def fit(self, df):
        self.cols = list(df.select_dtypes(include=["datetime64[ns]"]).columns.values)
        return self

    def transform(self, df):
        newnames = []
        df.sort_values(by = 'dateneg', inplace = True)
        for i,j in itertools.combinations(self.cols, 2):
            newcol = abs(df[i] - df[j])
            if newcol.std() > np.timedelta64(1, 'Y'):
                df['Duration between '+i+' and ' +j] = newcol/np.timedelta64(1,'Y')
            elif newcol.std() < np.timedelta64(1, 'Y') and newcol.std() > np.timedelta64(1, 'M'):
                df['Duration between '+i+' and ' +j]= newcol/np.timedelta64(1,'M')
            else:
                df['Duration between '+i+' and ' +j]= newcol/np.timedelta64(1,'D')
            newnames.append('Duration between '+i+' and ' +j)
        df.drop(labels = self.cols, axis = 1, inplace = True)
        return df, newnames

class BaseTypeConverter(TransformerMixin):
    """
        A security class to convert all number in float.
        Int are converted to float if there is a null value in the column.
        To protect us against undefined behavior we use the learning phase to secure
        future processing by converting data to float.

        The class also keep the dtype of every column during the fit
    """

    def fit(self, df, y=None):
        cols = list(df.select_dtypes(include=[np.number]).columns.values)
        self.columns_type = {col: str(df[col].dtype) for col in list(df)}
        for col in cols:
            self.columns_type[col] = "float64"
        return self

    def transform(self, df, y=None):
        for col, dtype in self.columns_type.items():
            df[col] = df[col].astype(dtype)
        return df


class DateTypeConverter(TransformerMixin):
    """
        Will try to convert columns to datetime.
    """

    def fit(self, df, y=None):
        self.dates_cols = []
        self.date_format="%Y-%m-%d"

        for col in list(df):
            if df[col].dtype != 'object': 
                continue
            else:
                self.dates_cols.append(col)

        return self


    def transform(self, df, y=None):
        for col in self.dates_cols:
            try:
                tmp_df = pd.to_datetime(df[col], format=self.date_format)
            except Exception as e:
                try:
                    tmp_df = pd.to_datetime(df[col], infer_datetime_format=True)
                except:
                    continue
            df[col] = tmp_df
        return df

class BaseEncoder(TransformerMixin):
    """
        A class designed to apply one hot encoding on categorical columns 
        and to fill missing.
    """

    def fit(self, df, y=None):
        """
            Apply one hot encoding to categorical, and save the Missing key for all the columns
            except dates.
            Dates have to be processed in another place.
        """
        self.columns_type = {col: str(df[col].dtype) for col in list(df)}
        self.not_dates = [col for col in list(df) if 'datetime' not in str(df[col].dtype) ]
 
        self.categoricals = [name
                    for name in self.not_dates if str(df[name].dtype) == "category"]

        self.not_categoricals = [name for name in self.not_dates if  \
                str(df[name].dtype) != "category"]


        self.treatments = {}
        self.dummies_col_dict = {}

        missing_str = "MISSING"

        for col in self.categoricals:

            self.treatments[col] = missing_str
            tmp_df = df[col].cat.add_categories([missing_str])
            tmp_df = tmp_df.fillna(missing_str)

            dummy_tmp = pd.get_dummies(tmp_df, prefix = col, prefix_sep = "@")
            self.dummies_col_dict[col] = dummy_tmp.columns
        
        for col in self.not_categoricals:

            if str(df[col].dtype) == "object":
                self.treatments[col] = missing_str
            else:
                self.treatments[col] = 0

        return self


    def transform(self, df, y=None):
        """
            Apply one hot encoding and fill na with values learned during fit.
        """
        for col in list(df):
            df[col] = df[col].astype(self.columns_type[col])

        for col in self.categoricals:

            tmp_df  = df[col].cat.add_categories(self.treatments[col])
            tmp_df = tmp_df.fillna(self.treatments[col])

            dummy_new = pd.get_dummies(tmp_df, prefix=col, prefix_sep="@")
            tmp = dummy_new.reindex(columns=self.dummies_col_dict[col], fill_value=0)

            df = df.drop(col, axis=1)
            df[self.dummies_col_dict[col]] = tmp

        for col in self.not_categoricals:
            df[col] = df[col].fillna(self.treatments[col])
        return df
    
    
  