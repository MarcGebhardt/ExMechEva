# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:25:51 2022

@author: mgebhard
"""
import pandas as pd

def read_dicfile(path, m_type='Combined', sep=';', index_col=0):
    if m_type in ['Combined','Points']:
        df = pd.read_csv(path,
                        sep=sep,index_col=index_col,header=[0,1],
                        na_values={"Nan"," Nan","nan"})
        df.index = pd.Int64Index(df.index)
        df.columns.names = ['Points','Vars']
        col_s = df.columns.to_frame()['Vars'].str.startswith('Unnamed')
        df_s = df.loc(axis=1)[col_s]
        df_s.columns=df_s.columns.droplevel(1)
        df_s.columns.name=None
        df_p = df.loc(axis=1)[~col_s]
        df_p = df_p.sort_index(axis=1)
        return df_s, df_p
    elif m_type in ['2D','Simple']:
        df_s = pd.read_csv(path,
                sep=sep,index_col=index_col,header=0,
                na_values={"Nan"," Nan","nan"})
        df_s.index = pd.Int64Index(df_s.index)
        df_s.index.name = None
        return df_s
    else:
        raise NotImplementedError("Type %s not implemented!"%m_type)
    
    