# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:33:39 2021

@author: mgebhard
Changelog:
    - 22-06-13: YM_sigeps_lin: NaN handling, guess entfernt
"""
import os
import warnings
from functools import partial, wraps
import pandas as pd
import numpy as np
import lmfit
from scipy import stats
import scipy.signal as scsig
import scipy.interpolate as scint
import matplotlib.pyplot as plt

#%% Tests
if __name__ == "__import__":
    def FuncinMod(func, mod):
        if str(func) in dir(mod):
            o = True
        else:
            o = False
        return o
    
    _overrides=["indlim","meanwoso","stdwoso"]
    _impmods=[pd.Series,pd.DataFrame]
    for ta in _overrides:
        for tb in _impmods:
            if FuncinMod(func=ta, mod=tb):
                warnings.warn("Function %s in %s and will be overwritten!"%(ta,tb),
                              ImportWarning)
            # else:
            #     print("Function %s  not in %s and will be overwritten!"%(ta,tb))
    del _overrides, _impmods

#%% Output

def str_indent(po, indent=3):
    # if isinstance('s',str):
    #     pos = po
    # elif isinstance(po, pd.core.base.ABCSeries):
    #     pos = po.to_string()
    # elif isinstance(po, pd.core.base.ABCDataFrame):
    #     pos = po.to_string()
    # else:
    #     raise NotImplementedError("Type %s not implemented!"%type(po))
    pos = str(po)
    if not pos.startswith("\n"):
        pos = indent*" " + pos
        # pos = "\n" + pos
    poo = ("\n"+pos.replace("\n","\n"+indent*" "))
    # poo = (pos.replace("\n","\n"+indent*" "))
    return poo

def MG_strlog(s, logfp, output_lvl = 1, logopt=True, printopt=True):
    """
    Write lines to log and/or console.

    Parameters
    ----------
    s : string
        DESCRIPTION.
    output_lvl : positiv integer
        DESCRIPTION.
    logfp : string
        DESCRIPTION.

    """
    if output_lvl>=1:
        if logopt:      logfp.write(s)
        # if logopt:      logfp.writelines(s)
        # if printopt:    print(s)
        if printopt:    print(s, end='')
#%% Helper

def check_empty(x, empty_str=['',' ','  ','   ',
                              '#NV''NaN','nan','NA']):
    # t = (x == '' or (isinstance(x, float) and  np.isnan(x)))
    t = (x in empty_str or (isinstance(x, float) and  np.isnan(x)))
    return t

def str_to_bool(x):
    pt=["True","true","Yes","1","On"]
    if x is True:
        t = x
    elif x in pt:
        t = True
    else:
        t = False
    return t
#%% Numeric

def sigdig(x, sd=3):
    """
    Returns the significant digits of a floating number.

    Parameters
    ----------
    x : float
        Number.
    sd : int, optional
        Significant digits. The default is 3.

    Returns
    -------
    xr : float
        Rounded number to significant digits.

    """
    # import math
    # ndig = int(math.floor(math.log10(abs(x))))
    if ((x==0) or (abs(x)==np.inf) or (np.isnan(x))):
        ndig = 0
    else:
        ndig = int(np.floor(np.log10(abs(x))))
    rdig = sd - ndig - 1
    return rdig

def round_to_sigdig(x, sd=3):
    """
    Round a floating number to a number of significant digits.

    Parameters
    ----------
    x : float
        Number.
    sd : int, optional
        Significant digits. The default is 3.

    Returns
    -------
    xr : float
        Rounded number to significant digits.

    """
    # if ((x==0) or (abs(x)==np.inf) or (np.isnan(x))):
    #     ndig = 0
    # else:
    #     ndig = int(np.floor(np.log10(abs(x))))
    # rdig = sigdig - ndig - 1
    rdig = sigdig(x, sd)
    xr =  round(x, rdig)
    return xr


#%% Pandas
def pd_limit(self, iS=None, iE=None):
    """
    Limit a pandas Series or Dataframe to given indexes

    Parameters
    ----------
    iS : matching type of self.index, optional
        Start index. The default is None.
    iE : matching type of self.index, optional
        End index. The default is None.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Limited pandas object.

    """
    if not (iS is None):
        self = self.loc[iS:]
    if not (iE is None):
        self = self.loc[:iE]
    return self
# Overwrite pandas
pd.Series.indlim = pd_limit
pd.DataFrame.indlim = pd_limit

def pd_slice_index(index,vals,option='range'):
    if option=='range':
        ind=[]
        ind.append(index.searchsorted(vals[0], side='left'))
        ind.append(index.searchsorted(vals[-1], side='right'))
        ind_new=index[ind[0]:ind[-1]]
    elif option=='list':
        ind_new=index[index.searchsorted(vals)]
    else:
        raise NotImplementedError("option %s not implemented!"%option)
    return ind_new

def pd_combine_index(pd1, pd2, option='dropna'):
    """Returns a combined index of two pandas objects (series or dataframe) with valide values in both."""
    if option=='dropna':
        if isinstance(pd1,pd.core.indexes.base.Index):
            index1=pd1
        else:
            index1=pd1.dropna().index
        if isinstance(pd2,pd.core.indexes.base.Index):
            index2=pd2
        else:
            index2=pd2.dropna().index
    elif option=='fl_valid':
        if isinstance(pd1,pd.core.indexes.base.Index):
            index1=pd1
        else:
            f1 = pd1.first_valid_index()
            l1 = pd1.last_valid_index()
            index1=pd1.loc[f1:l1].index
        if isinstance(pd2,pd.core.indexes.base.Index):
            index2=pd2
        else:
            f2 = pd2.first_valid_index()
            l2 = pd2.last_valid_index()
            index2=pd2.loc[f2:l2].index
    else:
        raise NotImplementedError("option %s not implemented!"%option)
    ind_new = index1.join(index2, how='inner')
    return ind_new

def pd_valid_index(val, pdo, opt='ba', na_hand='Value', na_out=None, add_ind=None):
    """Returning valid index of pandas object in respect to defined option.
    (b=before,a=after,n=nearest,ba=before and after,bna=before, nearest and after)
    Filling not available values according a handler (na_hand) with 
    standard value (na_out) or a part of an additional index (add_ind)."""
    if isinstance(pdo,pd.core.indexes.base.Index):
        index=pdo
    elif (type(pdo) is pd.core.frame.DataFrame) or (type(pdo) is pd.core.series.Series):
        index=pdo.index
    elif isinstance(pdo,list):
        index=pd.Index(pdo)
    else:
        raise TypeError('Unexpected type of input (%s)'%type(pdo))
    
    def pd_valid_index_na(na_hand, na_out, pos, add_ind):
        """Not available handler for function pd_valid_index."""
        if na_hand=='Value':
            na_ret=na_out
        elif na_hand=='fi_la_add':
            if pos=='b':
                na_ret=add_ind[0]
            elif pos=='a':
                na_ret=add_ind[-1]
            elif pos=='n':
                na_ret=na_out
            else:
                raise ValueError("Position (%s) not valid!"%pos)
        else:
            raise NotImplementedError("NA-Handler (%s) not implemented!"%na_hand)
        return na_ret            
    
    try:
        ob=index[index.get_indexer_for([val])[0]-1]
    except IndexError:
        # ob=nf_out
        ob=pd_valid_index_na(na_hand=na_hand, na_out=na_out, 
                             pos='b', add_ind=add_ind)
    if ob == index[-1]: #No jump from first to last position
        # ob=nf_out
        ob=pd_valid_index_na(na_hand=na_hand, na_out=na_out, 
                             pos='b', add_ind=add_ind)
    try:
        oa=index[index.get_indexer_for([val])[0]+1]
    except IndexError:
        # oa=nf_out
        oa=pd_valid_index_na(na_hand=na_hand, na_out=na_out, 
                             pos='a', add_ind=add_ind)
    try:
        on=index[index.get_indexer_for([val])[0]]
    except IndexError:
        # on=nf_out
        on=pd_valid_index_na(na_hand=na_hand, na_out=na_out, 
                             pos='n', add_ind=add_ind)
        
    # out=nf_out
    if opt=='b':
        out=ob
    elif opt=='a':
        out=oa
    elif opt=='n':
        out=on
    elif opt=='ba':
        out=[ob,oa]
    elif opt=='bna':
        out=[ob,on,oa]
    else:
        raise NotImplementedError("Option %s not implemented!"%opt)
    return out
    
def pd_find_index(ser, s):
    if not isinstance(s, list):
        s=list(s)
    o = ser.loc[ser.Type.apply(lambda x: x in s)].index
    return o

def Find_closest(pds, val, iS=None, iE=None, option='abs'):
    """Returns the index of the closest value of a Series to a value"""
    if not ((iS is None) and (iE is None)):
        pds = pds.indlim(iS=iS,iE=iE)
    if option == 'abs':
        pdt = abs(pds-val)
    elif option == 'lower':
        pdt = abs(pds.where(pds < val)-val)
    elif option == 'higher':
        pdt = abs(pds.where(pds > val)-val)
    i=pdt.idxmin()
    return i

def Find_closest_perc(pds, p, iS=None, iE=None, 
                      range_sub='min', norm='max', option='abs'):
    """Returns the index of the closest value of a Series 
    to a percentage value in a range of this series"""
    if not ((iS is None) and (iE is None)):
        pds = pds.indlim(iS=iS,iE=iE)
    if range_sub == 'min':
        pds = pds-pds.min() 
    elif range_sub == 'max':
        pds = pds-pds.max()
    elif range_sub == 'iS':
        pds = pds-pds.loc[iS]
    elif range_sub == 'iE':
        pds = pds-pds.loc[iE]
    pdn = normalize(pdo=pds, norm=norm)
    if isinstance(p,float):
        i=Find_closest(pds=pdn, val=p, option=option)
    elif isinstance(p,list) or isinstance(p,np.ndarray) or isinstance(p, pd.core.base.ABCSeries):
        i=[]
        for t in p:
            i.append(Find_closest(pds=pdn, val=t, option=option))
    else:
        raise TypeError("Type %s not implemented!"%type(p))
    return i

def Find_closestv(pds1, pds2, val1, val2, iS=None, iE=None, option='quad'):
    """Returns the index of the closest value of two corresponding Series
    to a pair of values"""
    if not ((iS is None) and (iE is None)):
        pds1 = pds1.indlim(iS=iS,iE=iE)
        pds2 = pds2.indlim(iS=iS,iE=iE)
    if option == 'abs':
        pdt = abs(pds1-val1)+abs(pds2-val2)
    # elif option == 'quad': # 221014 - option größer/kleiner val1
    elif option in ['quad','quad_hieq','quad_loeq',
                    'quad_1hieq','quad_1loeq','quad_2hieq','quad_2loeq']:
        pdt = ((pds1-val1)**2+abs(pds2-val2)**2)**0.5
        if option == 'quad_1hieq':
            pdt = pdt.where(pds1 >= val1)
        elif option == 'quad_1loeq':
            pdt = pdt.where(pds1 <= val1)
        elif option == 'quad_2hieq':
            pdt = pdt.where(pds2 >= val2)
        elif option == 'quad_2loeq':
            pdt = pdt.where(pds2 <= val2)
    i=pdt.idxmin()
    return i

def Find_first_sc(pds, val, iS=None, iE=None, 
                  direction='normal', 
                  option='after', exclude_1st=True,
                  nan_policy='omit'):
    """Returns the index of the first value of a Series with change in sign to a value"""
    if not ((iS is None) and (iE is None)):
        pdt = pds.indlim(iS=iS,iE=iE).copy(deep=True)
    else:
        pdt=pds.copy(deep=True)
    pdt=pd_nan_handler(pdt, nan_policy=nan_policy)
    if direction=='reverse':
        pdt=pdt.iloc[::-1]
        
    tmp=sign_n_change(pdt-val)[1]
    if exclude_1st:
        tmp=tmp.iloc[1:] #first index always==True?!
    if option in ['before','b','bf']:
        i=tmp[tmp].index[0]
        # if direction=='reverse':
        #     i=pd_valid_index(i+1, tmp, opt='b')
        # else:
        #     i=pd_valid_index(i-1, tmp, opt='b')
        i=pd_valid_index(i-1, tmp, opt='b')
    elif option in ['after','a','af']:
        i=tmp[tmp].index[0] 
    return i

def pd_vec_length(pdo, norm=False, norm_kws={}, out='Series'):
    """Calculate (normalized) vector length and return ether whole series or index"""
    if norm:
        pds=normalize(pdo, **norm_kws)
    else:
        pds=pdo
    pds=((pds**2).sum(axis=1))**0.5
    if out=='Series':
        out=pds
    else:
        out=pds.idxmax()
    return out
    

def pd_outsort(data, outsort='ascending'):
    if outsort in ['A','a','ascending',True,'r','rising','rise']:
        dout = data.sort_values(ascending=True)
    elif outsort in ['D','d','descending','f','falling','fall']:
        dout = data.sort_values(ascending=False)
    else:
        dout = data
    return dout

def pd_axischange(axis):
    if axis==0:
        a=1
    elif axis==1:
        a=0
    if axis=="columns":
        a="index"
    elif axis=="index":
        a="columns"
    else:
        NotImplementedError("Axis %s not implemented!"%axis)
    return a

def pd_isDF(pdo):
    return isinstance(pdo, pd.core.base.ABCDataFrame)
def pd_isSer(pdo):
    return isinstance(pdo, pd.core.base.ABCSeries)

def pd_exclnan(pdo,axis=1):
    if pd_isDF(pdo):
        exm=pdo.isna().any(axis=axis)
    elif pd_isSer(pdo):
        exm=pdo.isna()
    else:
        NotImplementedError("Type %s not implemented!"%type(pdo))
    return pdo.loc[~exm]

def pd_nan_handler(pdo, ind=None, axis=0, nan_policy='omit'):
    """
    Handles NaN in pandas object according to NaN-policy.

    Parameters
    ----------
    pdo : pandas.Series or pandas.DataFrame
        Input Value.
    ind : (same as corresponding axis), optional
        Index limitation on not scanned axis. The default is None.
    axis : [0,1]/["index","columns"], optional
        Scanned axis. The default is 0.
    nan_policy : string, optional
        Policy to handle NaN. Implemented: omit, raise, interpolate. 
        The default is 'omit'.

    Raises
    ------
    ValueError
        Raise error if nan_policy is raise and NaN's detected.

    Returns
    -------
    pds : pandas.Series or pandas.DataFrame
        Output value with NaN handled acc. NaN-policy.

    """
    pds=pdo.copy(deep=True)
    if pd_isDF(pds):
        if not ind is None: pds=pds.loc(axis=pd_axischange(axis))[ind]
        if nan_policy=='omit':
            pds=pd_exclnan(pdo=pds, axis=pd_axischange(axis))
        elif nan_policy=='raise' and pds.isna().any(None):
            raise ValueError("NaN in objectiv and NaN-policy = %s!"%nan_policy)
        elif nan_policy=='interpolate':
            pds=pds.interpolate(axis=axis)
    elif pd_isSer(pds):
        if nan_policy=='omit':
            pds=pd_exclnan(pdo=pds)
        elif nan_policy=='raise' and pds.isna().any(None):
            raise ValueError("NaN in objectiv and NaN-policy = %s!"%nan_policy)
        elif nan_policy=='interpolate':
            pds=pds.interpolate()
    else:
        NotImplementedError("Type %s not implemented!"%type(pds))
    return pds

#%%% List Operations
def list_cell_compiler(ser,sep=',', replace_whitespaces=True, replace_nans=True):
    ser = ser.astype(str)
    if replace_whitespaces:
        ser = ser.agg(lambda x: x.replace(' ',''))
    ser = ser.agg(lambda x: x.split(sep))
    return ser

def list_interpreter(ser, inter_list, option = 'exclude'):
    if option == 'exclude':
        func = lambda x: False if len(set(x).intersection(set(inter_list)))>0 else True
    elif option == 'include':
        func = lambda x: True if len(set(x).intersection(set(inter_list)))>0 else False
    else:
        raise NotImplementedError('Option %s not implemented!'%option)
    bool_ser = ser.agg(func)
    return bool_ser

def list_ser_to_1D(series, drop_duplicates=True, 
                   sort_values=True, exclude=[]):
    new = pd.Series([x for _list in series for x in _list])
    if drop_duplicates: new = new.drop_duplicates()
    if sort_values: new = new.sort_values().values
    for e in exclude:
        # new = np.delete(new,e)
        new = np.delete(new,new==e)
    return new

def list_boolean_df(series, unique_items, as_int=False):
    # Create empty dict
    bool_dict = {}
    # Loop through all the tags
    for i, item in enumerate(unique_items):
        # Apply boolean mask
        bool_dict[item] = series.apply(lambda x: item in x)
    # Return the results as a dataframe
    df = pd.DataFrame(bool_dict)
    if as_int: df = df.astype(int)
    return df

#%%% Failure Codes
def Failure_code_format(fcstr, pattern="[A-Z][0-9][0-9][.][0-3]"):
    import re
    matched = re.match(pattern, fcstr)
    fc_b=bool(matched)
    return fc_b

def Failure_code_checker(fc_ser, exclude =['nan'], pattern="[A-Z][0-9][0-9][.][0-3]"):
    fcu=pd.Series(list_ser_to_1D(fc_ser,**{'drop_duplicates':True,
                                           'sort_values':True,
                                           'exclude':exclude}))
    fc_wrong   = fcu[fcu.apply(Failure_code_format)==False].values.tolist()
    fc_wrongid = list_interpreter(fc_ser, fc_wrong, option = 'include')
    fc_wrongid = fc_ser[fc_wrongid]
    return fc_wrongid, fc_wrong

def Failure_code_lister(_list, level=1, strength=[1,2,3],
                        drop_duplicates=True, replace_whitespaces=True):
    """ Shortens List of Failure-codes (Version 3.1) to levels and optional strength
    Level: 0=Procedure, 1=Procedure Failure_Type, 2=Porcedure.Failure_Type.Strength
           'P'=Procedure, 'F'=Failure_Type 
    """
    
    strength = [str(x) for x in strength]
    
    if replace_whitespaces:
        _list = [x.replace(' ','') for x in _list]
    if strength:
        _list = [i for i in _list if i.split('.')[-1] in strength]
        
    if level == 0 or level == 'P':
        new = [x[0] for x in _list]
    elif level == 1:
        new = [x.split('.')[0] for x in _list]
    elif level == 2:
        new = [x for x in _list]
    elif level == 'F':
        new = [x.split('.')[0][1:] for x in _list]
        
    if drop_duplicates:
        new = list(dict.fromkeys(new))
    return new

def Failure_code_bool_df(list_ser, sep=',', level=1, strength=[1,2,3],
                         drop_duplicates=True, sort_values=True,
                         exclude =['nan'],
                         replace_whitespaces=True, as_int=True):
    # list_ser = list_cell_compiler(series,sep,replace_whitespaces)
    list_ser = list_ser.apply(Failure_code_lister,**{'level':level,'strength':strength,
                                                     'drop_duplicates':drop_duplicates,
                                                     'replace_whitespaces':replace_whitespaces})
    uniques = list_ser_to_1D(list_ser,**{'drop_duplicates':drop_duplicates,
                                         'sort_values':sort_values,
                                         'exclude':exclude})
    df = list_boolean_df(list_ser,uniques,**{'as_int':as_int})
    return df, uniques


#%%% ICD Codes
def ICD_lister(_list, level=1,
               drop_duplicates=True, replace_whitespaces=True):
    """ Shortens List of ICD-codes to levels """
    if replace_whitespaces:
        _list = [x.replace(' ','') for x in _list]
    if level == 0:
        new = [x[0] for x in _list]
    elif level == 1:
        new = [x.split('.')[0] for x in _list]
    elif level == 2:
        new = [x for x in _list]
    if drop_duplicates:
        new = list(dict.fromkeys(new))
    return new

def ICD_bool_df(series, sep=',', level=1,
                drop_duplicates=True, sort_values=True,
                exclude =[],
                replace_whitespaces=True, as_int=True):
    list_ser = list_cell_compiler(series,sep,replace_whitespaces)
    list_ser = list_ser.apply(ICD_lister,**{'level':level,
                                            'drop_duplicates':drop_duplicates,
                                            'replace_whitespaces':replace_whitespaces})
    uniques = list_ser_to_1D(list_ser,**{'drop_duplicates':drop_duplicates,
                                         'sort_values':sort_values,
                                         'exclude':exclude})
    df = list_boolean_df(list_ser,uniques,**{'as_int':as_int})
    return df

#%%% HDF
def pack_hdf(in_paths, out_path, 
             hdf_naming = 'Designation', var_suffix = [""],
             h5_conc = 'Material_Parameters', h5_data = 'Measurement',
             opt_pd_out = True, opt_hdf_save = True):
    
    dfc=pd.DataFrame([],dtype='float64')
    dfd=pd.Series([],dtype='object')
    for p in in_paths.index:
        prot=pd.read_excel(in_paths.loc[p,'prot'],
                           header=11, skiprows=range(12,13),
                           index_col=0)
        for ms in var_suffix:
            for i in prot[hdf_naming].loc[np.invert(prot[hdf_naming].isna())]:
                if os.path.exists(in_paths.loc[p,'hdf']+i+ms+'.h5'):
                    data_read = pd.HDFStore(in_paths.loc[p,'hdf']+i+ms+'.h5','r')
                    df1 = data_read.select(h5_conc)
                    df2 = data_read.select(h5_data)
                    data_read.close()
                    dfp=prot.loc[[prot.loc[prot[hdf_naming]==i].index[0]]]
                    dfp.index=dfp.index+ms
                    df1=pd.concat([dfp,df1],axis=1)
                    dfc = dfc.append(df1)
                    # dfd = dfd.append(df2)
                    dfd[prot.loc[prot[hdf_naming]==i].index[0]+ms] = df2
                    del df1,df2
                else: # added 211028
                    dfp=prot.loc[[prot.loc[prot[hdf_naming]==i].index[0]]]
                    dfp.index=dfp.index+ms
                    dfc = dfc.append(dfp)
                    
            
    if opt_hdf_save:
        HDFst = pd.HDFStore(out_path+'.h5')
        HDFst['Summary'] = dfc
        HDFst['Test_Data'] = dfd
        HDFst.close()
    if opt_pd_out:
        return dfc, dfd
    
def pack_hdf_mul(in_paths, out_path, 
                 hdf_naming = 'Designation', var_suffix = [""],
                 h5_conc = 'Material_Parameters', h5_data = 'all',
                 opt_pd_out = True, opt_hdf_save = True):
    
    dfc=pd.DataFrame([],dtype='float64')
    dfd=pd.Series([],dtype='object')
    # dfd={}
    for p in in_paths.index:
        prot=pd.read_excel(in_paths.loc[p,'prot'],
                           header=11, skiprows=range(12,13),
                           index_col=0)
        for ms in var_suffix:
            for i in prot[hdf_naming].loc[np.invert(prot[hdf_naming].isna())]:
                if os.path.exists(in_paths.loc[p,'hdf']+i+ms+'.h5'):
                    data_read = pd.HDFStore(in_paths.loc[p,'hdf']+i+ms+'.h5','r')
                    df1 = data_read.select(h5_conc)
                    didks=data_read.keys()
                    for k in didks:
                        df2 = data_read.select(k)
                        if k not in dfd.keys():
                            dfd[k]=pd.Series([],dtype='object')
                        dfd[k][prot.loc[prot[hdf_naming]==i].index[0]+ms]=df2
                        del df2
                    data_read.close()
                    dfp=prot.loc[[prot.loc[prot[hdf_naming]==i].index[0]]]
                    dfp.index=dfp.index+ms
                    df1=pd.concat([dfp,df1],axis=1)
                    dfc = dfc.append(df1)
                    # dfd = dfd.append(df2)
                    # dfd[prot.loc[prot[hdf_naming]==i].index[0]+ms] = df2
                    del df1
                else: # added 211028
                    dfp=prot.loc[[prot.loc[prot[hdf_naming]==i].index[0]]]
                    dfp.index=dfp.index+ms
                    dfc = dfc.append(dfp)
    if opt_hdf_save:
        HDFst = pd.HDFStore(out_path+'.h5','w')
        HDFst['Summary'] = dfc
        # HDFst['Data'] = dfd
        for k in dfd.keys():
            HDFst['Add_'+k] = dfd[k]
        HDFst.close()
    if opt_pd_out:
        return dfc, dfd

#%% Evaluation
#%%% Measuring curve analysis and manipulation
def smooth(y, box_pts): 
    """
    Computes rolling mean of y about distance of box_pts.

    Parameters
    ----------
    y : array of float
        DESCRIPTION.
    box_pts : integer
        DESCRIPTION.

    Returns
    -------
    y_smooth : array of float
        DESCRIPTION.

    """
    warnings.warn("Method will be replaced by Smoothsel", DeprecationWarning)
    box = np.ones(box_pts)/box_pts 
    y_smooth = np.convolve(y, box, mode='same') 
    return y_smooth

def rise_curve(meas_curve, smoothbool, smooth_lvl):
    """ 
    Computes partial integration (1st[rise] and 2nd-grade[curvature])
    to find points of inconstancy in measured curves
    """ 
#    if smoothbool==True:
#        meas_curve=pd.Series(smooth(meas_curve,smooth_lvl),index=meas_curve.index)
    driC=meas_curve.diff()
    if smoothbool==True:
        driC=pd.Series(smooth(driC,smooth_lvl),index=driC.index)
    dcuC=driC.diff()
    driC_sign = np.sign(driC)
    driC_signchange = ((np.roll(driC_sign, True) - driC_sign) != False).astype(bool)
    driC_signchange = pd.Series(driC_signchange,index=driC.index)
    return (driC,dcuC,driC_signchange)

def sign_n_change(y):
    """
    Computes sign and sign change of an array.

    Parameters
    ----------
    y : pandas.Series of float
        Array which should analyzed.

    Returns
    -------
    y_sign : pandas.Series of integer
        Signs of y.
    y_signchange : pandas.Series of boolean
        Sign changes of y.

    """
    y_sign       = np.sign(y)
    y_signchange = ((np.roll(y_sign, True) - y_sign) != False).astype(bool)
    y_sign       = pd.Series(y_sign,index=y.index)
    y_signchange = pd.Series(y_signchange,index=y.index)
    return (y_sign,y_signchange)

def find_SandE(Val,Val2,drop_op,drop_val):
    """
    Computes first and last indices which fullfill the choosen condition.

    Parameters
    ----------
    Val : pd.Series([],dtype='float64')
        DESCRIPTION.
    Val2 : pd.Series([],dtype='float64')
        DESCRIPTION.
    drop_op : string, case-sensitive
        Condition name (abV_self,pgm_self,pgm_other,qua_self),
    drop_val : float64
        Value for condition.

    Returns
    -------
    iS : TYPE
        First index which fullfill the choosen condition.
    iE : TYPE
        Last index which fullfill the choosen condition.

    """
    if drop_op=="abV_self":
        iS=Val[abs(Val)>=drop_val].index[0]
        iE=Val[abs(Val)>=drop_val].index[-1]
    elif drop_op=="pgm_self":
        iS=Val[abs(Val)>=(abs(Val).max()*drop_val)].index[0]
        iE=Val[abs(Val)>=(abs(Val).max()*drop_val)].index[-1]
    elif drop_op=="pgm_other":
        iS=Val[abs(Val)>=(abs(Val2)*drop_val)].index[0]
        iE=Val[abs(Val)>=(abs(Val2)*drop_val)].index[-1]
    elif drop_op=="qua_self":
        iS=Val[abs(Val)>=(abs(Val).quantile(drop_val))].index[0]
        iE=Val[abs(Val)>=(abs(Val).quantile(drop_val))].index[-1]
    elif drop_op=="qua_other":
        iS=Val[abs(Val)>=(abs(Val2).quantile(drop_val))].index[0]
        iE=Val[abs(Val)>=(abs(Val2).quantile(drop_val))].index[-1]
    else:
        print("Error in find_SandE (case)!")
    return (iS,iE)

def Diff_Quot(meas_curve_A, meas_curve_B, 
              smoothbool, smooth_lvl, opt_shift=False):
    """
    Computes difference quotient (1st[rise] and 2nd-grade[curvature])
    to find points of inconstancy in measured curves.

    Parameters
    ----------
    meas_curve_A : pd.Series of float
        Abscissa of the curve for which the difference quotient is to be determined.
    meas_curve_B : pd.Series of float
        Ordinate of the curve for which the difference quotient is to be determined.
    smoothbool : bool
        Controles determination of rolling mean.
    smooth_lvl : positv integer
        Width of rolling mean determination.
    opt_shift : bool
        Shifting returned values to pointed input values.
        If False it points to the index before.
        Leading to a difficult to handle shift, ingreasing with difference quotient grade.
        Optional. The defalut is False.
        
    Test
    -------
    t=pd.DataFrame([[0,0],[1,0],[1.5,1],[2.5,2],[3.5,3],[4,3.5],
                    [5,2.5],[6,2],[7,2]], columns=['x','y'])
    DQ_df = pd.concat(Diff_Quot(t.x, t.y, False, 2, True), axis=1)
        "   DQ1 DQ1_signchange  DQ2 DQ2_signchange  DQ3 DQ3_signchange
         0  NaN           True  NaN           True  NaN           True
         1  0.0           True  NaN           True  NaN           True
         2  2.0          False  4.0           True  NaN           True
         3  1.0          False -1.0           True -5.0           True
         4  1.0          False  0.0          False  1.0           True
         5  1.0           True  0.0           True  0.0           True
         6 -1.0          False -2.0           True -2.0           True
         7 -0.5           True  0.5          False  2.5           True
         8  0.0            NaN  0.5            NaN  0.0            NaN"
        
    Returns
    -------
    DQ1 : pd.Series of float
        Differential quotient 1st grade (rise) of meas_curve_B to meas_curve_A.
    DQ1_signchange : pd.Series of bool
        Indicates sign changes of DQ1.
    DQ2 : pd.Series of float
        Differential quotient 2nd grade (curvature) of meas_curve_B to meas_curve_A.
    DQ2_signchange : pd.Series of bool
        Indicates sign changes of DQ2.
    DQ3 : pd.Series of float
        Differential quotient 3rd grade of meas_curve_B to meas_curve_A.
    DQ3_signchange : pd.Series of bool
        Indicates sign changes of DQ3.

    """
    warnings.warn("Method will be replaced by DiffQuot2", DeprecationWarning)
    if smoothbool==True:
        meas_curve_A = pd.Series(smooth(meas_curve_A,smooth_lvl),
                                 index=meas_curve_A.index)
        meas_curve_B = pd.Series(smooth(meas_curve_B,smooth_lvl),
                                 index=meas_curve_B.index)
    DQ1=meas_curve_B.diff() / meas_curve_A.diff() # m=dSpannung/dDehnung
    # if opt_shift:
    #     DQ1 = DQ1.shift(-1)
    if smoothbool==True:
        DQ1s = pd.Series(smooth(DQ1,smooth_lvl), index=DQ1.index)
    else:
        DQ1s = DQ1
    DQ2 = DQ1s.diff() / meas_curve_A.diff()
    # if opt_shift:
    #     DQ2 = DQ2.shift(-1)
    if smoothbool==True:
        DQ2s = pd.Series(smooth(DQ2,smooth_lvl), index=DQ2.index)
    else:
        DQ2s = DQ2
    DQ3 = DQ2s.diff() / meas_curve_A.diff()
    # if opt_shift:
    #     DQ3 = DQ3.shift(-1)
    
    DQ1_sign, DQ1_signchange, = sign_n_change(DQ1)
    DQ2_sign, DQ2_signchange, = sign_n_change(DQ2)
    DQ3_sign, DQ3_signchange, = sign_n_change(DQ3)
    
    DQ1.name='DQ1'
    DQ2.name='DQ2'
    DQ3.name='DQ3'
    DQ1_signchange.name='DQ1_signchange'
    DQ2_signchange.name='DQ2_signchange'
    DQ3_signchange.name='DQ3_signchange'
    if opt_shift:
        DQ1_signchange = DQ1_signchange.shift(-1, fill_value=False)
        DQ2_signchange = DQ2_signchange.shift(-1, fill_value=False)
        DQ3_signchange = DQ3_signchange.shift(-1, fill_value=False)
    return DQ1,DQ1_signchange,DQ2,DQ2_signchange,DQ3,DQ3_signchange

def test_pdmon(df,cols,m,dist):
    """
    Test of monotonic de-/increasing in dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe which includes cols as Column-names.
    cols : string or array of strings
        Column-names for monotonic test.
    m : np.int
        Kind of monotoni (1=increasing, -1=decreasing).
    dist : np.int
        test distance of monotonic creasing.

    Returns
    -------
    d : pd.Series([],dtype='float64')
        Series with index 'cols' of last 'df.index' with monotonic de-/increasing of 'dist'-length.
    """
    
    b=df.loc[:,cols].transform(lambda x: np.sign(x.diff()))
    d=pd.Series([],dtype='float64')
    for i in b.iloc[dist+1:].index:
        c=(m==b.loc[i-dist+1:i]).all(axis=0)
        for j in c.index:
            if c[j]:  d[j]=i
    return (d)

def normalize(pdo, axis=0, norm='absmax', normadd=0.5, pdo_n=None,
              warn_excl=['x_sc','dx_sc']):
    """
    Normalize an array in respect to given option.

    Parameters
    ----------
    pdo : pandas.Series or pandas.DataFrame
        Input values.
    axis : {0 or ‘index’, 1 or ‘columns’},, optional
        Axis to apply normalizing. The default is 0.
    norm : string, optional
        Apllied option for normalizing.
        Possible are:
            - 'absmax': Normalize to maximum of absolut input values.
            - 'absmin': Normalize to minimum of absolut input values.
            - 'absqua': Normalize to Quantile of absolut input values (specified with normadd).
            - 'max': Normalize to maximum of input values.
            - 'min': Normalize to minimum of input values.
            - 'qua': Normalize to Quantile of input values (specified with normadd).
            - 'val': Normalize to given value (specified with normadd).
            - None: No normalization (return input values).
        The default is 'absmax'.
    normadd : int or float, optional
        Additional parameter for option 'qua'(to set quantile value), 
        or 'val'(to set norm parameter directly).
        The default is 0.5.
    pdo_n : pandas.Series or pandas.DataFrame or None, optional
        Additional values to get normalizing parameter.
        If None use input values. If type is Dataframe, same shape required.
        The default is None.
    warn_excl : list
        Defines exclusions of ZeroDivisionError-prevention-warnings.
        The default is ['x_sc','dx_sc'].
    
    Raises
    ------
    NotImplementedError
        Option not implemented.

    Returns
    -------
    o : pandas.Series or pandas.DataFrame
        Normalized output values.

    """
    if (axis==1) and (norm in ['absmeanwoso','meanwoso']):
        raise NotImplementedError("Axis %s and norm %s not implemented!"%(axis,norm))
        
    if pdo_n is None:
        pdo_n = pdo
        
    if   norm == 'absmax':
        npar = abs(pdo_n).max(axis=axis)
    elif norm == 'absmin':
        npar = abs(pdo_n).min(axis=axis)
    elif norm == 'absmean':
        npar = abs(pdo_n).mean(axis=axis)
    elif norm == 'absmeanwoso':
        npar = abs(pdo_n).meanwoso()
    elif norm == 'absmed':
        npar = abs(pdo_n).median(axis=axis)
    elif norm == 'absqua':
        if isinstance(pdo_n, pd.core.base.ABCSeries):
            npar = abs(pdo_n).quantile(normadd)
        else:
            npar = abs(pdo_n).quantile(normadd,axis=axis)
            
    elif norm == 'max':
        npar = pdo_n.max(axis=axis)
    elif norm == 'min':
        npar = pdo_n.min(axis=axis)
    elif norm == 'mean':
        npar = pdo_n.mean(axis=axis)
    elif norm == 'meanwoso':
        npar = pdo_n.meanwoso()
    elif norm == 'med':
        npar = pdo_n.median(axis=axis)
    elif norm == 'qua':
        if isinstance(pdo, pd.core.base.ABCSeries):
            npar = pdo_n.quantile(normadd)
        else:
            npar = pdo_n.quantile(normadd,axis=axis)
            
    elif norm == 'val':
        if isinstance(pdo_n, pd.core.base.ABCDataFrame):
            npar = pd.Series([normadd,]*len(pdo_n.columns),
                             index=pdo_n.columns)
        else:
            npar = normadd
    elif norm is None:
        if isinstance(pdo_n, pd.core.base.ABCDataFrame):
            npar = pd.Series([1,]*len(pdo_n.columns),
                             index=pdo_n.columns)
        else:
            npar = 1
    else:
        raise NotImplementedError("Norming %s not implemented!"%norm)
    if isinstance(pdo_n, pd.core.base.ABCSeries):
        if npar == 0:
            npar = 1
            npar_0 = pdo_n.name
            if not npar_0 in warn_excl:
                warnings.warn('ZeroDivisionError prevented by setting norming to 1 for variable: {}'.format(npar_0))
    elif (npar == 0).any():
        npar[npar == 0] = 1
        # npar_0 = npar[npar == 0].index.values
        test=npar[npar==0].index.to_series().apply(lambda x: x not in warn_excl)
        if test.any():
            npar_0=npar[npar==0][test].index.tolist()
            warnings.warn('ZeroDivisionError prevented by setting norming to 1 for variables: {}'.format(npar_0))
    o = pdo/npar
    return o

def threshhold_setter(pdo, th=None, option='abs', set_val='th'):
    if set_val == 'th':
        set_val = th
    if th is None:
        return pdo
    if option == 'abs':
        o = pdo.where(pdo.abs() > th, set_val)
    elif option == 'lower':
        o = pdo.where(pdo       > th, set_val)
    elif option == 'higher':
        o = pdo.where(pdo       < th, set_val)
    else:
        raise NotImplementedError("Option %s not implemented!"%option)
    return o

def normalize_th(pdo, axis=0, pdo_n=None, norm='absmax', normadd=0.5,
                 th=None, th_option='abs', th_set_val='th',
                 warn_excl=['x_sc','dx_sc']):
    pds=normalize(pdo, axis=axis, norm=norm, normadd=normadd, pdo_n=pdo_n,
                  warn_excl=warn_excl)
    pdt=threshhold_setter(pds, th=th, option=th_option, set_val=th_set_val)
    return pdt

def sign_n_changeth(y, axis=0, norm=None, normadd=0.5,
                    th=None, th_option='abs', th_set_val=0,
                    rename=True, opt_out='Tuple'):
    """
    Computes sign and sign change of an array.

    Parameters
    ----------
    y : pandas.Series of float
        Array which should analyzed.

    Returns
    -------
    y_sign : pandas.Series of integer
        Signs of y.
    y_signchange : pandas.Series of boolean
        Sign changes of y.

    """
    def sign_tmp(x):
        if not x:
            o=x
        else:
            o=x // abs(x)
        return o
    def sign_ch_detect(i, std_val=False):
        if i == 0:
            o = False
        elif check_empty(i):
            o = std_val
        else:
            o = True
        return o
    if isinstance(y,np.ndarray):
        y = pd.Series(y)

    y = normalize(pdo=y, axis=axis, norm=norm, normadd=normadd)
    y = threshhold_setter(pdo=y, th=th, option=th_option, set_val=th_set_val)
    
    if isinstance(y, pd.core.base.ABCSeries):
        y_sign = y.apply(sign_tmp)
        y_signchange = y_sign.diff().apply(sign_ch_detect)
    else:
        y_sign = y.applymap(sign_tmp)
        y_signchange = y_sign.diff(axis=axis).applymap(sign_ch_detect)
        
    if rename:
        if isinstance(y, pd.core.base.ABCSeries):
            y_sign.name = y_sign.name+'_si'
            y_signchange.name = y_signchange.name+'_sc'
        else:
            if axis==0:
                y_sign.columns = y_sign.columns+'_si'
                y_signchange.columns = y_signchange.columns+'_sc'
            else:
                y_sign.index = y_sign.index+'_si'
                y_signchange.index = y_signchange.index+'_sc'
    if opt_out in ['DF','DataFrame','Dataframe']:
        out = pd.concat([y_sign,y_signchange], 
                        axis=1 if axis==0 else 0)
        return out
    elif opt_out=='Tuple':
        return y_sign, y_signchange
    else:
        raise NotImplementedError("Output option %s not implemented!"%opt_out)
        return y_sign, y_signchange

def Extend_Series_Poly(y, n=3, polydeg=1, kind='fb'):
    """
    Extend an Array or series with polynomial values.

    Parameters
    ----------
    y : numpy.array or pandas.Series
        Input values.
    n : int or list of or dict of kind{'f':int,'b':int}
        Number of points to extend.
        Must be at least equal to the order of the polynomial.
        Could be 0 (skipping extension).
    polydeg : int or list of or dict of kind{'f':int,'b':int}, optional
        Order of polynom. The default is 1.
    kind : str, optional
        Kind of extension ('f'-forward, 'b'-backward and 'fb'-both).
        The default is 'fb'.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    yout : numpy.array or pandas.Series
        Extended input values.

    """
    if isinstance(n, int):
        nf, nb = n, n
    elif isinstance(n, list):
        nf, nb = n
    elif isinstance(n, dict):
        nf, nb = n['f'],n['b']
    else:
        raise TypeError("Type %s for parameter n not implemented!"%type(n))
    if kind =='f': nb=0
    if kind =='b': nf=0
    if isinstance(polydeg, int):
        pdegf, pdegb = polydeg, polydeg
    elif isinstance(polydeg, list):
        pdegf, pdegb = polydeg
    elif isinstance(polydeg, dict):
        pdegf, pdegb = polydeg['f'], polydeg['b']
    else:
        raise TypeError("Type %s for parameter polydeg not implemented!"%type(polydeg))
        
    
    if isinstance(y, pd.core.base.ABCSeries):
        ya = y.values
        xa = y.index.values
        t = 'Series'
    elif isinstance(y, np.ndarray):
        ya = y
        ydt = ya.dtype
        xa = np.arange(len(y))
        t = 'Array'
    else:
        raise TypeError("Type %s not implemented!"%type(y))
        
    ydt = ya.dtype
    xdt = xa.dtype
        
    if (kind in ['f','fb']) and (nf > 0):
        yf_p = np.polyfit(xa[:nf+1],  ya[:nf+1],  pdegf)
        xf = xa[0] - np.cumsum(np.diff(xa[:nf+1]))[::-1]
        yf = np.polyval(yf_p, xf)
    else:
        xf = np.array([], dtype=xdt)
        yf = np.array([], dtype=ydt)
    if (kind in ['b','fb']) and (nb > 0):
        yb_p = np.polyfit(xa[-(nb+1):], ya[-(nb+1):], pdegb)
        xb = xa[-1] - np.cumsum(np.diff(xa[-1:-(nb+2):-1]))
        yb = np.polyval(yb_p, xb)
    else:
        xb = np.array([], dtype=xdt)
        yb = np.array([], dtype=ydt)

    yout = np.concatenate((yf, ya, yb), axis=0)
    xout = np.concatenate((xf, xa, xb), axis=0)
    if t == 'Series':
        yout = pd.Series(yout, index=xout, name=y.name)
    return yout

def Extend_Series_n_setter(ffunctype='Smooth', ffuncargs=(), ffunckwargs={}):
    ntmp=None
    if ffunctype=='Smooth':
        idks=['window_length','size','box','box_pts']
        for idk in idks:
            if idk in ffunckwargs.keys():
                ntmp=ffunckwargs[idk]
            else:
                no_idk=True
        if no_idk and isinstance(ffuncargs, list):
            for a in ffuncargs[::-1]:
                if type(a) is int:
                    ntmp = a
        elif no_idk and isinstance(ffuncargs,int):
            ntmp = ffuncargs
        else:
            ntmp = 1
        ntmp = (ntmp -1)//2
        
    elif ffunctype in ['Diff','diff','Shift','shift']:
        idks=['periods','shift','shift_value']
        for idk in idks:
            if idk in ffunckwargs.keys():
                ntmp=ffunckwargs[idk]
            else:
                no_idk=True
        if no_idk and isinstance(ffuncargs, list):
            for a in ffuncargs[::-1]:
                if type(a) is int:
                    ntmp = a
        elif no_idk and isinstance(ffuncargs,int):
            ntmp = ffuncargs
        ntmp = abs(ntmp)
    if ntmp is None:
        raise ValueError('Extension length not specified and no valid key found in smoothing options!')
    return ntmp

def Retrim_Series(y, axis=0, n=0, kind='fb'):
    """
    Retrim an Array or series after extension.

    Parameters
    ----------
    y : numpy.array or pandas.Series or pandas.DataFrame
        Input values.
    n : int or list of or dict of kind{'f':int,'b':int}
        Number of points which are extend.
        Could be 0 (skip retrimming).
    kind : str, optional
        Kind of extension ('f'-front, 'b'-back and 'fb'-both).
        The default is 'fb'.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    yout : numpy.array or pandas.Series
        Extended input values.

    """
    if isinstance(n, int):
        nf, nb = n, n
    elif isinstance(n, list):
        nf, nb = n
    elif isinstance(n, dict):
        nf, nb = n['f'],n['b']
    else:
        raise TypeError("Type %s for parameter n not implemented!"%type(n))

    if kind =='f': nb=0
    if kind =='b': nf=0

    if isinstance(y, pd.core.base.ABCSeries):
        yt='Series'
    elif isinstance(y, pd.core.base.ABCDataFrame):
        yt='DataFrame'
    elif isinstance(y, np.ndarray):
        yt='Array'
        y=pd.Series(y)
    else:
        raise TypeError("Type %s not implemented!"%type(y))
    indf =  nf if nf > 0 else None
    indb = -nb if nb > 0 else None
    if yt == 'DataFrame':
        yout = y.iloc(axis=axis)[indf:indb]
    else:
        yout = y.iloc[indf:indb]
    if yt == 'Array':
        yout = yout.values
        
    return yout

def check_params(test_val, test_var='n', 
                 func=Extend_Series_Poly, args=(), kwargs={'polydeg':1}):
    if func == Extend_Series_Poly:
        if test_var=='n' and test_val!=0:
            if 'polydeg' in kwargs.keys():
                ref_val = kwargs['polydeg']
            else:
                ref_val = 1 # min order of polynom
            test = (test_val/ref_val) >= 1
            adj = ref_val
            ref_var = 'polydeg'
            ref_cond = 'equal'
    else:
        raise NotImplementedError("Option %s not implemented!"%func)
    if test:
        leg_val = True
        adj_val = test_val
        leg_str = test_var+' is '+ref_cond+' then '+ref_var+' for '+str(func)
    else:
        leg_val = False
        adj_val = adj
        leg_str = test_var+' have to be '+ref_cond+' then '+ref_var+' for '+str(func)
        leg_str += '\n --> new value set to {}'.format(adj_val)
                
    return leg_val, adj_val, leg_str
    

def Predict_apply_retrim(x, afunc=None, pfunc=Extend_Series_Poly, 
                         aargs=[], akwargs={}, 
                         pargs=[], pkwargs={'n':3, 'polydeg':1, 'kind':'fb'},
                         shift_value=0, fill_value=None):
    """
    Predict input over their boundary values, apply a function and 
    retrim to original size.

    Parameters
    ----------
    x : numpy.array or pandas.Series
        Input values.
    afunc : function, optional
        Function for manipulation to apply after extending the input values. 
		The default is None.
    pfunc : function, optional
        Function for extending the input values. 
		The default is Extend_Series_Poly.
    aargs : list, optional
        Arguments passed to manipulation funtion (afunc). 
		The default is [].
    akwargs : dict, optional
        Keyword arguments passed to manipulation funtion (afunc). 
		The default is {}.
    pargs : list, optional
        Arguments passed to extening funtion (pfunc). 
		The default is [].
    pkwargs : dict, optional
        Keyword arguments passed to extening funtion (pfunc). 
		The default is {'n':1, 'polydeg':1, 'kind':'fb'}.
    shift_value : int, optional
        Value for shifting. The default is 0.
    fill_value : int or None, optional
        Value for filling free values after shifting. 
		The default is None, which apply self.dtype.na_value.

    Raises
    ------
    TypeError
        Not expected data type.

    Returns
    -------
    xout : numpy.array or pandas.Series
        Manipualted output values.

    """
       
    if isinstance(x, pd.core.base.ABCSeries):
        xt='Series'
    elif isinstance(x, np.ndarray):
        xt='Array'
    else:
        raise TypeError("Type %s not implemented!"%type(x))
    
    if isinstance(pkwargs['n'], int):
        nf, nb = pkwargs['n'], pkwargs['n']
    elif isinstance(pkwargs['n'], list):
        nf, nb = pkwargs['n']
    elif isinstance(pkwargs['n'], dict):
        nf, nb = pkwargs['n']['f'],pkwargs['n']['b']
    else:
        raise TypeError("Type %s for parameter n not implemented!"%type(pkwargs['n']))
        
    xlen = pfunc(x, *pargs, **pkwargs)
    if afunc is None:
        xman = xlen
    else:
        xman = afunc(xlen, *aargs, **akwargs)
        
    if xt == 'Array':
        xman = pd.Series(xman)
        
    if not shift_value == 0: xman = xman.shift(periods=shift_value, 
                                               fill_value=fill_value)
    indf =  nf if nf > 0 else None
    indb = -nb if nb > 0 else None
    xout = xman.iloc[indf:indb]
    if xt == 'Array':
        xout = xout.values
    
    return xout


def Smoothsel(x, smooth_type='SMA',
              # smooth_opts={'window_length':3, 'mode':'same'},
              smooth_opts={'window_length':3, 'mode':'nearest'},
              snip=False, conv_method='scipy'):
    """
    Computes a smoothed version of an input array, according to different smoothing types.

    Parameters
    ----------
    x : array or pandas.Series of float
        Input values.
    smooth_type : string, case-sensitive, optional
        Choosen smoothing type.
        Optional. The defalut is 'SMA'.
        Possible are:
            - 'SMA': Simple moving average based on numpy.convolve.
            - 'BMA': Moving average with binomial coefficents based on numpy.convolve.
            - 'SMA_f1d': Simple moving average based scipy.ndimage.filters.uniform_filter1d.
            - 'SavGol': Savitzky-Golay filter based on scipy.signal.savgol_filter.
    smooth_opts : dict, optional
        Keywords and values to pass to smoothing method. 
        For further informations see smooth_type and linked methods.
        Optional. The defalut is {'window_length':3, 'mode':'nearest'}.
    snip : bool or integer, optional
        Trimming of output. Either, if True with (window_length-1)//2 in smooth_opts,
        none if False, or with inserted distance. 
        The default is False.
    conv_method : string, optional
        Convolve method to use ('numpy' for numpy.convolve,
                                'scipy' for scipy.ndimage.convolve1d)
        The default is 'scipy'.

    Raises
    ------
    TypeError
        Type not expected.
    NotImplementedError
        Not implemented.

    Returns
    -------
    out : array or pandas.Series of float
        Output values.

    """
    if conv_method == 'numpy':
        conmet = np.convolve
        box_mode_std = 'same'
    elif conv_method == 'scipy':
        import scipy.ndimage as scnd
        conmet = scnd.convolve1d
        box_mode_std = 'nearest'
    elif conv_method == '2D':
        conmet = scsig.convolve2d
        box_mode_std = 'same'
        raise NotImplementedError("Convolving method %s not implemented!"%conv_method)
    else:        
        raise NotImplementedError("Convolving method %s not implemented!"%conv_method)
        
    if isinstance(x, pd.core.base.ABCSeries):
        #x = x.dropna() #???
        xa = x.values
        xt='Series'
        xi = x.index.values # nur für Kompatibilität USplineA
    elif isinstance(x, np.ndarray):
        xa = x
        xt='Array'
        xi = np.linspace(0,len(x),len(x)) # nur für Kompatibilität USplineA
    else:
        raise TypeError("Type %s not implemented!"%type(x))

    if smooth_type=='SMA':
        if isinstance(smooth_opts,int):
            box_pts  = smooth_opts
            box_mode = box_mode_std
        elif isinstance(smooth_opts,dict):
            box_pts  = smooth_opts['window_length']
            if not 'mode' in smooth_opts.keys():
                smooth_opts['mode'] = box_mode_std
            box_mode = smooth_opts['mode']
        else:
            raise TypeError("Unexpected type of smooth option (%s)."%type(smooth_opts))
        out=conmet(xa, np.ones(box_pts)/box_pts, mode=box_mode)
    
    elif smooth_type=='USplineA':
        box_pts  = 2 # Kompatibilität
        spline=scint.UnivariateSpline(xi, xa, **smooth_opts)
        out=spline(xi)
    
    elif smooth_type=='BMA':
        def binomcoeffs(n): 
            return (np.poly1d([0.5, 0.5])**n).coeffs
        if isinstance(smooth_opts,int):
            box_pts  = smooth_opts
            box_mode = box_mode_std
        elif isinstance(smooth_opts,dict):
            box_pts  = smooth_opts['window_length']
            if not 'mode' in smooth_opts.keys():
                smooth_opts['mode'] = box_mode_std
            box_mode = smooth_opts['mode']
        else:
            raise TypeError("Unexpected type of smooth option (%s)."%type(smooth_opts))
        out=conmet(xa, binomcoeffs(box_pts-1), mode=box_mode)
        
    elif smooth_type=='SMA_f1d':
        if isinstance(smooth_opts,int):
            box_pts  = smooth_opts
            orig = -(smooth_opts//2)
            box_mode = 'constant'
            smooth_opts  = {'size': smooth_opts, 'mode': box_mode,
                            'origin': orig}
        elif isinstance(smooth_opts,dict):
            box_pts  = smooth_opts['size']
            if not 'origin' in smooth_opts.keys():
                smooth_opts['origin'] = -(box_pts//2)
            if not 'mode' in smooth_opts.keys():
                smooth_opts['mode'] = 'constant'
        else:
            raise TypeError("Unexpected type of smooth option (%s)."%type(smooth_opts))
        import scipy.ndimage.filters as ndif
        out = ndif.uniform_filter1d(xa, **smooth_opts)
        
    elif smooth_type=='SavGol':
        box_pts  = smooth_opts['window_length']
        out = scsig.savgol_filter(xa,**smooth_opts)
    else:
        raise NotImplementedError("Smoothing type %s not implemented!"%smooth_type)
        
    if xt=='Series':
        out = pd.Series(out, index=x.index, name=x.name)
    if not snip is False:
        if snip is True:
            # out = out[:-(box_pts-1)]
            out = out[:-(box_pts-1)//2]
        else:
            out = out[:-snip]
    return out
    
def Smoothsel_ext(x, axis=0, smooth_type='SMA',
                  smooth_opts={'window_length':3, 'mode':'nearest'},
                  snip=False, conv_method='scipy',
                  pfunc=Extend_Series_Poly,
                  pkwargs={}, shift_value=0, fill_value=None,
                  so_idks=['window_length','size','box','box_pts']):
    
    akwargs = {'smooth_type':smooth_type, 'smooth_opts':smooth_opts,
               'snip':snip, 'conv_method':conv_method}
    
    if isinstance(x, pd.core.base.ABCSeries):
        xt='1D'
    elif isinstance(x, pd.core.base.ABCDataFrame):
        xt='2D'
    elif isinstance(x, np.ndarray):
        if len(x.shape) >1:
            raise TypeError("Array with shape %d not implemented (only 1D)!"%len(x.shape))
        else:
            xt='1D'
    else:
        raise TypeError("Type %s not implemented!"%type(x))
        
    if not 'n' in pkwargs.keys():
        if isinstance(smooth_opts,int):
            ntmp=smooth_opts
        elif isinstance(smooth_opts,dict):
            for so_idk in so_idks:
                if so_idk in smooth_opts.keys():
                    ntmp=smooth_opts[so_idk]
                else:
                    raise ValueError('Extension length not specified and no valid key found in smoothing options!')
        else:
            raise TypeError("Type %s not implemented to calculate extension length!"%type(smooth_opts))
        ntmp = (ntmp-1)//2
        n_check = check_params(test_val=ntmp, test_var='n',
                               func=Extend_Series_Poly, kwargs=pkwargs)
        if not n_check[0]:
            warnings.warn('\n Automatic set extension within Smoothsel_ext too small'+n_check[2])
            pkwargs['n'] = n_check[1]
        else:
            pkwargs['n']=ntmp
        
    if xt == '2D':
        xout = x.apply(Predict_apply_retrim, axis=axis,
                       **{'afunc':Smoothsel, 'pfunc':pfunc, 
                          'aargs':[], 'akwargs':akwargs, 
                          'pargs':[], 'pkwargs':pkwargs,
                          'shift_value':shift_value, 
                          'fill_value':fill_value})
    else:
        xout = Predict_apply_retrim(x, afunc=Smoothsel,
                                    pfunc=pfunc, 
                                    aargs=[], akwargs=akwargs, 
                                    pargs=[], pkwargs=pkwargs,
                                    shift_value=shift_value, 
                                    fill_value=fill_value)
                        
    return xout

def Diff_ext(x, periods=1, axis=0, pfunc=Extend_Series_Poly,
             pkwargs={}, shift_value=0, fill_value=None):
    
    if isinstance(x, pd.core.base.ABCSeries):
        xt='1D'
        afunc = pd.Series.diff
        akwargs = {'periods':periods}
    elif isinstance(x, pd.core.base.ABCDataFrame):
        xt='2D'
        afunc = pd.Series.diff
        akwargs = {'periods':periods}
    elif isinstance(x, np.ndarray):
        if len(x.shape) >1:
            raise TypeError("Array with shape %d not implemented (only 1D)!"%len(x.shape))
        else:
            xt='1D'
            afunc = np.diff
            akwargs = {'periods':periods, 'axis':axis, 
                       'prepend':[np.nan,]*periods}
    else:
        raise TypeError("Type %s not implemented!"%type(x))
        
    if not 'n' in pkwargs.keys():
        pkwargs['n']=abs(periods)
        
    n_check = check_params(test_val=pkwargs['n'], test_var='n',
                           func=Extend_Series_Poly, kwargs=pkwargs)
    if not n_check[0]:
        warnings.warn('\n Automatic set extension within Smoothsel_ext too small'+n_check[2])
        pkwargs['n'] = n_check[1]
        
    if xt == '2D':
        xout = x.apply(Predict_apply_retrim, axis=axis,
                       **{'afunc':afunc, 'pfunc':pfunc, 
                          'aargs':[], 'akwargs':akwargs, 
                          'pargs':[], 'pkwargs':pkwargs,
                          'shift_value':shift_value, 
                          'fill_value':fill_value})
    else:
        xout = Predict_apply_retrim(x, afunc=afunc,
                                    pfunc=pfunc, 
                                    aargs=[], akwargs=akwargs, 
                                    pargs=[], pkwargs=pkwargs,
                                    shift_value=shift_value, 
                                    fill_value=fill_value)
    return xout

def Diff_Quot2(x, y, 
               smooth_bool=False, smooth_type='SMA', 
               smooth_opts={'window_length':3}, 
               smooth_snip=False, 
               # sc_kwargs={'norm':None, 'normadd':0.5,
               #            'th':None, 'th_option':'abs', 'th_set_val':0},
               sc_kwargs={'norm':'absmax', 'normadd':0.5,
                          'th':0.05, 'th_option':'abs', 'th_set_val':0},
               opt_shift=False, opt_out='Tuple'):
    """
    Computes difference quotient of input arrays (1st[rise], 2nd[curvature] and 3rd-grade)
    to find points of inconstancy in measured curves.

    Parameters
    ----------
    x : pd.Series of float
        Abscissa of the curve for which the difference quotient is to be determined.
    y : pd.Series of float
        Ordinate of the curve for which the difference quotient is to be determined.
    smoothbool : bool or string
        Controles determination of rolling mean.
    smooth_type : string, case-sensitive, optional
        Choosen smoothing type.
        Optional. The defalut is 'SMA'.
        Possible are:
            - 'SMA': Moving average based on numpy.convolve.
            - 'SMA_f1d': Moving average based scipy.ndimage.filters.uniform_filter1d.
            - 'SavGol': Savitzky-Golay filter based on scipy.signal.savgol_filter.
    smooth_opts : dict, optional
        Keywords and values to pass to smoothing method. 
        For further informations see smooth_type and linked methods.
        Optional. The defalut is {'window_length':3, 'mode':'same'}.
    smooth_snip : bool or integer, optional
        Trimming of output. Either, if True with window_length in smooth_opts,
        none if False, or with inserted distance. 
        The default is False.
    opt_shift : bool
        Shifting returned values to pointed input values.
        If False it points to the index before.
        Leading to a difficult to handle shift.
        Optional. The defalut is False.
    opt_out : string
        Type of Output. Either Tuple or pandas dataframe.
        Optional. The defalut is 'Tuple'.
        
    Test
    -------
    t=pd.DataFrame([[0,0],[1,0],[1.5,1],[2.5,2],[3.5,3],[4,3.5],
                    [5,2.5],[6,2],[7,2]], columns=['x','y'])
    DQ_df = pd.concat(Diff_Quot(t.x, t.y, False, 2, True), axis=1)
        "   DQ1 DQ1_signchange  DQ2 DQ2_signchange  DQ3 DQ3_signchange
         0  NaN           True  NaN           True  NaN           True
         1  0.0           True  NaN           True  NaN           True
         2  2.0          False  4.0           True  NaN           True
         3  1.0          False -1.0           True -5.0           True
         4  1.0          False  0.0          False  1.0           True
         5  1.0           True  0.0           True  0.0           True
         6 -1.0          False -2.0           True -2.0           True
         7 -0.5           True  0.5          False  2.5           True
         8  0.0          False  0.5          False  0.0          False"
        
    Returns
    -------
    DQ1 : pd.Series of float
        Differential quotient 1st grade (rise) of y to x.
    DQ1_si : pd.Series of bool
       Signs of DQ1.
    DQ1_sc : pd.Series of bool
        Indicates sign changes of DQ1.
    DQ2 : pd.Series of float
        Differential quotient 2nd grade (curvature) of y to x.
    DQ2_si : pd.Series of bool
       Signs of DQ2.
    DQ2_sc : pd.Series of bool
        Indicates sign changes of DQ2.
    DQ3 : pd.Series of float
        Differential quotient 3rd grade (curvature change) of y to x.
    DQ3_si : pd.Series of bool
       Signs of DQ3.
    DQ3_sc : pd.Series of bool
        Indicates sign changes of DQ3.

    """
    
    if smooth_bool in [True,'Input','x-only','y-only']:
        if smooth_bool in [True,'Input','x-only']:
            x = Smoothsel(x=x, smooth_type=smooth_type,
                          smooth_opts=smooth_opts, snip=smooth_snip)
        if smooth_bool in [True,'Input','y-only']:
            y = Smoothsel(x=y, smooth_type=smooth_type,
                          smooth_opts=smooth_opts, snip=smooth_snip)
    DQ1=y.diff() / x.diff() # m=dOrdinate/dAbscissa
    # if opt_shift:
    #     DQ1 = DQ1.shift(-1)
    if smooth_bool==True:
        DQ1s = Smoothsel(x=DQ1, smooth_type=smooth_type,
                         smooth_opts=smooth_opts, snip=smooth_snip)
    else:
        DQ1s = DQ1
    DQ2 = DQ1s.diff() / x.diff()
    # if opt_shift:
    #     DQ2 = DQ2.shift(-1)
    if smooth_bool==True:
        DQ2s = Smoothsel(x=DQ2, smooth_type=smooth_type,
                         smooth_opts=smooth_opts, snip=smooth_snip)
    else:
        DQ2s = DQ2
    DQ3 = DQ2s.diff() / x.diff()
    # if opt_shift:
    #     DQ3 = DQ3.shift(-1)
    
    # DQ1_sign, DQ1_signchange, = sign_n_change(DQ1)
    # DQ2_sign, DQ2_signchange, = sign_n_change(DQ2)
    # DQ3_sign, DQ3_signchange, = sign_n_change(DQ3)
    
    # DQ1_signchange = DQ1_signchange.astype(bool)
    # DQ2_signchange = DQ2_signchange.astype(bool)
    # DQ3_signchange = DQ3_signchange.astype(bool)
    
    DQ1.name='DQ1'
    DQ2.name='DQ2'
    DQ3.name='DQ3'
    # DQ1_signchange.name='DQ1_signchange'
    # DQ2_signchange.name='DQ2_signchange'
    # DQ3_signchange.name='DQ3_signchange'
    
    DQ1_si, DQ1_sc, = sign_n_changeth(DQ1,**sc_kwargs)
    DQ2_si, DQ2_sc, = sign_n_changeth(DQ2,**sc_kwargs)
    DQ3_si, DQ3_sc, = sign_n_changeth(DQ3,**sc_kwargs)
    
    if opt_shift:
        DQ1 = DQ1.shift(-1, fill_value=np.nan)
        DQ2 = DQ2.shift(-1, fill_value=np.nan)
        DQ3 = DQ3.shift(-1, fill_value=np.nan)
        DQ1_si = DQ1_si.shift(-1, fill_value=np.nan)
        DQ2_si = DQ2_si.shift(-1, fill_value=np.nan)
        DQ3_si = DQ3_si.shift(-1, fill_value=np.nan)
        DQ1_sc = DQ1_sc.shift(-1, fill_value=False)
        DQ2_sc = DQ2_sc.shift(-1, fill_value=False)
        DQ3_sc = DQ3_sc.shift(-1, fill_value=False)
    if opt_out in ['DF','DataFrame','Dataframe']:
        out = pd.concat([DQ1,DQ1_si,DQ1_sc,
                         DQ2,DQ2_si,DQ2_sc,
                         DQ3,DQ3_si,DQ3_sc], axis=1)
        # temporäre Lösung
        out[['DQ1_sc','DQ2_sc','DQ3_sc']]=out[['DQ1_sc','DQ2_sc','DQ3_sc']].fillna(False)
        return out
    elif opt_out=='Tuple':
        return DQ1, DQ1_si, DQ1_sc, DQ2, DQ2_si, DQ2_sc, DQ3, DQ3_si, DQ3_sc
    else:
        raise NotImplementedError("Output option %s not implemented!"%opt_out)
        return DQ1, DQ1_si, DQ1_sc, DQ2, DQ2_si, DQ2_sc, DQ3, DQ3_si, DQ3_sc

def Diff_Quot3(y, x=None, deep=3, 
               ex_bool = True, ex_kwargs={'polydeg':1},
               smooth_bool=False, smooth_type='SMA', 
               smooth_opts={'window_length':3}, 
               sc_kwargs={'norm':'absmax', 'normadd':0.5,
                          'th':0.05, 'th_option':'abs', 'th_set_val':0},
               shift_value=False, opt_out='DataFrame'):

    if x is None:
        x=y.index
        
    if ex_bool:
        if 'n' not in ex_kwargs.keys():
            ntmp1 = ntmp2 = 0
            if smooth_bool in [True,'Input','x-only','y-only','yandDQ','DQ']:
                if isinstance(smooth_opts, dict):
                    ntmp1 = Extend_Series_n_setter('Smooth',
                                                   ffunckwargs=smooth_opts)
                elif isinstance(smooth_opts, list):
                    ntmp1 = Extend_Series_n_setter('Smooth',
                                                   ffuncargs=smooth_opts)
                elif isinstance(smooth_opts, int or float):
                    ntmp1 = Extend_Series_n_setter('Smooth',
                                                   ffuncargs=[smooth_opts])
                else:
                    raise TypeError("Type %s of smoothopts not implemented!(Determination extension length)")
            if shift_value or (shift_value !=0):
                if shift_value: shift_value = -1 
                ntmp2 = Extend_Series_n_setter('Shift',
                                               ffuncargs=shift_value)
            ntmp3 = Extend_Series_n_setter('Diff',ffuncargs=deep)
            ex_kwargs['n'] = max(ntmp1,ntmp2,ntmp3)
        if 'kind' not in ex_kwargs.keys(): 
            ex_kwargs['kind'] = 'fb'
        x = Extend_Series_Poly(x, **ex_kwargs)
        y = Extend_Series_Poly(y, **ex_kwargs)
    
    if smooth_bool in [True,'Input','x-only','y-only','yandDQ']:
        if smooth_bool in [True,'Input','x-only']:
            x = Smoothsel(x=x, smooth_type=smooth_type,
                          smooth_opts=smooth_opts, snip=False)
        if smooth_bool in [True,'Input','y-only','yandDQ']:
            y = Smoothsel(x=y, smooth_type=smooth_type,
                          smooth_opts=smooth_opts, snip=False)
    df=pd.concat([x,y],axis=1)
    df.columns=['x','y']
    df['dx'] = x.diff()
    df['dy'] = y.diff()
    DQs_tmp = df['y']
    for d in range(1,deep+1):
        DQ_name = 'DQ{}'.format(d) 
        df[DQ_name] = DQs_tmp.diff() / df['dx'] # m=dOrdinate/dAbscissa
        if smooth_bool in [True,'yandDQ','DQ']:
            DQs_tmp = Smoothsel(x=df[DQ_name], smooth_type=smooth_type,
                                smooth_opts=smooth_opts, snip=False)
        else:
            DQs_tmp = df[DQ_name]
            
    t = sign_n_changeth(df,**sc_kwargs,opt_out = 'DF')
    df = pd.concat([df,t],axis=1)
    if shift_value: shift_value = -1
    if shift_value != 0: 
        # df = df.shift(periods=shift_value, fill_value=None)
        # exclude original values (x and y)
        dfs=df.iloc(axis=1)[2:].shift(-1, fill_value=None)
        df[dfs.columns] = dfs
    if ex_bool:
        df = Retrim_Series(y=df, axis=0, n=ex_kwargs['n'], kind=ex_kwargs['kind'])
        
    if opt_out in ['DF','DataFrame','Dataframe']:
        out = df
        return out
    else:
        raise NotImplementedError("Output option %s not implemented!"%opt_out)
    
def Peaky_Finder(df, cols='all', norm='absmax',
                 fp_kwargs={'prominence':0.1, 'height':0.1},
                 out_opt='valser-loc'):
    """
    Find peaks in passed input data.

    Parameters
    ----------
    df : pandas DataFrame
        Input data.
    cols : string or array of strings or boolean, optional
        Columns (axis=1) of data to use. The default is 'all'.
    norm : string, optional
        Normalization method. The default is 'absmax'.
    fp_kwargs : dict, optional
        Keyword arguments to pass to scipy.signal.find_peaks.
        The default is {'prominence':0.1, 'height':0.1}.
    out_opt : string, optional
        Output options. The default is 'valser-loc'.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    o : Series of arrays
        Series with input-columns as index and array of input index as values.

    """
    if cols=='all':
        df2 = df
    else:
        # df2 = df[cols]
        df2 = df.loc(axis=1)[cols]
        
    # if norm == 'absmax':
    #     df2 = df2/abs(df2).max(axis=0)
    # elif norm == 'max':
    #     df2 = df2/df2.max(axis=0)
    # elif norm == 'min':
    #     df2 = df2/df2.min(axis=0)
    # elif norm is None:
    #     df2 = df2
    # else:
    #     raise NotImplementedError("Norming %s not implemented!"%norm)
    df2 = normalize(pdo=df2, axis=0, norm=norm)
    
    o=df2.agg(scsig.find_peaks,**fp_kwargs)
    if out_opt == 'complete-iloc':
        o = o
    elif out_opt == 'complete-loc':
        o.iloc[0] = o.iloc[0].apply(lambda x: df2.index[x].values)
    elif out_opt == 'valser-iloc':
        o = o.iloc[0]
    elif out_opt == 'valser-loc':
        o = o.iloc[0].apply(lambda x: df2.index[x].values)
    else:
        raise NotImplementedError("Output option %s not implemented!"%out_opt)
    
    return o

def Peaky_Finder_MM(df, cols='all', norm='absmax',
                    fp_kwargs={'prominence':0.1, 'height':0.1},
                    out_opt='valser-loc'):
    """Pacckage method of Peaky_Finder to find maxima and minima."""
    omax=Peaky_Finder(df= df, cols=cols, norm=norm,
                      fp_kwargs=fp_kwargs, out_opt=out_opt)
    omin=Peaky_Finder(df=-df, cols=cols, norm=norm,
                      fp_kwargs=fp_kwargs, out_opt=out_opt)
    return omax, omin

def Curvecar_Section(i_start, i_end, ref_ser, 
                     sm_w_length=0, threshold=0.01, norm='abs_max_ref',
                     out_vals=['','Const','Rise','Fall'], kind="median"):
    """
    Determine type of section by comparing a reference series to a given threshold.

    Parameters
    ----------
    i_start : corresponding type of index
        Start index of section.
    i_end : corresponding type of index
        End index of section.
    ref_ser : pands Series
        Reference series.
    sm_w_length : corresponding type of index, optional
        Determination distance to start and end. The default is 0.
    threshold : float, optional
        Determination threshold. The default is 0.01.
    norm : string, optional
        Nomralization method. The default is 'abs_max_ref'.
    out_vals : array of strings, optional
        Strings for determination type. The default is ['','Const','Rise','Fall'].

    Returns
    -------
    string
        Determined type of curve section.
    float
        Mean value over range of reference series.

    """
    i_sb = pd_valid_index(i_start+sm_w_length, ref_ser, opt='n')
    i_eb = pd_valid_index(i_end-sm_w_length, ref_ser, opt='n')
    if kind== "mean":
        mob_mean = ref_ser.loc[i_sb:i_eb].mean()
    elif kind== "median":
        mob_mean = ref_ser.loc[i_sb:i_eb].median()
    else:
        return NotImplementedError("Kind %s not implemented!"%kind)
    
    if norm == 'abs_max_ref':
        mob_mean_norm = mob_mean/abs(ref_ser).max()
    if norm == 'abs_med_ref':
        mob_mean_norm = mob_mean/abs(ref_ser).median()
    elif norm is None:
        mob_mean_norm = mob_mean 
    else:
        return NotImplementedError("Norm %s not implemented!"%norm)
    
    if  abs(mob_mean_norm) <= threshold:
        o = out_vals[1]
    elif mob_mean_norm > threshold:
        o = out_vals[2]
    elif mob_mean_norm < threshold:
        o = out_vals[3]
    else:
        o = out_vals[0]
    return o, mob_mean

def Curvecar_Refine(i_start,i_mid,i_end, ref_ser,
                    threshold=0.5, norm='abs_mid',
                    out_vals=['','Pos','Neg']):
    """
    Determine type, as well as new start and end points,
    of a refined section by comparing a reference series to a given threshold.

    Parameters
    ----------
    i_start : corresponding type of index
        Start index of first section.
    i_mid : corresponding type of index
        Index between sections.
    i_end : corresponding type of index
        End index of second section.
    ref_ser : pands Series
        Reference series.
    threshold : float, optional
        Determination threshold. The default is 0.5.
    norm : string, optional
        Nomralization method. The default is 'abs_mid'.
    out_vals : array of strings, optional
        Strings for determination type. The default is ['','Pos','Neg'].

    Returns
    -------
    string
        Determined type of curve section.
    corresponding type of index
        New start point of refined curve section.
    corresponding type of index
        New end point of refined curve section.
    float
        Mean value over range of reference series.

    """
    if norm == 'abs_mid':
        tn=abs(ref_ser/ref_ser.loc[i_mid])
    elif norm == 'abs_absmax_range':
        tn=abs(ref_ser/abs(ref_ser.loc[i_start:i_end]).max())
    elif norm == 'abs_absmax_complete':
        tn=abs(ref_ser/abs(ref_ser).max())
    elif norm == 'abs_absmed_complete':
        tn=abs(ref_ser/abs(ref_ser).median())
    elif norm == 'abs':
        tn=abs(ref_ser)
    elif norm is None:
        tn=ref_ser
    else:
        return NotImplementedError("Norm %s not implemented!"%norm)
        
    tnb=tn.loc[i_start:i_mid].iloc[::-1]
    tna=tn.loc[i_mid:i_end]
    try:
        i_snew=tnb.loc[tnb<=threshold].index[0]
    except IndexError:
        i_snew=None
    try:
        i_enew=tna.loc[tna<=threshold].index[0]
    except IndexError:
        i_enew=None
    if i_snew is None and i_enew is None:
        o = out_vals[0]
        return o, i_snew, i_enew, np.nan
    elif i_snew is None:
        i_snew=i_mid
    elif i_snew is None:
        i_enew=i_mid
    mean = ref_ser.loc[i_snew:i_enew].mean()
    if mean > 0:
        o = out_vals[1]
    elif mean < 0:
        o = out_vals[2]
    else:
        raise ValueError('mean is zero!')
    return o, i_snew, i_enew, mean

def Inter_Lines(r1,c1, r2,c2, out='x'):
    """
    Determine intersection point of two lines.

    Parameters
    ----------
    r1 : float
        Rise of first line.
    c1 : float
        Constant part of first line.
    r2 : float
        Rise of second line.
    c2 : float
        Constant part of second line.
    out : string, optional
        Option for output ('x', 'y', or 'xy'). The default is 'x'.

    Returns
    -------
    float or tuple of float
        Output values.

    """
    x = (c2-c1)/(r1-r2)
    if out == 'x':
        return x
    else: 
        y = r1 * x + c1
        if out == 'y':
            return x, y
        elif out == 'xy':
            return x, y


    
def MCurve_Characterizer(x, y,
                         ex_bool = True, ex_kwargs={'polydeg':1},
                         smooth_bool=True, smooth_type='SMA',
                         smooth_opts={'window_length':3},
                         # smooth_snip=True,  opt_shift=True,
                         shift_value=-1,
                         sc_kwargs={'norm':'absmax', 'normadd':0.5,
                                    'th':0.05, 'th_option':'abs', 'th_set_val':0},
                         peak_norm='absmax', peak_kwargs={'prominence':0.1, 'height':0.1}, 
                         cc_snip=3, cc_threshold = 0.01, cc_norm='abs_max_ref',
                         cc_refine=True, ccr_threshold=0.75, ccr_norm='abs_mid',
                         nan_policy='omit'):
    """
    Characterize a measured curve.

    Parameters
    ----------
    x : pandas Series of float, or pandas.DataFrame
        Abscissa or difference quotient dataframe of input.
    y : pandas Series of float
        Ordinate of input.
		
    smooth_bool : bool or string, optional
        Smoothing behavior. The default is True.
    smooth_type : string, case-sensitive, optional
        Choosen smoothing type.
        Optional. The defalut is 'SMA'.
        Possible are:
            - 'SMA': Moving average based on numpy.convolve.
            - 'SMA_f1d': Moving average based scipy.ndimage.filters.uniform_filter1d.
            - 'SavGol': Savitzky-Golay filter based on scipy.signal.savgol_filter.
    smooth_opts : dict, optional
        Keywords and values to pass to smoothing method. 
        For further informations see smooth_type and linked methods.
        Optional. The defalut is {'window_length':3, 'mode':'same'}.
    smooth_snip : bool or integer, optional
        Trimming of output. Either, if True with window_length in smooth_opts,
        none if False, or with inserted distance. 
        The default is False.
    opt_shift : bool
        Shifting returned values to pointed input values.
        If False it points to the index before.
        Leading to a difficult to handle shift.
        Optional. The defalut is False.
		
    peak_norm : string, optional
        Normalization method. The default is 'absmax'.
    peak_kwargs : dict, optional
        Keyword arguments to pass to scipy.signal.find_peaks.
        The default is {'prominence':0.1, 'height':0.1}.
		
    cc_snip : corresponding type of index, optional
        Determination distance to start and end for characterization of section. The default is 3.
    cc_threshold : float, optional
        Determination threshold for characterization of section. The default is 0.01.
    cc_norm : string, optional
        Nomralization method for characterization of section. The default is 'abs_max_ref'.
		
    cc_refine : bool, optional
        Turn refinement by curvature of curve section on. The default is True.
    ccr_threshold : float, optional
        Determination threshold of refinement. The default is 0.75.
    ccr_norm : string, optional
        Nomralization method of refinement. The default is 'abs_mid'.

    Returns
    -------
    pandas Dataframe
        Characterization of input (for linear parts: linearisation and intersection to previous included).
    pandas Series
        Points of first characteriation(S=Start,E=End, I=Increase, D=Decrease).
    pandas Dataframe
        Maxima and minima of input, as well as differential quotients.
    pandas Dataframe
		Input, as well as differential quotients.
    """
    if nan_policy in ['i', 'interpolate']:
        x = x.interpolate(limit_direction='both') if x.isna().any() else x
        y = y.interpolate(limit_direction='both') if y.isna().any() else y
        
    if isinstance(x, pd.core.base.ABCDataFrame):
        df = x.copy(deep=True)
    else:
        df = Diff_Quot3(y=y, x=x, deep=3, ex_bool=ex_bool, ex_kwargs=ex_kwargs,
                        smooth_bool=smooth_bool, smooth_type=smooth_type,
                        smooth_opts=smooth_opts,
                        sc_kwargs=sc_kwargs,
                        shift_value=shift_value, opt_out='DataFrame')
    
    # Find peaks
    a,b=Peaky_Finder_MM(df=df, cols='all', norm=peak_norm,
    # # exclude sign and n
    # a,b=Peaky_Finder_MM(df=df, cols=~df.columns.str.contains('_s.*',regex=True),
                        fp_kwargs=peak_kwargs, out_opt='valser-loc')
    def Max_or_Min(col,ind,max_p_ser,min_p_ser):
        if ind in max_p_ser[col]:
            o='Max'
        elif ind in min_p_ser[col]:
            o='Min'
        else:
            o=''
        return o
    dfp = df.apply(lambda x: pd.DataFrame(x).apply(lambda y: Max_or_Min(x.name, y.name, a, b), axis=1))
    # dfp = df.apply(lambda x: pd.DataFrame(x).apply(lambda y: Max_or_Min(x.name, y.name, a, b), axis=1))
    dfp = dfp.loc[np.invert((dfp=='').all(axis=1))]
    
    #pre-identify linear sections
    cps = pd.DataFrame([], columns=['Type','Start','End','Special'], dtype='O')
    cip = pd.Series([], dtype=str)
    j = 0
    ib=df.index[0]
    cip[ib]='S'
    ind_DQ2MM = dfp.loc[(dfp.DQ2 == 'Max') | (dfp.DQ2 == 'Min')].index
    for i in ind_DQ2MM:
        cps.loc[j,'Start']=ib
        cps.loc[j,'End']=i
        if dfp.loc[i,'DQ2'] == 'Max':
            cip[i]='I'
        else:
            cip[i]='D'
        t1,t2=Curvecar_Section(i_start=ib, i_end=i, ref_ser=df.DQ1,
                               sm_w_length=cc_snip,
                               threshold=cc_threshold, norm=cc_norm)
        cps.loc[j,'Type']= t1
        cps.loc[j,'Special']= t2
        ib = i
        j += 1
        if i == ind_DQ2MM[-1]:
            ia=df.index[-1]
            cps.loc[j,'Start']=i
            cps.loc[j,'End']=ia
            t1,t2=Curvecar_Section(i_start=i, i_end=ia, ref_ser=df.DQ1,
                                   sm_w_length=cc_snip,
                                   threshold=cc_threshold, norm=cc_norm)
            cps.loc[j,'Type']= t1
            cps.loc[j,'Special']= t2
            cip[ia]='E'
            
    if cc_refine:
        cps_new = cps.copy(deep=True)
        cps_i_crf=cps.loc[cps.Type.apply(lambda x: x in ['Const','Rise','Fall'])].index
        for i in cps_i_crf[:-1]:
            t=Curvecar_Refine(cps_new.loc[i,'Start'],cps_new.loc[i,'End'],
                              cps_new.loc[i+1,'End'],
                              df['DQ2'], threshold=ccr_threshold,
                              norm=ccr_norm, out_vals=['','Pos','Neg'])
            if t[0] != '':
                cps_new.loc[i,'End']=t[1]
                cps_new.loc[i+1,'Start']=t[2]
                cps_new.loc[i+0.1,['Type','Start','End','Special']]=t
        cps=cps_new.sort_index()
        t=cps.loc[cps_i_crf].apply(lambda i: Curvecar_Section(i_start=i.Start,
                                                              i_end=i.End,
                                                              ref_ser=df.DQ1,
                                                              sm_w_length=cc_snip,
                                                              threshold=cc_threshold,
                                                              norm=cc_norm),
                                   axis=1, result_type='expand')
        t.columns=['Type','Special']
        cps.loc[t.index,t.columns]=t
        
    # leat-squares-fit on linear sections
    cps_i_crf=cps.loc[cps.Type.apply(lambda x: x in ['Const','Rise','Fall'])].index
    t = cps.loc[cps_i_crf].apply(lambda i: YM_sigeps_lin(df.y, df.x,
                                                         ind_S=i.Start,
                                                         ind_E=i.End,
                                                         nan_policy='propagate'),
                                     axis=1, result_type='expand')
    t.columns=['r','c','Rq','lmfR']
    cps[['r','c','Rq']]=t[['r','c','Rq']]
    
    #Find points of intersection for linear sections
    # for i in cps.index:
    for i in cps_i_crf:
        if i==cps.index[0]:
            # cps.loc[i,'IntP_x'] = inter_lines(np.inf, df.loc[0,'x'],
            #                                   cps.loc[i  ,'r'],cps.loc[i  ,'c'], out='x')
            cps.loc[i,'IntP_x'] = df.iloc[0]['x']
        else:
            cps.loc[i,'IntP_x']=Inter_Lines(cps.loc[i-1,'r'],cps.loc[i-1,'c'],
                                            cps.loc[i  ,'r'],cps.loc[i  ,'c'], out='x')
        if i==cps.index[-1]:
            cps.loc[i+1,'Type'] = 'Last'
            cps.loc[i+1,'Start'] = cps.loc[i+1,'End']  = cps.loc[i,'End']
            cps.loc[i+1,['r','c']] = cps.loc[i,['r','c']]
            cps.loc[i+1,'IntP_x'] = df.iloc[-1]['x']
    cps['IntP_y'] = cps['IntP_x']*cps['r']+cps['c']
    
    return cps, cip, dfp, df

def MCurve_Char_Plotter(cps, cip, dfp, df,
                        head=None, 
                        xlabel=None, ylabel_l=None, ylabel_r=None, 
                        cco={'y':'m'},
                        cco_sc={},
                        ccd={'DQ1':'b','DQ2':'g','DQ3':'y'},
                        ccd_sc={'DQ1_sc':'b','DQ2_sc':'g','DQ3_sc':'y'},
                        cc={'Const':'b','Rise':'g','Fall':'r','Pos':'m','Neg':'y'},
                        disp_opt_DQ='Normalized', 
                        do_kwargs={'norm':'absmax', 'normadd':0.5,
                                   'th':0.05, 'th_option':'abs', 'th_set_val':0},
                        limDQ=False, limDQvals=[-1.1,1.1]):
    """Plotting method for MCurve_Characterizer results."""
    if disp_opt_DQ == 'True':
        df_u_DQ = df
    else:
        df_n = normalize(df.loc(axis=1)[~df.columns.str.contains('_s.*',regex=True)],
                         norm=do_kwargs['norm'], 
                         normadd=do_kwargs['normadd'])
        if disp_opt_DQ == 'Normalized':
            df_u_DQ = df_n
        elif disp_opt_DQ == 'Normalized_th':
            df_u_DQ = threshhold_setter(pdo=df_n, th=do_kwargs['th'],
                                        option=do_kwargs['th_option'],
                                        set_val=do_kwargs['th_set_val'])
        else:
            raise NotImplementedError("Display option %s not implemented!"%disp_opt_DQ)
    
    fig, (ax1,ax3) = plt.subplots(nrows=2, ncols=1, 
                                  sharex=True, sharey=False, 
                                  figsize = (6.3,2*3.54))
    fig.suptitle(head)
    ax1.set_title('Curve, difference quotients and extrema')
    ax1.plot(df.x, df.y,'r-', label='x-y')
    for i in cco:
        a = dfp[dfp[i]=='Max'].index
        b = dfp[dfp[i]=='Min'].index
        ax1.plot(df.loc[a,'x'],  df.loc[a,'y'], 
                 color=cco[i], marker='^', linestyle='')
        ax1.plot(df.loc[b,'x'],  df.loc[b,'y'], 
                 color=cco[i], marker='v', linestyle='')
    for i in cco_sc:
        ax1.plot(df.loc[df[i],'x'],  df.loc[df[i],i.replace('_sc','')], 
                 color=cco_sc[i], marker='x', linestyle='')
    ax1.grid()
    ax1.set_ylabel(ylabel_l)
    ax2=ax1.twinx()
    for i in ccd:
        a = dfp[dfp[i]=='Max'].index
        b = dfp[dfp[i]=='Min'].index
        ax2.plot(df.x, df_u_DQ[i],
                 ccd[i]+'--', label=i)
        ax2.plot(df.loc[a,'x'], df_u_DQ.loc[a,i],  
                 color=ccd[i], marker='^', linestyle='')
        ax2.plot(df.loc[b,'x'], df_u_DQ.loc[b,i],  
                 color=ccd[i], marker='v', linestyle='')
    for i in ccd_sc:
        ax2.plot(df.loc[df[i],'x'], df_u_DQ.loc[df[i],i.replace('_sc','')], 
                 color=ccd_sc[i], marker='x', linestyle='')
    ax2.axhline(0.0, color='gray', linestyle=':')
    ax2.axhline(1.0, color='gray', linestyle=':')
    ax2.axhline(-1.0, color='gray', linestyle=':') 
    if limDQ:
        ax2.set_ylim(limDQvals)
    ax2.set_ylabel(ylabel_r) 
    
    ax3.set_title('Curve characterisation')
    ax3.plot(df.x, df.y,'r-', label='x-y')
    for i in cps.index[:-1]:
        i_S=cps.loc[i,'Start']
        i_E=cps.loc[i,'End']
        ax3.plot(df.loc[i_S:i_E].x, df.loc[i_S:i_E].y,
                 cc[cps.loc[i,'Type']]+'-', linewidth=10,
                 solid_capstyle='butt', alpha=0.2,
                 label=cps.loc[i,'Type'])
    for i in cip.index:
        ax3.plot(df.loc[i].x, df.loc[i].y, color='k', marker='$%s$'%cip[i])
    cps_i_crf=cps.loc[cps.Type.apply(lambda x: x in ['Const','Rise','Fall','Last'])].index
    ax3.plot(cps.loc[cps_i_crf,'IntP_x'], cps.loc[cps_i_crf,'IntP_y'],
             'b:', label='lin')
    ax3.grid()
    ax3.set_ylabel(ylabel_l)
    ax3.set_xlabel(xlabel)
    # handles, labels = fig.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    for a in [ax1,ax2,ax3]:
        handles, labels = a.get_legend_handles_labels()
        if a==ax1:
            by_label = dict(zip(labels, handles))
        else:
            by_label.update(dict(zip(labels, handles)))
    fig.legend(by_label.values(), by_label.keys(),
               loc='lower right', bbox_to_anchor=(1, 0.25))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show(block=False)
    plt.close(fig)


def MCurve_merger(cps1,cps2, how='1stMax', option='nearest'):
    """
    

    Parameters
    ----------
    cps1 : pandas Dataframe (result of MCurve_Characterizer)
        First curve characterization.
    cps2 : pandas Dataframe (result of MCurve_Characterizer)
        Second curve characterization.
    how : string, optional
        Switch. The default is '1stMax'.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    t_off : float
        Offset to apply on first to merge second.
    t1 : float or pd.Series
        Timestamp/-s of first curve characterization.
    t2 : float or pd.Series
        Timestamp/-s of second curve characterization.

    """
    if how == '1stMax':
        t1=cps1.loc[cps1.Type=='Fall','IntP_x'].iloc[0]
        t2=cps2.loc[cps2.Type=='Fall','IntP_x'].iloc[0]
    elif how == 'wo_1st+last':
        t1=cps1.IntP_x.iloc[1:-1]
        t2=cps2.IntP_x.iloc[1:-1]
    elif how == 'complete':
        t1=cps1.IntP_x
        t2=cps2.IntP_x
    else:
        howt=how.split(';')
        if howt[0] == 'list':
            j=0
            how1=howt[1].split(',')
            for how2 in how1:
                how1[j] = int(how2)
                j+=1
            t1=cps1.IntP_x.iloc[how1]
            t2=cps2.IntP_x.iloc[how1]
        elif howt[0] == 'int':
            how1=int(howt[1])
            t1=cps1.IntP_x.iloc[how1]
            t2=cps2.IntP_x.iloc[how1]
        elif howt[0] == 'fix':
            t1=float(howt[1])
            t2=0.0
        else:
            raise NotImplementedError("Method %s not implemented!"%how)

    if isinstance(t1, float) and isinstance(t2, float):
        t_off=t1-t2
    else:
        if option.split('_')[0]=='nearest':
            t=t2.apply(lambda x: Find_closest(t1,x))
            tt=t.duplicated(False)
            if tt.any():
                for i in tt[tt].index:
                    ti=Find_closest(t2,t1[t[i]])
                    # print(i,ti)
                    if i!= ti and t[i] == t[ti]:
                        # print('drop',i)
                        t.drop(i,inplace=True)
            t1=t1[t.values]
            t2=t2[t.index]
            t_off=np.mean(t1.values-t2.values)
        else:
            t_off=(t1-t2).mean()
    return t_off, t1, t2

#%%% Fitting
def YM_eva_range_refine(m_df, VIP, n_strain, n_stress,
                         n_loBo='S', n_upBo='U',
                         d_loBo=0.05, d_max=0.75, 
                         rise_det=[True,4],
                         n_Outlo='F3',n_Outmi='FM',n_Outhi='F4'):
    """
    Refines the Youngs Modulus determinition range according to 
    "Keuerleber, M. (2006) - Bestimmung des Elastizitätsmoduls von Kunststoffen
    bei hohen Dehnraten am Beispiel von PP. Von der Fakultät Maschinenbau der 
    Universität Stuttgart zur Erlangung der Würde eines Doktor-Ingenieurs (Dr.-Ing.) 
    genehmigte Abhandlung. Doktorarbeit. Universität Stuttgart, Stuttgart"

    Parameters
    ----------
    m_df : pd.DataFrame
        Measured data.
    VIP : pd.Series
        Important points corresponding to measured data.
    n_strain : string
        Name of used strain (have to be in measured data).
    n_stress : string
        Name of used stress (have to be in measured data).
    n_loBo : string
        Lower border for determination (have to be in VIP). The default is 'S'.
    n_upBo : string
        Upper border for determination (have to be in VIP). The default is 'U'.
    d_loBo : float/str, optional
        When float: Percentage of range between n_upBo and n_loBo as start distance to n_loBo.
        When str starting with 'S', followed by integer: Distance in steps to n_loBo.
        The default is 0.05.
    d_max : float, optional
        Percentage of . The default is 0.75.
    rise_det : [bool, int], optional
        Determination options for stress rising ([smoothing, smoothing factor]).
    n_Outlo : string, optional
        Name of new lower border. The default is 'F3'.
    n_Outmi : string, optional
        Name of maximum differential quotient. The default is 'FM'.
    n_Outhi : string, optional
        Name of new upper border. The default is 'F4'.

    Yields
    ------
    VIP_new : pd.Series
        Important points corresponding to measured data.
    txt : string
        Documantation string.
    """
    if isinstance(d_loBo, str):
        if d_loBo.startswith('S'):
            i_loBo = int(d_loBo[1:])
        else:
            raise ValueError("Distance to lower border %s seems to be a string, but doesn't starts with 'S'"%d_loBo)
        Lbord = VIP[n_loBo]+i_loBo
        ttmp = m_df.loc[VIP[n_loBo]+i_loBo:VIP[n_upBo],n_stress]
    else:
        ttmp = m_df.loc[VIP[n_loBo]:VIP[n_upBo],n_stress]
        ftmp=float(ttmp.iloc[0]+(ttmp.iloc[-1]-ttmp.iloc[0])*d_loBo)
        Lbord=abs(ttmp-ftmp).idxmin()

    DQdf=pd.concat(Diff_Quot(m_df.loc[:,n_strain], m_df.loc[:,n_stress],
                              rise_det[0], rise_det[1]), axis=1)
    DQdf=m_df.loc(axis=1)[[n_strain,n_stress]].join(DQdf,how='outer')
    DQdfs=DQdf.loc[Lbord:VIP[n_upBo]]
    
    
    VIP_new = VIP
    txt =""
    VIP_new[n_Outmi]=DQdfs.DQ1.idxmax()
    try:
        VIP_new[n_Outlo]=DQdfs.loc[:VIP_new[n_Outmi]].iloc[::-1].loc[(DQdfs.DQ1/DQdfs.DQ1.max())<d_max].index[0]+1
    except IndexError:
        VIP_new[n_Outlo]=DQdfs.index[0]
        txt+="%s set to start of diff.-quot. determination "%n_Outlo
    try:
        VIP_new[n_Outhi]=DQdfs.loc[VIP_new[n_Outmi]:].loc[(DQdfs.DQ1/DQdfs.DQ1.max())<d_max].index[0]-1
    except IndexError:
        # VIP_new[n_Outhi]=VIP_new[n_Outmi]-1 #-1 könnte zu Problemen führen
        VIP_new[n_Outhi]=VIP_new[n_upBo] #-1 könnte zu Problemen führen
        txt+="%s set on maximum of diff.-quot. "%n_Outhi
    VIP_new=VIP_new.sort_values()
    return VIP_new, DQdfs, txt


def YM_sigeps_lin(stress_ser, strain_ser, 
                  method='leastsq', nan_policy='omit',
                  ind_S=None, ind_E=None):
    """
    Calculates Youngs Modulus of a stress-strain-curve with linear approach using non linear least squares curve fitting.

    Parameters
    ----------
    strain_col : pd.Series
        Strain curve.
    stress_col : pd.Series
        Stress_curve.
    ind_S : integer
        First used index in df.
    ind_E : integer
        Last used index in df.

    Returns
    -------
    E : float
        Youngs Modulus (corresponds to slope of linear fit).
    Eabs : float
        Stress value on strain origin (corresponds to interception of linear fit).
    Rquad : float
        Coefficient of determination.
    fit : lmfit.model.ModelResult
        Fitting result from lmfit (use with fit.fit_report() for report).

    """
    if not (ind_S is None):
        stress_ser = stress_ser.loc[ind_S:]
        strain_ser = strain_ser.loc[ind_S:]
    if not (ind_E is None):
        stress_ser = stress_ser.loc[:ind_E]
        strain_ser = strain_ser.loc[:ind_E]
        
    try:
        m       = lmfit.models.LinearModel()
        # guess doesn't work with NaN's
        # pg      = m.guess(stress_ser, x=strain_ser)
        # fit     = m.fit(stress_ser, pg, x=strain_ser,
        #                 method=method, nan_policy=nan_policy)
        fit     = m.fit(stress_ser, x=strain_ser,
                        method=method, nan_policy=nan_policy)
        E       = fit.params.valuesdict()['slope']
        Eabs    = fit.params.valuesdict()['intercept']
        # Rquad    = 1 - fit.residual.var() / np.var(stress_ser)
        Rquad    = 1 - fit.residual.var() / np.nanvar(stress_ser)
    except np.linalg.LinAlgError:
        warnings.warn("SVD did not converge in Linear Least Squares, all values set to NaN.")
        E, Eabs, Rquad, fit = np.nan,np.nan,np.nan,'SVD-Error'
    return E, Eabs, Rquad, fit

def fit_report_adder(fit,Var,Varname='R-square', show_correl=False):
    if isinstance(fit, str):
        fit = fit
    elif isinstance(fit, lmfit.model.ModelResult):
        fit=fit.fit_report(show_correl=False)
    else:
        raise TypeError('Datatype %s is not implemented!'%(type(fit)))
    orgstr="\n[[Variables]]"
    repstr="\n    {0:18s} = {1:1.5f}\n[[Variables]]".format(*[Varname,Var])
    txt=fit.replace(orgstr,repstr)
    return txt
  
def YM_eva_com_sel(stress_ser,
                   strain_ser,
                   comp=True, name='A', 
                   det_opt='incremental',**kws):
    """
    Calculates Young's Modulus over defined range with definable method.

    Parameters
    ----------
    stress_ser : pd.Series
        Series with stress values corresponding strain_ser.
    strain_ser : pd.Series
        Series with strain values corresponding stress_ser.
    comp : boolean, optional
        Compression mode. The default is True.
    name : string, optional
        Name of operation. The default is 'A'.
    det_opt : TYPE, optional
        Definable method for determination.
        Ether incremental or leastsq. The default is 'incremental'.
    **kws : dict
        Keyword dict for least-square determination.

    Returns
    -------
    det_opt == "incremental":
        YM_ser : pd.Series
            Series of Young's Moduli.
    or
    det_opt == "leastsq":
        YM : float
            Youngs Modulus (corresponds to slope of linear fit).
        YM_abs : float
            Stress value on strain origin (corresponds to interception of linear fit).
        YM_Rquad : float
            Coefficient of determination.
        YM_fit : lmfit.model.ModelResult
            Fitting result from lmfit (use with fit.fit_report() for report).
    """
    step_range=pd_combine_index(stress_ser, strain_ser)
    stress_ser = stress_ser.loc[step_range]
    strain_ser = strain_ser.loc[step_range]
    if det_opt=="incremental":
        YM_ser = stress_ser / strain_ser
        YM_ser = pd.Series(YM_ser, name=name)
        return YM_ser
    elif det_opt=="leastsq":
        YM, YM_abs, YM_Rquad, YM_fit = YM_sigeps_lin(stress_ser,
                                                     strain_ser, **kws)
        return YM, YM_abs, YM_Rquad, YM_fit

def Rquad(y_true, y_predicted, nan_policy='omit'):
    if nan_policy == 'omit':
        sse = np.nansum((y_true - y_predicted)**2)
        y_t_l_woNan = len(y_true) - np.count_nonzero(np.isnan(y_true))
        tse = (y_t_l_woNan - 1) * np.nanvar(y_true, ddof=1)
        r2_score = 1 - (sse / tse)
        return r2_score
    elif nan_policy == 'raise':
        if np.isnan(y_true).any() or np.isnan(y_predicted):
            raise ValueError("NaN values during Rquad detected")
    else:
        if nan_policy != 'propagate':
            raise NotImplementedError("NaN policy %s not implemented"%nan_policy)
    sse = np.sum((y_true - y_predicted)**2)
    tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    r2_score = 1 - (sse / tse)
    return r2_score


def stress_linfit(strain_ser, YM, YM_abs, strain_offset=0.002):
    """Linearised stress fit corresponding Youngs Modulus and strain"""
    stress_fit_ser=(strain_ser - strain_offset) * YM + YM_abs
    return stress_fit_ser

def stress_linfit_plt(strain_ser, inds, YM, YM_abs, strain_offset=0, ext=0.1):
    """Linearised stress fit for plotting corresponding Youngs Modulus and strain"""
    strain_vals = strain_ser.loc[inds].values
    strain_vals = np.array([max(0,strain_vals[0]-ext*(strain_vals[-1]-strain_vals[0])),
                          min(strain_ser.max(),strain_vals[-1]+ext*(strain_vals[-1]-strain_vals[0]))])
    stress_vals = stress_linfit(strain_vals, YM, YM_abs, strain_offset)
    return strain_vals, stress_vals

def Yield_redet(m_df, VIP, n_strain, n_stress,
                n_loBo, n_upBo, n_loBo_int, 
                YM, YM_abs, strain_offset=0.002, 
                rise_det=[True,4], n_yield='Y'):
    """
    Redetermine yield point to different conditions 
    (intersection with linearised strain offset, zero rising, fixed endpoint).

    Parameters
    ----------
    m_df : pd.DataFrame
        Measured data.
    VIP : pd.Series
        Important points corresponding to measured data.
    n_strain : string
        Name of used strain (have to be in measured data).
    n_stress : string
        Name of used stress (have to be in measured data).
    n_loBo : [string]
        List of lower borders for determination (have to be in VIP).
    n_upBo : [string]
        List of upper borders for determination (have to be in VIP).
    n_loBo_int : [string]
        List of lower borders for interseption (have to be in VIP).
    YM : float
        Youngs Modulus.
    YM_abs : float
        Absolut value of Youngs Modulus.
    strain_offset : float, optional
        Strain offset (eq. plastic strain). The default is -0.002.
    rise_det : [bool, int], optional
        Determination options for stress rising ([smoothing, smoothing factor]).
        The default is [True,4].
    n_yield : string, optional
        Name of yield point (have to be in VIP). The default is 'Y'.

    Returns
    -------
    VIP_new : pd.Series
        Important points corresponding to measured data.
    txt : string
        Documantation string.

    """
    #     mit 0.2% Dehnversatz E(F+-F-) finden
    m_lim = m_df.loc[min(VIP[n_loBo]):max(VIP[n_upBo])]
    stress_fit = stress_linfit(m_lim[n_strain],
                               YM, YM_abs, strain_offset)
    i = m_lim[n_stress]-stress_fit
    i_sign, i_sich = sign_n_change(i)
    i_sich = i_sich.loc[min(VIP[n_loBo_int]):] # Y erst ab F- suchen
    
    _,_,r_sich = rise_curve(m_df[n_stress],*rise_det)
    r_sich = r_sich.loc[min(VIP[n_loBo])+2:max(VIP[n_upBo])+2]
    
    VIP_new = VIP
    txt =""
    if (i_sich.any()==True)and(r_sich.any()==True):
        if(i_sich.loc[i_sich].index[0])<=(r_sich.loc[r_sich==True].index[0]):
            VIP_new[n_yield] = i_sich.loc[i_sich].index[0]
            txt+="Fy set on intersection with %.2f %% pl. strain!"%(strain_offset*100)
        else:
            VIP_new[n_yield]=r_sich.loc[r_sich==True].index[0]-2 #-2 wegen Anstiegsberechnung
            txt+="Fy on first point between F+ and Fu with rise of 0, instead intersection %f %% pl. strain! (earlier)"%(strain_offset*100)
    elif (i_sich.any()==True)and(r_sich.all()==False):
        VIP_new[n_yield]=i_sich.loc[i_sich].index[0]
        txt+="Fy set on intersection with %.2f %% pl. strain!"%(strain_offset*100)
    else:
        if r_sich.any():
            VIP_new[n_yield]=r_sich.loc[r_sich==True].index[0]-2
            txt+="Fy on first point between F+ and Fu with rise of 0, instead intersection %.2f %% pl. strain! (No intersection found!)"%(strain_offset*100)
        else:
            # txt+="\nFy kept on old value, instead intersection 0.2% pl. strain! (No intersection found!)"
            tmp = VIP[VIP==max(VIP[n_upBo])].index[0]
            VIP_new[n_yield] = VIP[tmp]
            txt+="Fy set to %s (max of %s), instead intersection %.2f %% pl. strain or rise of 0! (No intersection found!)"%(tmp,n_upBo,(strain_offset*100))
    VIP_new = VIP_new.sort_values()
    return VIP_new, txt

#%%% Misc
def pd_trapz(pdo, y=None, x=None, axis=0, nan_policy='omit'):
    if pd_isDF(pdo):
        pdo=pdo.loc(axis=pd_axischange(axis))[[x,y]]
        if nan_policy=='omit':
            pdo=pd_exclnan(pdo=pdo, axis=pd_axischange(axis))
        elif nan_policy=='raise' and pdo.isna().any(None):
            raise ValueError("NaN in objectiv!")
        elif nan_policy=='interpolate':
            pdo=pdo.interpolate(axis=axis)
        out=np.trapz(y=pdo.loc(axis=pd_axischange(axis))[y],
                     x=pdo.loc(axis=pd_axischange(axis))[x])
    elif pd_isSer(pdo):
        if nan_policy=='omit':
            pdo=pd_exclnan(pdo=pdo)
        elif nan_policy=='raise' and pdo.isna().any(None):
            raise ValueError("NaN in objectiv!")
        elif nan_policy=='interpolate':
            pdo=pdo.interpolate()
        out=np.trapz(y=pdo)
    else:
        NotImplementedError("Type %s not implemented!"%type(pdo))
    return out

#%%%% Circle and radius/curvature from 3 points
def TP_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1.0e-6:
        # return (None, np.inf)
        return ((np.nan,np.nan), np.inf)
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def TP_radius(pts, outopt='signed_curvature'):
    """
    Determines the radius or curvature of a circle through three points.
    The sign indicates the direction (positve=left turn/curvature downwards).

    Parameters
    ----------
    pts : np.array or pd.DataFrame of shape (2=[x,y],3=[left,mid,right])
        Array of point coordinates.
    outopt : string, optional
        Option for output. 
        Possible are: 
            - ['signed_radius','sr']: signed radius
            - ['signed_curvature','sc']: signed curvature
            - ...: all values (Center of circle, radius, sign, quadrant to mid)
        The default is 'signed_curvature'.

    Returns
    -------
    float or tuple
        Radius, curvature or all outputs.

    """
    c,r=TP_circle(pts[0],pts[1],pts[2])
    if r==np.inf:
        m=[0,0]
    else:
        m=c-pts[[0,2]].mean(axis=1).values
    if np.sign(m[1])==1: # quadrant to determine sign of radius
        q=1 if (np.sign(m[0]) in [0,1]) else 4
    elif np.sign(m[1])==-1:
        q=2 if (np.sign(m[0]) in [0,1]) else 3
    else:
        q=1 if (np.sign(m[0]) in [0,1]) else 2
    s=1 if q in [1,4] else -1 # sign is positive if center is above mid
    if outopt in ['signed_radius','sr']:
        return r*s
    elif outopt in ['signed_curvature','sc']:
        return 1/r*s
    else:
        return (c,r,s,q)


def Geo_curve_TBC(func, params, length, outopt='signed_curvature'):
    """
    Determines the radius or curvature of a circle through three points.
    The sign indicates the direction (positve=left turn/curvature downwards).    

    Parameters
    ----------
    func : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    length : TYPE
        DESCRIPTION.
    outopt : string, optional
        Option for output. 
        Possible are: 
            - ['signed_radius','sr']: signed radius
            - ['signed_curvature','sc']: signed curvature
            - ...: all values (Center of circle, radius, sign, quadrant to mid)
        The default is 'signed_curvature'.

    Returns
    -------
    r : float or tuple
        Radius, curvature or all outputs.

    """
    ptx=np.array([-length/2,0,length/2])
    pty=func(ptx,params)
    pts=pd.DataFrame([ptx,pty],index=['x','y'])
    r=TP_radius(pts, outopt=outopt)
    return r


#%% Statistics

def stat_outliers(data, option='IQR', span=1.5,
                  out='all', outsort = 'ascending'):
    # if isinstance(data, pd.core.base.ABCSeries):
    #     d = data
    # elif isinstance(data, np.ndarray):
    #     d = pd.Series(data)
    # else:
    #     raise NotImplementedError("Datatype %s not implemented!"%type(data))
        
    if option == 'IQR':
        dQ1 = data.quantile(q=0.25)
        dQ3 = data.quantile(q=0.75)
        IQR = dQ3 - dQ1
        lBo = dQ1 - span * IQR
        uBo = dQ3 + span * IQR
    else:
        raise NotImplementedError("Option %s not implemented!"%option)
    
    if outsort == 'ascending':
        data = data.sort_values(ascending=True)
    elif outsort == 'descending':
        data = data.sort_values(ascending=False)
    else:
        data = data
        
    if out == 'all':
        stol = data[(data < lBo) | (data > uBo)].index.to_list()
    elif out == 'lower':
        stol = data[(data < lBo)].index.to_list()
    elif out == 'higher':
        stol = data[(data > uBo)].index.to_list()
    elif out == 'inner':
        stol = data[(data >= lBo) & (data <= uBo)].index.to_list()
    else:
        raise NotImplementedError("Output %s not implemented!"%out)
        
    return stol

def stat_box_vals(data, option='IQR', span=1.5):
    if option == 'IQR':
        dQ1 = data.quantile(q=0.25)
        med = data.quantile(q=0.50)
        dQ3 = data.quantile(q=0.75)
        IQR = dQ3 - dQ1
        lBo = dQ1 - span * IQR
        uBo = dQ3 + span * IQR
        # box_vals = {'lBo':lBo,'dQ1':dQ1,'med':med,'dQ3':dQ3,'uBo':uBo}
        inner_vals = stat_outliers(data=data, option=option, span=span,
                                   out='inner', outsort=None)
        outer_vals = stat_outliers(data=data, option=option, span=span,
                                   # out='outer', outsort=None)
                                   out='all', outsort=None)
        minin = data.loc[inner_vals].min()
        maxin = data.loc[inner_vals].max()
        box_vals = {'lBo':lBo,'minin':minin,'dQ1':dQ1,'med':med,
                    'dQ3':dQ3,'maxin':maxin,'uBo':uBo}
    else:
        raise NotImplementedError("Option %s not implemented!"%option)
    return box_vals, inner_vals, outer_vals

def NaN_stat_outliers(df, numeric_only=True, 
                      option='IQR', span=1.5,
                      out='all', outsort = None):
    if numeric_only:
        cols = df.select_dtypes(include=['int','float']).columns
    else:
        cols = df.columns
    kws = {'option':option, 'span':span,
           'out':out, 'outsort' : outsort}
    t=df[cols].agg(stat_outliers,**kws)
    dfout = df.copy(deep=True)
    for i in cols:
        dfout.loc[t[i],i]=np.nan
    return dfout

def meanwoso(data, option='IQR', span=1.5,
             out='inner', outsort = None):
    # if isinstance(data, pd.core.base.ABCSeries):
    #     data = data
    # elif isinstance(data, np.ndarray) or isinstance(data, 'float'):
    #     data = pd.Series(data)
    # else:
    #     raise NotImplementedError("Datatype %s not implemented!"%type(data))
    used = stat_outliers(data=data, option=option, span=span, out=out, outsort=outsort)
    dout = data.loc[used].mean()
    return dout
# Overwrite pandas
pd.Series.meanwoso = meanwoso
pd.DataFrame.meanwoso = meanwoso

def stdwoso(data, option='IQR', span=1.5,
            out='inner', outsort = None):
    used = stat_outliers(data=data, option=option, span=span, out=out, outsort=outsort)
    dout = data.loc[used].std()
    return dout
# Overwrite pandas
pd.Series.stdwoso = stdwoso
pd.DataFrame.stdwoso = stdwoso

def coefficient_of_variation(data, outsort = None):
    dstd  = data.std()
    dmean = data.mean()
    if (isinstance(dmean, float) and dmean == 0):
        dout  = np.nan
    else:
        dout  = dstd/dmean
    # dout  = dstd/dmean
    if not outsort is None:
        dout = pd_outsort(data = dout, outsort = outsort)
    return dout

def coefficient_of_variation_woso(data, option='IQR', span=1.5,
                                  out='inner', outsort = None):
    dstd  =  stdwoso(data=data, option=option, span=span, out=out, outsort=None)
    dmean = meanwoso(data=data, option=option, span=span, out=out, outsort=None)
    if (isinstance(dmean, float) and dmean == 0):
        dout  = np.nan
    else:
        dout  = dstd/dmean
    # dout  = dstd/dmean
    if not outsort is None:
        dout = pd_outsort(data = dout, outsort = outsort)
    return dout

def confidence_interval(data, confidence=0.95, method="Seaborn_Bootstrap",
                        func = np.nanmean, n_boot = 1000, axis = None,
                        units = None, seed = 0, outtype="List"):
    """
    Calculates the confidence interval of given data.

    Parameters
    ----------
    data : pandas.Series of float
        Series of values to be analysed.
    confidence : float, optional
        Confidence value. The default is 0.95.
    method : string, optional
        Method to use.
        Implemented are:
            - Seaborn_Bootstrap: Seaborn Bootstrapping (Compare Seaborn barplot)
            - Wald: Wald-Confidence-Interval
        The default is "Seaborn_Bootstrap".
    func : string or callable, optional
        DESCRIPTION. The default is "nanmean".
    n_boot : int, optional
        Number of iterations. The default is 1000.
    axis : int, optional
        Applied axis. The default is None.
    units : array, optional
        Sampling units (see seaborn.algorithms). The default is None.
    seed : Generator | SeedSequence | RandomState | int | None, optional
        Seed for rondom number generator. The default is 0.
    outtype : str, optional
        Switch for output (List, Dict, Min, Max). The default is List.

    Raises
    ------
    NotImplementedError
        Error to raise if option not implemented.

    Returns
    -------
    CI : list or or dict or value
        Confidence interval values [lower, higher].

    """
    def CIoutsel(inp, outtype):
        if outtype=="List":
            out = inp
        elif outtype=="Dict":
            out = {'CImin':inp[0],'CImax':inp[1]}
        elif outtype=="Min":
            out = inp[0]
        elif outtype=="Max":
            out = inp[1]        
        return out    
    # if numeric_only:
    #     if not pd.api.types.is_numeric_dtype(data):
    #         try:
    #             data=data.apply(pd.to_numeric, errors='raise')
    #         except:
    #             return CIoutsel([np.nan, np.nan], outtype)
    # if skipna:
    #     data=pd_exclnan(data,axis=axis)
    if method == "Wald":
        if confidence == 0.95 and func in [np.nanmean,"mean"]:
            m = data.mean().values[0]
            f = 1.96 # only implemented for 95%-confidence 
            v = np.sqrt(data.std()/data.count()).values[0]
            CI = np.array([m - f*v , m + f*v])
        else:
            raise NotImplementedError("Wald-Confidence-Interval only implemented for 95 % confidence!")
    # elif method == "Scikits_Bootstrap":
    #     import scipy
    #     import scikits.bootstrap as bootstrap
    #     CI = bootstrap.ci(data=data, statfunction=scipy.mean, alpha = 1-confidence, n_samples=n_boot)
    elif method == "Seaborn_Bootstrap":
        import seaborn as sb
        CI = sb.utils.ci(a=sb.algorithms.bootstrap(data, func= func, n_boot=n_boot,
                                                   axis=axis, units=units, seed=seed),
                         which = confidence*100, axis=axis)
    else:
        CI = [np.nan, np.nan]
        raise NotImplementedError("Method %s not implemented!"%method)
    return CIoutsel(CI, outtype)
def CImin(data, confidence=0.95, method="Seaborn_Bootstrap",
          func = np.nanmean, n_boot = 1000, axis = None,
          units = None, seed = 0):
    """Return only minimum of confidence interval. (Use same seed and n_boot!)"""
    out = confidence_interval(data=data,confidence=confidence,method=method,
                              func=func, n_boot=n_boot, axis=axis,
                              units=units, seed=seed,
                              outtype="Min")
    return out
    
def CImax(data, confidence=0.95, method="Seaborn_Bootstrap",
          func = np.nanmean, n_boot = 1000, axis = None,
          units = None, seed = 0):
    """Return only maximum of confidence interval. (Use same seed and n_boot!)"""
    out = confidence_interval(data=data,confidence=confidence,method=method,
                              func=func, n_boot=n_boot, axis=axis,
                              units=units, seed=seed,
                              outtype="Max")
    return out

def pd_agg(pd_o, agg_funcs=['mean','median','std','max','min'], numeric_only=False):
    """Aggregate pandas object with defined functions."""
    if (type(pd_o) is pd.core.series.Series):
        out = pd_o.agg(agg_funcs)
    elif (type(pd_o) is pd.core.frame.DataFrame):
        if numeric_only:
            cols = pd_o.select_dtypes(include=['int','float']).columns
        else:
            cols = pd_o.columns
        out = pd_o[cols].agg(agg_funcs)
    elif (type(pd_o) is pd.core.groupby.generic.DataFrameGroupBy):
        if numeric_only:
            cols = pd_o.first().select_dtypes(include=['int','float']).columns
        else:
            # cols = pd_o.columns
            cols = pd_o.first().columns
        out = pd_o.agg(agg_funcs)[cols]
    else:
        raise NotImplementedError("Type %s not implemented!"%type(pd_o))
    return out

def agg_add_ci(pdo, agg_funcs=['mean','std','min','max']):
    """Adds confidence interval to pandas aggregate function"""
    a=pdo.agg(agg_funcs)
    a1=pdo.agg(confidence_interval)
    a1.index=['ci_min','ci_max']
    a=pd.concat([a,a1])
    return a

def pd_agg_custom(pdo, agg_funcs=['mean',meanwoso,'median',
                                  'std',coefficient_of_variation, 
                                  stdwoso, coefficient_of_variation_woso,
                                  'min','max',confidence_interval],
                  numeric_only=False, 
                  af_ren={'coefficient_of_variation_woso':'CVwoso',
                          'coefficient_of_variation':'CV'},
                  af_unp={'confidence_interval': ['CImin','CImax']}):
    """Aggregate pandas object with defined functions, including unpacked multi-value functions. 
    

    Parameters
    ----------
    pdo : pd.DataFrame or pd.Series
        Pandas object (DataFrame or Series).
    agg_funcs : list of functions, optional
        List of aggregatable functions (pandas aggregate accepted). 
        The default is ['mean',meanwoso,'median',
                        'std',coefficient_of_variation,stdwoso,
                        coefficient_of_variation_woso,
                        'min','max',confidence_interval].
    numeric_only : bool, optional
        Switch for consideration of numerical values only (int and float). The default is False.
    af_ren : dict, optional
        Dictionary for renaming of aggregate function names.
        The default is {'coefficient_of_variation_woso':'CVwoso',
                        'coefficient_of_variation':'CV'}.
    af_unp : dict, optional
        Dictionary for unpacking of aggregate function. 
        The default is {'confidence_interval': ['CImin','CImax']}.

    Returns
    -------
    pd.DataFrame or pd.Series
        Aggregation values depending on given functions.

    """
    def unpack_aggval(o, n, rn):
        tmp=o.loc[n].apply(lambda x: pd.Series([*x], index=rn))
        i = o.index.get_indexer_for([n])[0]
        s=pd.concat([o.iloc[:i],tmp.T,o.iloc[i+1:]],axis=0)
        return s
    pda = pd_agg(pd_o=pdo, agg_funcs=agg_funcs, numeric_only=numeric_only)
    for i in af_unp.keys():
        pda = unpack_aggval(pda, i, af_unp[i])
    if len(af_ren.keys())>0:
        pda = pda.rename(af_ren)
    return pda

def Dist_test(pds, alpha=0.05, mcomp='Shapiro', mkws={},
              skipna=True, add_out = False):
    """
    Distribution test of data to Hypothesis sample looks Gaussian.

    Parameters
    ----------
    pds : pd.Series
        Series of data.
    alpha : float, optional
        Test criterion to reject zero hypothesis (normal distribution).
        The default is 0.05.
    mcomp : str, optional
        Test method. The default is 'Shapiro'.
    mkws : dict, optional
        Keyword arguments for used test. The default is {}.
    skipna : bool, optional
        Switch for skipping NaNs in data. The default is True.
    add_out : bool or str, optional
        Switch for output. The default is False.

    Raises
    ------
    NotImplementedError
        Method not implemented.

    Returns
    -------
    str, Dict, Series, [Series, Series, str]
        Output of test results, depending on add_out.

    """
    if mcomp in ['Shapiro','S','shapiro']:
        stats_test  = wraps(partial(stats.shapiro, **mkws))(stats.shapiro)
    elif mcomp in ['Normaltest','normaltest','K2','Ksquare',
                   'DAgostino','dagostino','D’Agostino’s K^2']:
        stats_test  = wraps(partial(stats.normaltest, **mkws))(stats.normaltest)
    else:
        raise NotImplementedError('Method %s for distribution test not implemented!'%mcomp)
    if skipna:
        data = pds.dropna()
    else:
        data = pds
    ano_n = data.count()
    t = stats_test(data)
    F = t.statistic
    p = t.pvalue
    if p < alpha:
        rtxt = 'H0 rejected!'
        H0=False
    else:
        rtxt = 'Fail to reject H0!'
        H0=True
    txt=("- F(%d) = %.3e, p = %.3e (%s)"%(ano_n,F,p,rtxt)) # Gruppen sind normalverteielt p<0.05
    odict={'N':ano_n, 'Stat':F, 'p':p, 'H0':H0}
    if add_out is True:
       return txt, t
    elif add_out=='Dict':
       return odict
    elif add_out=='Series':
       return pd.Series(odict)
    elif add_out=='Test':
       return data, pd.Series(odict), txt
    else:
       return txt
   
def Dist_test_multi(pdo, axis=0,
                    alpha=0.05, mcomps=['Shapiro','DAgostino'], mkws={},
                    skipna=True, add_out = 'DF'):
    """Performs multiple distribution tests."""
    df_out = pd.DataFrame([],dtype='O')
    for mcomp in mcomps:
        tmp=pdo.apply(Dist_test, alpha=alpha, mcomp=mcomp, mkws=mkws, 
                      skipna=skipna, add_out='Dict')
        tmp=tmp.apply(pd.Series)
        tmp.columns = pd.MultiIndex.from_product([[mcomp],tmp.columns])
        df_out=pd.concat([df_out,tmp],axis=1)
    return df_out


def group_Anova(df, groupby, ano_Var, group_str=None, ano_str=None, alpha=0.05):
    ano_data=pd.Series([],dtype='O')
    j=0
    for i in df[groupby].drop_duplicates().values:
        ano_data[j]=df.loc[df[groupby]==i] #Achtung: keine statistics-Abfrage!
        j+=1
    ano_df1=df[groupby].drop_duplicates().count()-1 #Freiheitsgrad 1 = Gruppen - 1
    # ano_df2=df.count()[0]-(ano_df1+1) #Freiheitsgrad 2 = Testpersonen - Gruppen
    ano_df2=df[ano_Var].count()-(ano_df1+1) #Freiheitsgrad 2 = Testpersonen - Gruppen #23-02-20: auf Var bezogen
    
    if ano_str is None:
        ano_str = ano_Var
    if group_str is None:
        group_str = groupby
    [F,p]=stats.f_oneway(*[ano_data[i][ano_Var] for i in ano_data.index])
    if p < alpha:
        rtxt = 'H0 rejected!'
    else:
        rtxt = 'Fail to reject H0!'
    txt=("F(%4d,%4d) = %7.3f, p = %.3e, for %s to %s (%s)"%(ano_df1,ano_df2,
                                                            F,p,
                                                            ano_str,group_str,
                                                            rtxt)) # Gruppen sind signifikant verschieden bei p<0.05
    return txt

def group_ANOVA_MComp(df, groupby, ano_Var, 
                      group_str=None, ano_str=None,
                      mpop = "ANOVA", alpha=0.05, group_ren={},
                      do_mcomp_a=1, mcomp='TukeyHSD', mpadj='bonf', Ffwalpha=2,
                      mkws={}, nan_policy='omit',
                      add_T_ind=3, add_out = False):
    """
    Performs an one way ANOVA and multi comparision test for given variable, in respect to given groups.
    Returns an output string with summary and optional additional test outputs.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with groups and values as columns for statistical test.
    groupby : str
        Name of groups for statistical test.
    ano_Var : str
        Name of variable for statistical test.
    group_str : str, optional
        String for output implementation of group names. The default is None.
    ano_str : str, optional
        String for output implementation of value names. The default is None.
    mpop : str, optional
        Method for population test.
        Implemented are ANOVA and Kruskal-Wallis H-test.
        The default is ANOVA.
    alpha : float, optional
        Test criterion to reject zero hypothesis (all means are equal). The default is 0.05.
    group_ren : dict, optional
        Renaming of groups in df. The default is {}.
    do_mcomp_a : int, optional
        Performance level of tukey test 
        (0 - never, 1 - only if p of ANOVA lower alpha, 2 - allways).
        The default is 1.
    mcomp : str, optional
        Method for multi comparison.
        Implemented are TukeyHSD, ttest_ind, ttest_rel, mannwhitneyu.
        The default is TukeyHSD.
    mpadj : str, optional
        Method for testing and adjustment of pvalues.
        For further information see: statsmodels.stats.multitest.multipletests
        The default is bonf.
    Ffwalpha : float, optional
        Factor for family-wise error in comparision to alpha.
        The default is 2.
    mkws : dict, optional
        Keyword arguments for used multi comparision test.
        p.e.: {'equal_var': False, 'nan_policy': 'omit'}
        The default is {}. 
    nan_policy : str, optional
        Handling of NaN values. The default is 'omit'.
    add_T_ind : int, optional
        Additional indentation of lines in txt for Tukey HSD. The default is 3.
    add_out : bool, optional
        Additional output request. The default is False.
    
    Returns
    -------
    txt : str
        Text output with ANOVA and optional Tukey-HSD results.
    [F,p] : [float,float], optional
        ANOVA results
    t : statsmodels.iolib.table.SimpleTable, optional
        Results of multi comaprison test.
    """
    from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
    atxt=''
    df=df.copy(deep=True) #No Override
    if not groupby in df.columns:
        if groupby in df.index.names:
            df[groupby]=df.index.get_level_values(groupby)
        else:
            raise KeyError("Group ether in columns nor in index names!")
    if not len(group_ren.keys()) == 0:
        # df[groupby] = df[groupby].replace(group_ren) #SettingWithCopyWarning
        df[groupby].replace(group_ren, inplace=True)
    # direct implemented, else: RecursionError: maximum recursion depth exceeded while calling a Python object
    # Atxt = group_ANOVA_Tukey(df, groupby, ano_Var, group_str, ano_str, alpha)
    ano_data=pd.Series([],dtype='O')
    j=0
    # for i in df[groupby].drop_duplicates().values:
    #     ano_data[j]=df.loc[df[groupby]==i] #Achtung: keine statistics-Abfrage!
    #     j+=1
    # ano_df1=df[groupby].drop_duplicates().count()-1 #Freiheitsgrad 1 = Gruppen - 1
    for i in df[groupby].drop_duplicates().values:
        tmp=df.loc[df[groupby]==i][ano_Var]
        if nan_policy == 'raise':
            if tmp.isna().any():
                raise ValueError('NaN in group %s detected!'%i)
        elif nan_policy == 'omit':
            tmp=tmp.dropna()
        if tmp.count()==0:
            atxt += '(-%s)'%i
        else:
            ano_data[i]=tmp
            j+=1
    ano_df1=j-1 #Freiheitsgrad 1 = Gruppen - 1
    # ano_df2=df.count()[0]-(ano_df1+1) #Freiheitsgrad 2 = Testpersonen - Gruppen
    ano_df2=df[ano_Var].count()-(ano_df1+1) #Freiheitsgrad 2 = Testpersonen - Gruppen #23-02-20: auf Var bezogen
    if ano_str is None:
        ano_str = ano_Var
    if group_str is None:
        group_str = groupby
    if mpop in ["ANOVA", "f_oneway", "F-Test"]:
        [F,p]=stats.f_oneway(*[ano_data[i] for i in ano_data.index])
    elif mpop in ["Kruskal-Wallis H-test", "kruskal", "H-test", "Kruskal-Wallis"]:
        [F,p]=stats.kruskal(*[ano_data[i] for i in ano_data.index])
    elif mpop in ["Friedmann", "FriedmannChi²", "friedmann", "friedmanchisquare"]:
        [F,p]=stats.friedmanchisquare(*[ano_data[i] for i in ano_data.index])
    else:
        raise NotImplementedError('Method %s for population test not implemented!'%mpop)
    if p < alpha:
        rtxt = 'H0 rejected!'
        if do_mcomp_a > 0: do_mcomp_a+=1
        H0=False
    else:
        rtxt = 'Fail to reject H0!'
        H0=True
    Atxt=("- F(%3d,%4d) = %7.3f, p = %.3e, for %s to %s (%s)"%(ano_df1,ano_df2,
                                                               F,p,
                                                               ano_str,group_str,
                                                               rtxt+atxt)) # Gruppen sind signifikant verschieden bei p<0.05    
    if do_mcomp_a >= 2:
        # t = pairwise_tukeyhsd(endog=df[ano_Var], groups=df[groupby], alpha=alpha)
        mcp = MultiComparison(data=df[ano_Var], groups=df[groupby])
        if mcomp=='TukeyHSD':
            t = mcp.tukeyhsd(alpha=alpha*Ffwalpha)
            Ttxt = str_indent(t.summary(),add_T_ind)
        # elif mcomp=='ttest_ind':
        #     t = mcp.allpairtest(stats.ttest_ind, alpha=alpha, method=mpadj)
        # elif mcomp=='ttest_rel':
        #     t = mcp.allpairtest(stats.ttest_rel, alpha=alpha, method=mpadj)
        else:
            if mcomp=='ttest_ind':
                stats_test  = wraps(partial(stats.ttest_ind, **mkws))(stats.ttest_ind)
            elif mcomp=='ttest_rel':
                stats_test  = wraps(partial(stats.ttest_rel, **mkws))(stats.ttest_rel)
            elif mcomp=='mannwhitneyu':
                stats_test  = wraps(partial(stats.mannwhitneyu, **mkws))(stats.mannwhitneyu)
            elif mcomp=='wilcoxon':
                stats_test  = wraps(partial(stats.wilcoxon, **mkws))(stats.wilcoxon)
            else:
                raise NotImplementedError('Method %s for multi comparison not implemented!'%mcomp)
            t = mcp.allpairtest(stats_test, alpha=alpha*Ffwalpha, method=mpadj)[0]
            Ttxt = str_indent(t,add_T_ind)
        txt = Atxt + Ttxt
    else:
        t='No multi comparision done, see do_mcomp_a.'
        txt = Atxt
    if add_out==True:
        return txt, [F,p], t
    elif add_out=='Series':
        return pd.Series({"DF1": ano_df1, "DF2":ano_df2,
                          "Stat":F, "p": p, "H0": H0,
                          "txt": txt, "MCP": t})
    elif add_out=='Test':
        return pd.Series({"DF1": ano_df1, "DF2":ano_df2,
                          "Stat":F, "p": p, "H0": H0,
                          "txt": txt, "MCP": t}), ano_data
    else:
        return txt
    
def MComp_interpreter(T_Result):
    import string
    import statsmodels
    if isinstance(T_Result, statsmodels.iolib.table.SimpleTable):
        t=pd.DataFrame(T_Result.data[1:], columns=T_Result.data[0])
    else:
        t=pd.DataFrame(T_Result.summary().data[1:], columns=T_Result.summary().data[0])
    df_True = t.loc[t.reject==True,:]
    # letters = list(string.ascii_lowercase)
    letters = list(string.ascii_lowercase)+list(string.ascii_uppercase) # List out of index
    n = 0
    txt=''
    
    group1_list = df_True.group1.tolist() #get the groups from the df with only True (True df) to a list
    group2_list = df_True.group2.tolist()
    group3 = group1_list+group2_list #concat both lists
    group4 = list(set(group3)) #get unique items from the list
    group5 = [str(i) for i in group4 ] #convert unicode to a str
    group5.sort() #sort the list
    
    gen = ((i, 0) for i in group5) #create dict with 0 so the dict won't be empty when starts
    dictionary = dict(gen)
    
    group6 = [(group5[i],group5[j]) for i in range(len(group5)) for j in range(i+1, len(group5))] #get all combination pairs
    for pairs in group6: #check for each combination if it is present in df_True
    
        txt += '\n' + str(n)
        txt += '\n' + str(dictionary)
    
        try:
            a = df_True.loc[(df_True.group1==pairs[0])&(df_True.group2==pairs[1]),:] #check if the pair exists in the df
    
        except:
            a.shape[0] == 0
    
        if a.shape[0] == 0: #it mean that the df is empty as it does not appear in df_True so this pair is equal
    
            txt += '\n' + str('equal')
    
            if dictionary[pairs[0]] != 0 and dictionary[pairs[1]] == 0: #if the 1st is populated but the 2nd in not populated
                txt += '\n' + str("1st is populated and 2nd is empty")
                dictionary[pairs[1]] = dictionary[pairs[0]]
    
            elif dictionary[pairs[0]] != 0 and dictionary[pairs[1]] != 0: #if both are populated, check matching labeles
                txt += '\n' + str("both are populated")
                if len(list(set([c for c in dictionary[pairs[0]] if c in dictionary[pairs[1]]]))) >0: #check if they have a common label
                        txt += '\n' + str("they have a shared character")
                else:
                    txt += '\n' + str("equal but have different labels")
                 #check if the 1st group label doesn't appear in anyother labels, if it is unique then the 2nd group can have the first group label
                    m = 0 #count the number of groups that have a shared char with 1st group
                    j = 0 #count the number of groups that have a shared char with 2nd group
                    for key, value in dictionary.items():
                        if key != pairs[0] and len(list(set([c for c in dictionary[pairs[0]] if c in value])))==0:
                            m+=1
                    for key, value in dictionary.items():
                        if key != pairs[1] and len(list(set([c for c in dictionary[pairs[1]] if c in value])))==0:
                            j+=1
                    if m == len(dictionary)-1 and j == len(dictionary)-1: #it means that this value is unique because it has no shared char with another group
                        txt += '\n' + str("unique")
                        dictionary[pairs[1]] = dictionary[pairs[0]][0]
    
                    else:
                        txt += '\n' + str("there is at least one group in the dict that shares a char with the 1st group")
                        dictionary[pairs[1]] = dictionary[pairs[1]] + dictionary[pairs[0]][0]
    
    
            else:  # if it equals 0, meaning if the 1st is empty (which means that the 2nd must be also empty)
                txt += '\n' + str("both are empty")
                dictionary[pairs[0]] = letters[n]
                dictionary[pairs[1]] = letters[n]
    
        else:
    
            txt += '\n' + str("not equal")
    
            if dictionary[pairs[0]] != 0: # if the first one is populated (has a value) then give a value only to the second 
    
                txt += '\n' + str('1st is populated')
                # if the 2nd is not empty and they don't share a charcter then no change is needed as they already have different labels
                if dictionary[pairs[1]] != 0 and len(list(set([c for c in dictionary[pairs[0]] if c in dictionary[pairs[1]]]))) == 0:
                    txt += '\n' + str("no change")
    
                elif dictionary[pairs[1]] == 0: #if the 2nd is not populated give it a new letter
                    dictionary[pairs[1]] = letters[n+1]
    
                #if the 2nd is populated and equal to the 1st, then change the letter of the 2nd to a new one and assign its original letter to all the others that had the same original letter       
                elif  dictionary[pairs[1]] != 0 and len(list(set([c for c in dictionary[pairs[0]] if c in dictionary[pairs[1]]]))) > 0:
                    #need to check that they don't share a charcter
                    txt += '\n' + str("need to add a letter")
                    original_value = dictionary[pairs[1]]
                    dictionary[pairs[1]] = letters[n]
                    for key, value in dictionary.items():
                        if key != pairs[0] and len(list(set([c for c in original_value if c in value])))>0: #for any given value, check if it had a character from the group that will get a new letter, if so, it means  that they are equal and thus the new letter should also appear in the value of the "old" group 
                            dictionary[key] = original_value + letters[n]  #add the original letter of the group to all the other groups it was similar to               
    
            else:
    
                txt += '\n' + str('1st is empty')
                dictionary[pairs[0]] = letters[n]
                dictionary[pairs[1]] = letters[n+1]
                txt += '\n' + str(dictionary)
            n+=1
    
    # get the letter out the dictionary
    
    labels = list(dictionary.values())
    labels1 = list(set(labels))
    labels1.sort()
    final_label = ''.join(labels1)
    
    df2=pd.concat([t.group1,t.group2])
    group_names=df2.unique()
    for GroupName in group_names:
        if GroupName in dictionary:
            txt += '\n' + str("already exists")
        else:
            dictionary[str(GroupName)] = final_label
    
    for key, value in dictionary.items(): #this keeps only the unique char per group and sort it by group
        dictionary[key] =  ''.join(set(value))
    
    dict2 = dict(sorted(dictionary.items())) # the final output
    return dict2, txt

def group_ANOVA_MComp_multi(df, group_main='Series', group_sub=['A'],
                            ano_Var=['WC_vol'],
                            mpop = "ANOVA", alpha=0.05, group_ren={},
                            do_mcomp_a=0, mcomp='TukeyHSD', mpadj='bonf', Ffwalpha=1,
                            mkws={},
                            Transpose=True):
    """
    Performs an one way ANOVA and multi comparision test for given variable,
    in respect to given groups and sub-groups.
    Returns an output string with summary and optional additional test outputs.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with groups and values as columns for statistical test.
    group_main : str
        Main group for statistical test.
    group_sub : list of str
        Sub-groups for statistical test (p.e. variants).
    ano_Var : list of str
        Names of variables for statistical test.
    group_str : str, optional
        String for output implementation of group names. The default is None.
    mpop : str, optional
        Method for population test.
        Implemented are ANOVA and Kruskal-Wallis H-test.
        The default is ANOVA.
    alpha : float, optional
        Test criterion to reject zero hypothesis (all means are equal). The default is 0.05.
    group_ren : dict, optional
        Renaming of groups in df. The default is {}.
    do_mcomp_a : int, optional
        Performance level of tukey test 
        (0 - never, 1 - only if p of ANOVA lower alpha, 2 - allways).
        The default is 1.
    mcomp : str, optional
        Method for multi comparison.
        Implemented are TukeyHSD, ttest_ind, ttest_rel, mannwhitneyu.
        The default is TukeyHSD.
    mpadj : str, optional
        Method for testing and adjustment of pvalues.
        For further information see: statsmodels.stats.multitest.multipletests
        The default is bonf.
    Ffwalpha : float, optional
        Factor for family-wise error in comparision to alpha.
        The default is 2.
    mkws : dict, optional
        Keyword arguments for used multi comparision test.
        p.e.: {'equal_var': False, 'nan_policy': 'omit'}
        The default is {}.
    Transpose : bool, optional
        Switch for transposing result. The default is False.
    
    Returns
    -------
    df_out : pd.DataFrame
        DataFrame output with variance analyses and additional results.
    """
    df_out=pd.DataFrame([],dtype='O')
    for av in ano_Var:
        if len(group_sub) == 0:
            tmp=group_ANOVA_MComp(df=df, groupby=group_main, 
                                       ano_Var=av,
                                       group_str=group_main,
                                       ano_str=av,
                                       mpop=mpop, alpha=alpha,
                                       do_mcomp_a=do_mcomp_a, 
                                       mcomp=mcomp, 
                                       mpadj=mpadj, Ffwalpha=Ffwalpha, mkws=mkws,
                                       add_out = 'Series')
            # name='_'.join([group_main,av,sg])
            name=av
            df_out[name]=tmp
        else:
            for sg in group_sub:
                tmp=group_ANOVA_MComp(df=df, groupby=(group_main,sg), 
                                           ano_Var=(av,sg),
                                           group_str=group_main,
                                           ano_str='_'.join([av,sg]),
                                           mpop=mpop, alpha=alpha,
                                           do_mcomp_a=do_mcomp_a, 
                                           mcomp=mcomp, 
                                           mpadj=mpadj, Ffwalpha=Ffwalpha, mkws=mkws,
                                           add_out = 'Series')
                # name='_'.join([group_main,av,sg])
                name='_'.join([av,sg])
                df_out[name]=tmp
    if Transpose:
        df_out=df_out.T
    return df_out

def Hypo_test(df, groupby, ano_Var,
              group_str=None, ano_str=None,
              alpha=0.05, group_ren={},
              mcomp='TukeyHSD', mkws={},
              rel=False, rel_keys=[],
              add_T_ind=3, add_out = False):
    if ano_str is None:
        ano_str = ano_Var
    if group_str is None:
        group_str = groupby
    dft = df.copy(deep=True)
    if rel and rel_keys!=[]:
        dft.index = pd.MultiIndex.from_frame(dft.loc(axis=1)[rel_keys+[groupby]])
    else:
        dft.index = pd.MultiIndex.from_arrays([dft.index,
                                               dft.loc(axis=1)[groupby]])
    dft = dft[ano_Var]
    dft = dft.unstack(level=-1)
    if rel: dft=dft.dropna(axis=0)
    
    dfgr=dft.columns.values
    if not len(dfgr)==2:
        raise ValueError('More than two groups (%s)!'%dfgr)
    a = dft[dfgr[0]].dropna()
    b = dft[dfgr[1]].dropna()
    ano_df2=a.count()+b.count()-2 #Freiheitsgrad = Testpersonen pro Gruppe - 1
    
    if mcomp=='TukeyHSD':
        stats_test  = wraps(partial(stats.tukey_hsd, **mkws))(stats.tukey_hsd)
    elif mcomp=='ttest_ind':
        stats_test  = wraps(partial(stats.ttest_ind, **mkws))(stats.ttest_ind)
    elif mcomp=='ttest_rel':
        stats_test  = wraps(partial(stats.ttest_rel, **mkws))(stats.ttest_rel)
    elif mcomp=='mannwhitneyu':
        stats_test  = wraps(partial(stats.mannwhitneyu, **mkws))(stats.mannwhitneyu)
    elif mcomp=='wilcoxon':
        stats_test  = wraps(partial(stats.wilcoxon, **mkws))(stats.wilcoxon)
    else:
        raise NotImplementedError('Method %s for multi comparison not implemented!'%mcomp)
    t = stats_test(a,b)
    F = t.statistic
    p = t.pvalue
    if p < alpha:
        rtxt = 'H0 rejected!'
        H0=False
    else:
        rtxt = 'Fail to reject H0!'
        H0=True
    txt=("- F(%d) = %.3e, p = %.3e, for %s to %s (%s)"%(ano_df2,
                                                          F,p,
                                                          ano_str,group_str,
                                                          rtxt)) # Gruppen sind signifikant verschieden bei p<0.05
    if add_out is True:
        return txt, t
    elif add_out=='Series':
        return pd.Series({'DF2':ano_df2, 'Stat':F, 'p':p, 'H0':H0})
    elif add_out=='Test':
        return dft, pd.Series({'DF2':ano_df2, 'Stat':F, 'p':p, 'H0':H0}), txt
    else:
        return txt

def Hypo_test_multi(df, group_main='Series', group_sub=['A'],
                    ano_Var=['WC_vol'],
                    mcomp='mannwhitneyu', alpha=0.05, mkws={},
                    rel=False, rel_keys=[],
                    Transpose=True):
    df_out=pd.DataFrame([],dtype='O')
    # for sg in group_sub:
    #     for av in ano_Var:
    for av in ano_Var:
        for sg in group_sub:
            tmp=Hypo_test(df=df, groupby=(group_main,sg), 
                          ano_Var=(av,sg), alpha=alpha,
                          mcomp=mcomp, mkws=mkws,
                          # rel=rel, rel_keys=rel_keys, add_out = 'Series')
                          rel=rel, rel_keys=[(x,sg) for x in rel_keys], 
                          add_out = 'Series')
            # name='_'.join([group_main,av,sg])
            name='_'.join([av,sg])
            df_out[name]=tmp
    if Transpose:
        df_out=df_out.T
    return df_out

def CD_rep(pdo, groupby='Series', var='DEFlutoB', 
           det_met='SM-RRT', outtype='txt', tnform='{:.3e}'):
    """ Calculate critical differences and compare to given.
    """
    # grsagg=pdo.groupby(groupby)[var].agg(['mean','std','count'])
    grsagg=pdo[[groupby,var]].groupby(groupby).agg(['mean','std','count'])
    if isinstance(var,tuple):
        grsagg=grsagg.droplevel([0,1],axis=1)
    grs_sumhrn=grsagg['count'].apply(lambda x: 1/(2*x)).sum()
    grs_MD=grsagg['mean'].max()-grsagg['mean'].min()
    
    CDF= 1.96 * 2**0.5 # 2.8 only implemented for 95%-probability level (only two groups?)
    if det_met=='SM-RRT': #https://www.methodensammlung-bvl.de/resource/blob/208066/e536126ed1723145e51fc90b12736f5e/planung-und-statistische-auswertung-data.pdf
        allagg=pdo[var].agg(['mean','std','count'])
        CD=CDF*allagg['std']*np.sqrt(grs_sumhrn)
    elif det_met=='CV-DC': #https://flexikon.doccheck.com/de/Kritische_Differenz
        allagg=pdo[var].agg(['max',coefficient_of_variation])
        CD=abs(allagg['coefficient_of_variation']*CDF*allagg['max'])
    elif det_met=='CV': #https://link.springer.com/chapter/10.1007/978-3-662-48986-4_887
        allagg=pdo[var].agg([coefficient_of_variation])
        CD=abs(allagg['coefficient_of_variation'])*CDF
    elif det_met=='SD': #https://edoc.hu-berlin.de/bitstream/handle/18452/11713/cclm.1982.20.11.817.pdf?sequence=1
        allagg=pdo[var].agg(['std'])
        CD=abs(allagg['std'])*CDF
    else:
        raise NotImplementedError("Method %s not implemented!"%det_met)
        
    eta = grs_MD/CD
    # MD_l_CD=True if abs(grs_MD) < CD else False
    MD_l_CD=False
    t_grs_MD=tnform.format(grs_MD)
    t_CD    =tnform.format(CD)
    txt="{} >  {} -> Repeatability not verified! (eta={})".format(t_grs_MD,t_CD,eta,)
    if abs(grs_MD) <= CD:
        MD_l_CD=True
        txt="{} <= {} -> Repeatability verified! (eta={})".format(t_grs_MD,t_CD,eta,)
    
    if outtype=='Ser_all':
        out= pd.Series({'df_groups_agg':grsagg,'df_all_agg':allagg,
                        'groups_MD':grs_MD, 'CD':CD, 'eta':eta, 'MDlCD':MD_l_CD})
    elif outtype=='Tuple_all':
        out=grsagg,allagg,grs_MD,CD,eta,MD_l_CD
    elif outtype=='txt':
        out= txt
    elif outtype=='Series':
        out= pd.Series({'MD':grs_MD,'CD':CD, 'eta':eta, 'H0': MD_l_CD})
    return out


def CD_test_multi(df, group_main='Series', group_sub=['A'],
                    ano_Var=['WC_vol'],
                    det_met='SM-RRT',
                    Transpose=True):
    df_out=pd.DataFrame([],dtype='O')
    # for sg in group_sub:
    #     for av in ano_Var:
    for av in ano_Var:
        for sg in group_sub:
            tmp=CD_rep(pdo=df, groupby=(group_main,sg), 
                          var=(av,sg), det_met=det_met,
                          outtype = 'Series')
            # name='_'.join([group_main,av,sg])
            name='_'.join([av,sg])
            df_out[name]=tmp
    if Transpose:
        df_out=df_out.T
    return df_out


def Multi_conc(df, group_main='Donor', anat='VA', 
               met='Kruskal', alpha=0.05,
               stdict={'WC_vol':['A','B','L'],'WC_vol_rDA':['B','C','L'],
                       'lu_F_mean':['B'],'DEFlutoB':['C','G','L'],
                       'Hyst_An':['B'],'DHAntoB':['C','G','L']},
               rel=False, rel_keys=[], kws={}):
    """
    Multiple conclusion of statistical tests (variance analysis, hyphotesis test or critical differences)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe.
    group_main : str, optional
        Main group name. The default is 'Donor'.
    anat : str, optional
        Type of test.
        Implemented are:
            - VA: variance analysis
            - VAwoSg: variance analysis without subgrouping
            - HT: hyphotesis test
            - CD: critical differences
        The default is 'VA'.
    met : str, optional
        Methode of test. The default is 'Kruskal'.
    alpha : float, optional
        Test criterion to reject zero hypothesis.
        The default is 0.05.
    stdict : dict, optional
        Dictionary of value and list of variants to evaluate.
        The default is {'WC_vol':['A','B','L'],'WC_vol_rDA':['B','C','L'],
                        'lu_F_mean':['B'],'DEFlutoB':['C','G','L'], 
                        'Hyst_An':['B'],'DHAntoB':['C','G','L']}.
    rel : bool, optional
        Switch for relation of samples (True= related). The default is False.
    rel_keys : list, optional
        Variabel/column names for determining relation. The default is [].
    kws : dict, optional
        Dictionarie of additional Keyword arguments for selected test. 
        The default is {}.

    Returns
    -------
    out : pd.DataFrame
        Dataframe of results.

    """
    out=pd.DataFrame([],dtype='O')
    for i in stdict.keys():
        if anat=='VA':
            out2=group_ANOVA_MComp_multi(df=df, 
                                 group_main=group_main, 
                                 group_sub=stdict[i],
                                 ano_Var=[i],
                                 mpop=met, alpha=alpha, **kws)
        if anat=='VAwoSg':
            out2=group_ANOVA_MComp_multi(df=df, 
                                 group_main=group_main, 
                                 group_sub=[],
                                 ano_Var=[i],
                                 mpop=met, alpha=alpha, **kws)
        elif anat=='HT':
            out2=Hypo_test_multi(df=df, 
                                 group_main=group_main, 
                                 group_sub=stdict[i],
                                 ano_Var=[i], mcomp=met, alpha=alpha,
                                 rel=rel, rel_keys=rel_keys, **kws)
        elif anat=='CD':
            out2=CD_test_multi(df=df, 
                                 group_main=group_main, 
                                 group_sub=stdict[i],
                                 ano_Var=[i], det_met=met, **kws)
        out=pd.concat([out,out2])
    out.index=pd.MultiIndex.from_product([[group_main],out.index])
    return out

def Corr_ext(df, method='spearman', 
             sig_level={0.001:'$^a$',0.01:'$^b$',0.05:'$^c$',0.10:'$^d$'},
             corr_round=2):
    """
    Performs correlation according to method and significance levels.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    method : string, optional
        Correlation method, implemented are:
            - Pearson: ['pearson','Pearson','P']
            - Spearman: ['spearman', 'Spearman','S']
            - Kendalltau: ['kendall', 'kendalltau', 'Kendall', 'Kendalltau', 'K']
        The default is 'spearman'.
    sig_level : list or dict, optional
        Significance level for anotation, if list: ascending number of *´s, 
        if dict: dictionary values.
        The default is {0.001:'$^a$',0.01:'$^b$',0.05:'$^c$',0.10:'$^d$'}.
    corr_round : int or None, optional
        Number of digits for rounding of annotation of correlation. 
        The default is 2.

    Raises
    ------
    NotImplementedError
        Method fpr correlation not implemented.

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        Output dataframes (Correlation, Annotation strings with significance levels).

    """
    # quelle: https://stackoverflow.com/questions/52741236/how-to-calculate-p-values-for-pairwise-correlation-of-columns-in-pandas
    if method in ['pearson','Pearson','P']:
        df_c = df.corr(method='pearson')
        pval = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1])
    elif  method in ['spearman', 'Spearman','S']:
        df_c = df.corr(method='spearman')
        pval = df.corr(method=lambda x, y: stats.spearmanr(x, y)[1])
    elif method in ['kendall', 'kendalltau', 'Kendall', 'Kendalltau', 'K']:
        df_c = df.corr(method='kendall')
        pval = df.corr(method=lambda x, y: stats.kendalltau(x, y)[1])
    else:
        raise NotImplementedError('Correlation method %s not implemented!'%method)
    if isinstance(corr_round,int):
        df_c=df_c.round(corr_round)
    if isinstance(sig_level, list):
        p = pval.applymap(lambda x: ''.join(['*' for t in sig_level if x<=t]))
    else:
        def p_extender(x, sld):
            if x > np.max(list(sld.keys())):
                return ''
            else:
                for i in np.sort(list(sld.keys())):
                    if x <= i:
                        return sld[i]
        p = pval.applymap(lambda x: p_extender(x, sld=sig_level))
    tmpf='{'+':.'+'%d'%(corr_round) +'f}'
    df_c2 = df_c.applymap(lambda x: tmpf.format(x)) + p
    return df_c, df_c2

#%% Plotting
def plt_handle_suffix(fig, path='foo', tight=True, show=True, 
                      save=True, s_types=["pdf","png"], 
                      clear=True, close=True):
    """
    Handler for end of plotting procedure.

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        Figure instance.
    path : str, optional
        Save path. The default is 'foo'.
    tight : bool, optional
        Tight the layout. The default is True.
    show : bool, optional
        Show figure. The default is True.
    save : bool, optional
        Save figure. The default is True.
    s_types : list of str, optional
        Types in which the figure is saved. The default is ["pdf","png"].
    clear : bool, optional
        Clear figure instance. The default is True.
    close : bool, optional
        Close figure window. The default is True.

    Returns
    -------
    None.

    """
    if tight: fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if show:  plt.show()
    if save and not (path is None):
        for st in s_types:
            # plt.savefig(path+'.'+st)
            fig.savefig(path+'.'+st)
    if clear: fig.clf()
    if close: plt.close(fig)

def tick_label_renamer(ax, renamer={}, axis='both'):
    def get_ren_label(ax, renamer, axis):
        if axis=='x':
            label = ax.get_xticklabels()
        elif axis=='y':
            label = ax.get_yticklabels()
        else:
            raise NotImplementedError('Axis %s not implemented!'%axis)
        label = pd.Series([item.get_text() for item in label])
        label = label.replace(renamer).values
        if axis=='x':
            label = ax.set_xticklabels(label)
        elif axis=='y':
            label = ax.set_yticklabels(label)
        else:
            raise NotImplementedError('Axis %s not implemented!'%axis)
    if axis=='both':
        get_ren_label(ax=ax, renamer=renamer, axis='x')
        get_ren_label(ax=ax, renamer=renamer, axis='y')
    elif axis=='x':
        get_ren_label(ax=ax, renamer=renamer, axis='x')
    elif axis=='y':
        get_ren_label(ax=ax, renamer=renamer, axis='y')
    else:
        raise NotImplementedError('Axis %s not implemented!'%axis)
        
def tick_legend_renamer(ax, renamer={}, title=''):
    # legend = ax.axes.flat[0].get_legend()
    legend = ax.axes.get_legend()
    if title != '':
        legend.set_title(title)
    for i in legend.texts:
        i.set_text(renamer[i.get_text()])
    
def tick_label_inserter(ax, pos=0, ins='', axis='both'):
    def get_ren_label(ax, pos, ins, axis):
        if axis=='x':
            label = ax.get_xticklabels()
        elif axis=='y':
            label = ax.get_yticklabels()
        else:
            raise NotImplementedError('Axis %s not implemented!'%axis)
        label = pd.Series([item.get_text() for item in label])
        if pos == 0:
            label = label.apply(lambda x: '{}{}'.format(ins,x))
        elif pos == -1:
            label = label.apply(lambda x: '{}{}'.format(x,ins))
        else:
            label = label.apply(lambda x: '{}{}{}'.format(x[:pos],ins,x[pos:]))
        label = label.values
        if axis=='x':
            label = ax.set_xticklabels(label)
        elif axis=='y':
            label = ax.set_yticklabels(label)
        else:
            raise NotImplementedError('Axis %s not implemented!'%axis)
    if axis=='both':
        get_ren_label(ax=ax, pos=pos, ins=ins, axis='x')
        get_ren_label(ax=ax, pos=pos, ins=ins, axis='y')
    elif axis=='x':
        get_ren_label(ax=ax, pos=pos, ins=ins, axis='x')
    elif axis=='y':
        get_ren_label(ax=ax, pos=pos, ins=ins, axis='y')
    else:
        raise NotImplementedError('Axis %s not implemented!'%axis)
        
#%%% Seaborn extra
import seaborn as sns
def sns_pointplot_MMeb(ax, data, x,y, hue=None,
                       dodge=0.2, join=False, palette=None,
                       markers=['o','P'], scale=1, barsabove=True, capsize=4,
                       controlout=False):
    """Generate Pointplot with errorbars marking minimum and maximum instead of CI."""
    axt = sns.pointplot(data=data, 
                         x=x, y=y, hue=hue,
                         ax=ax, join=join, dodge=dodge, legend=False,scale=scale,
                         markers=markers, palette=palette,
                         ci=None, errwidth=1, capsize=.1)
    if hue is None:
        # erragg = data.groupby(x).agg(['mean','min','max'])[y]
        erragg = data.groupby(x).agg(['mean','min','max'],sort=False)[y]
    else:
        # erragg = data.groupby([hue,x]).agg(['mean','min','max'])[y]
        erragg = data.groupby([hue,x]).agg(['mean','min','max'],sort=False)[y]
    # very qnd!!! (sonst falsch zugeordnet, prüfen!!!)
    if hue is not None:
        orgind=[data[hue].drop_duplicates().to_list(),data[x].drop_duplicates().to_list()]
        erragg=erragg.sort_index().loc(axis=0)[pd.MultiIndex.from_product(orgind)]
        
    errors=erragg.rename({'mean':'M','min':'I','max':'A'},axis=1)
    errors=errors.eval('''Min = M-I
                          Max = A-M''').loc(axis=1)[['Min','Max']].T
    i=0
    for point_pair in axt.collections:
        if hue is None:
            if i<1:
                i+=1
                colors=point_pair.get_facecolor()[0]
                x_coords = []
                y_coords = []
                for x, y in point_pair.get_offsets():
                    x_coords.append(x)
                    y_coords.append(y)
                ax.errorbar(x_coords, y_coords, yerr=errors.values,
                            c=colors, fmt=' ', zorder=-1, barsabove=barsabove, capsize=capsize)
        elif (i<=len(errors.columns.get_level_values(0).drop_duplicates())-1):
            errcol=errors.columns.get_level_values(0).drop_duplicates()[i]
            i+=1
            colors=point_pair.get_facecolor()[0]
            x_coords = []
            y_coords = []
            for x, y in point_pair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)
            ax.errorbar(x_coords, y_coords, yerr=errors.loc(axis=1)[errcol].values,
                        c=colors, fmt=' ', zorder=-1, barsabove=barsabove, capsize=capsize)
    if controlout:
        return erragg