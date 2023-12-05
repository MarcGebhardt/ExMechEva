# -*- coding: utf-8 -*-
"""
Adds functionality to temporary manipulate measurement curves.

@author: MarcGebhardt
"""
import warnings
import numpy as np
import pandas as pd

from .pd_ext import (pd_isSer)

#%% Extension and retrimming
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

#%% shifting
def DetFinSSC(mdf, YM, iS, iLE=None, 
              StressN='Stress', StrainN='Strain', 
              addzero=True, izero=None, option='YM'):
    """
    Determine final stress-strain curve (moved to strain offset from elastic modulus).

    Parameters
    ----------
    mdf : pd.DataFrame
        Original stress-strain curve data.
    YM : dict or pd.Series or float
        Elastic modulus (float) or 
        elastic modulus (key=E) and intersection on strain=0 (key=Eabs) or
        direct input of strain offset (if option is in ['strain_offset','SO','so']).
    iS : index of mdf
        Start of linear behavior (all values befor will dropped).
    iLE : TYPE, optional
        End of linear behavior (if None, strain offset will determined only with iS). 
        The default is None.
    StressN : string, optional
        Column name for stress variable. The default is 'Stress'.
    StrainN : TYPE, optional
        Column name for strain variable. The default is 'Strain'.
    addzero : bool, optional
        Switch to adding Zero value (index,StressN,StrainN=0).
        The default is True.
    izero: index of mdf, optional
        Index for zero value line (p.e. to match old start index)
    option: string, optional
        Option for strain offset. Possible:
            - 'YM': Strain offset determined by elastic modulus.
            - ['strain_offset','SO','so']: Direct input of strain offset.
        The default is 'YM'.

    Returns
    -------
    out : pd.DataFrame
        Moved stress-strain curve data.
    so : float
        Strain offset.

    """
    out=mdf.loc[iS:,[StressN,StrainN]].copy(deep=True)
    if option in ['YM']:
        if isinstance(YM,dict) or pd_isSer(YM):
            so=-YM['E_abs']/YM['E']
        elif isinstance(YM,float):
            if iLE is None:
                lindet=out.iloc[0][[StressN,StrainN]]
            else:
                lindet=out.loc[[iS,iLE],[StressN,StrainN]].mean(axis=0)
            so=lindet[StrainN]-lindet[StressN]/YM
    elif option in ['strain_offset','SO','so']:
        so = YM
    else:
        raise NotImplementedError("Option %s not implemented!"%option)
    out[StrainN]=out[StrainN]-so
    if addzero:
        if izero is None:
            i=0
        else:
            i=izero
        tmp=pd.DataFrame([],columns=out.columns, index=[i])
        tmp[[StressN,StrainN]]=0
        out=pd.concat([tmp,out],axis=0)
    return out, so