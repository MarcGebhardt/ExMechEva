# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:39:45 2023

@author: mgebhard
"""
import warnings
import numpy as np
import pandas as pd
import scipy
# import scipy.signal as scsig
# import scipy.interpolate as scint
# import scipy.ndimage.filters as ndif
# import scipy.ndimage as scnd

from .helper import check_empty
from .pd_ext import pd_valid_index

#%% smoothing
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
        conmet = scipy.ndimage.convolve1d
        box_mode_std = 'nearest'
    elif conv_method == '2D':
        conmet = scipy.signal.convolve2d
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
        spline=scipy.interpolate.UnivariateSpline(xi, xa, **smooth_opts)
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
        out = scipy.ndimage.filters.uniform_filter1d(xa, **smooth_opts)
        
    elif smooth_type=='SavGol':
        box_pts  = smooth_opts['window_length']
        out = scipy.signal.savgol_filter(xa,**smooth_opts)
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

#%% core analysing
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
    
    o=df2.agg(scipy.signal.find_peaks,**fp_kwargs)
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

#%% elastic modulus+
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

def Yield_redet(m_df, VIP, n_strain, n_stress,
                n_loBo, n_upBo, n_loBo_int, 
                YM, YM_abs, strain_offset=0.002, 
                rise_det=[True,4], n_yield='Y'):
    """
    Redetermine yield point to different conditions 
    (intersection with linearised strain offset (ones after), zero rising, fixed endpoint).

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
    # stress_fit = stress_linfit(m_lim[n_strain],
    #                            YM, YM_abs, strain_offset)
    # i = m_lim[n_stress]-stress_fit
    # i_sign, i_sich = sign_n_change(i)
    strain_fit = strain_linfit(m_lim[n_stress],
                               YM, YM_abs, strain_offset)
    i = m_lim[n_strain]-strain_fit
    i_sign, i_sich = sign_n_change(i)
    i_sich.iloc[0]=False # First signchange always true
    i_sich = i_sich.loc[min(VIP[n_loBo_int]):] # Y erst ab F- suchen
    
    _,_,r_sich = rise_curve(m_df[n_stress],*rise_det)
    # r_sich = r_sich.loc[min(VIP[n_loBo])+2:max(VIP[n_upBo])+2]
    r_sich = r_sich.loc[min(VIP[n_loBo_int])+2:max(VIP[n_upBo])+2]
    
    VIP_new = VIP
    txt =""
    if i_sign.loc[min(VIP[n_loBo_int])] >= 0:
        VIP_new[n_yield] = min(VIP[n_loBo_int])
        txt+="Fy set on 1st point of lower border of intersection (lower than %.3f %% pl. strain)!"%(strain_offset*100)
    # if (i_sich.any()==True)and(r_sich.any()==True):
    elif (i_sich.any()==True)and(r_sich.any()==True):
        if(i_sich.loc[i_sich].index[0])<=(r_sich.loc[r_sich==True].index[0]):
            VIP_new[n_yield] = i_sich.loc[i_sich].index[0]
            txt+="Fy set on intersection with %.3f %% pl. strain!"%(strain_offset*100)
        else:
            VIP_new[n_yield]=r_sich.loc[r_sich==True].index[0]-2 #-2 wegen Anstiegsberechnung
            txt+="Fy on first point between F+ and Fu with rise of 0, instead intersection %.3f %% pl. strain! (earlier)"%(strain_offset*100)
    elif (i_sich.any()==True)and(r_sich.all()==False):
        VIP_new[n_yield]=i_sich.loc[i_sich].index[0]
        txt+="Fy set on intersection with %.3f %% pl. strain!"%(strain_offset*100)
    else:
        if r_sich.any():
            VIP_new[n_yield]=r_sich.loc[r_sich==True].index[0]-2
            txt+="Fy on first point between F+ and Fu with rise of 0, instead intersection %.3f %% pl. strain! (No intersection found!)"%(strain_offset*100)
        else:
            tmp = VIP[VIP==max(VIP[n_upBo])].index[0]
            VIP_new[n_yield] = VIP[tmp]
            txt+="Fy set to %s (max of %s), instead intersection %.3f %% pl. strain or rise of 0! (No intersection found!)"%(tmp,n_upBo,(strain_offset*100))
    VIP_new = VIP_new.sort_values()
    return VIP_new, txt
    
def Find_intg2p(gsl, gin, pdo, i1=None, i2=None, 
                    x='Strain', y='Stress', so=0, n_yield='Y'):
    """Determine intersection of line and point (usefull to find yield point)"""
    pdn = pd_limit(pdo, i1, i2)[[x,y]]
    psl, pin = Line_from2P(pdn.iloc[0].values,pdn.iloc[-1].values)
    gso=pd.DataFrame([0,1], columns=[y])
    gso[x]=gso[y].apply(lambda x: strain_linfit(x, gsl, gin, so))
    gso=gso[[x,y]]
    gslso, ginso = Line_from2P(gso.iloc[0].values,gso.iloc[-1].values)
    inter = Inter_Lines(r1=gslso, c1=ginso, r2=psl, c2=pin, out='xy')
    i = pdn.iloc[0].name + (pdn.iloc[-1].name-pdn.iloc[0].name)*(inter[1]-pdn.iloc[0][y])/(pdn.iloc[-1][y]-pdn.iloc[0][y])
    inter = pd.Series([i,*inter], index=['ind_ex',x,y], name=n_yield)
    return inter

def Yield_redet2(m_df, VIP, n_strain, n_stress,
                n_loBo, n_upBo, n_loBo_int,
                YM, YM_abs, strain_offset=0.002,
                use_rd =True, rise_det=[True,4],
                n_yield='Y', ywhere='n'):
    """
    Redetermine yield point to different conditions 
    (intersection with linearised strain offset (ones after), zero rising, fixed endpoint).

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
    use_rd : bool, optional
        Switch for using change in stress rising. The default is True.
    rise_det : [bool, int], optional
        Determination options for stress rising ([smoothing, smoothing factor]).
        The default is [True,4].
    n_yield : string, optional
        Name of yield point (have to be in VIP). The default is 'Y'.
    ywhere : string, optional
        Which point should be choosen (n-next, a-after, b-before). The default is 'n'.

    Returns
    -------
    VIP_new : pd.Series
        Important points corresponding to measured data.
    txt : string
        Documantation string.

    """
    # Intersection with linearization and strain offset
    m_lim = m_df.loc[min(VIP[n_loBo]):max(VIP[n_upBo])]
    strain_fit = strain_linfit(m_lim[n_stress],
                               YM, YM_abs, strain_offset)
    i = m_lim[n_strain]-strain_fit
    i_sign, i_sich = sign_n_change(i)
    i_sich.iloc[0]=False # First signchange always true

    # find change in rise of stress
    if use_rd:
        _,_,r_sich = rise_curve(m_df[n_stress],*rise_det)
        r_sich = r_sich.loc[min(VIP[n_loBo_int])+2:max(VIP[n_upBo])+2]
        if r_sich.any()==True:
            i_r_sich_1st=r_sich.loc[r_sich==True].index[0]-2 # -2  due to determination
        else:
            i_r_sich_1st=max(VIP[n_upBo]) # no signchange, last point
    else:
        i_r_sich_1st=max(VIP[n_upBo]) # no signchange, last point

    # Intersection only from determination limit and to maximum/last sign change
    i_sign = i_sign.loc[min(VIP[n_loBo_int]):i_r_sich_1st]
    i_sich = i_sich.loc[min(VIP[n_loBo_int]):i_r_sich_1st]

    VIP_new = VIP
    plst_txt="%.3f %% plastic strain"%(strain_offset*100)
    txt ="%s "%(n_yield)
    if i_sich.any():
        iy=i_sich.loc[i_sich].index[-1] # Signchange allways after 'a'
        yexser=Find_intg2p(gsl=YM, gin=YM_abs, pdo=m_lim, 
                           i1=pd_valid_index(iy, i, opt='b'), i2=iy, 
                           x=n_strain, y=n_stress, so=strain_offset,n_yield=n_yield)
        if ywhere=='b':
            iy=pd_valid_index(iy, i, opt='b')
        elif ywhere=='n':
            iy=Find_closest(i.loc[[pd_valid_index(iy, i, opt='b'),iy]],0)
        txt+="set on intersection with %s (option=%s)"%(plst_txt,ywhere)
    else:
        if (i_sign <= 0).all():
            iy=i_r_sich_1st
            if i_r_sich_1st==max(VIP[n_upBo]):
                tmp = VIP[VIP==max(VIP[n_upBo])].index[0]
                txt+="set to %s (max of %s), instead intersection %s or rise of 0! (No intersection found!)"%(tmp,n_upBo,plst_txt)
            else:
                txt+="on first point between F+ and Fu with rise of 0, instead intersection %s! (No intersection found!)"%(plst_txt)
        else:
            iy=min(VIP[n_loBo_int])
            tmp = VIP[VIP==min(VIP[n_loBo_int])].index[0]
            txt+="set to %s (min of %s), instead intersection %s or rise of 0! (No intersection found!)"%(tmp,n_loBo_int,plst_txt)
        yexser = pd.Series([iy,*m_lim.loc[iy][[n_strain,n_stress]].values], 
                           index=['ind_ex',n_strain,n_stress], name=n_yield)
    VIP_new[n_yield] = iy
    VIP_new = VIP_new.sort_values()
    return VIP_new, txt, yexser

def Yield_redet2_Multi(m_df, VIP, YM, YM_abs,
                   strain_osd={'YK':0.0,'Y0':0.0,'Y':0.2/100,'Y1':0.007/100}, 
                   strain_osdf={'YK':'F4'},
                   n_strain='Strain', n_stress='Stress',
                   n_loBo=['F3'], n_upBo=['U'], n_loBo_int=['F3'],
                   use_rd =True, rise_det=[True,2], 
                   ywhere='n'):
    txt=''
    y_exact=pd.DataFrame(pd.Series(strain_osd), columns=['strain_os'])
    for i in strain_osd.keys():
        if i in strain_osdf.keys():
            VIP[i] = VIP[strain_osdf[i]]
            txt+="\n %s set on %s (fixed)"%(i, strain_osdf[i])
            tmp=VIP[i],VIP[i],*m_df.loc[VIP[i],[n_strain,n_stress]].values
            y_exact.loc[i, ['ind','ind_ex',n_strain,n_stress]] = tmp
        else:
            tmp = Yield_redet2(m_df=m_df, VIP=VIP,
                                n_strain=n_strain, n_stress=n_stress,
                                n_loBo=n_loBo, n_upBo=n_upBo, n_loBo_int=n_loBo_int,
                                YM     = YM, YM_abs = YM_abs,
                                strain_offset=strain_osd[i],
                                use_rd =use_rd, rise_det=rise_det, 
                                n_yield=i, ywhere=ywhere)
            VIP = tmp[0]
            txt += '\n '+ tmp[1]
            y_exact.loc[i, ['ind','ind_ex',n_strain,n_stress]] = VIP[i],*tmp[2].values
    return VIP, y_exact, txt

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