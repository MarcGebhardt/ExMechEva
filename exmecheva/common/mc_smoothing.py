# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:22:06 2023

@author: MarcGebhardt
"""
import warnings
import numpy as np
import pandas as pd
import scipy

from .mc_man import (check_params, Extend_Series_Poly, Predict_apply_retrim)

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
                  # pfunc=Extend_Series_Poly, # not possible to use before defined
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
                               # func=Extend_Series_Poly, kwargs=pkwargs)
                               func=pfunc, kwargs=pkwargs)
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
