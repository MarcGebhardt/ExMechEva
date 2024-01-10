# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:43:14 2023

@author: mgebhard
"""
import warnings

import numpy as np
import pandas as pd
import lmfit

from .pd_ext import (pd_combine_index)

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

def Refit_YM_vals(m_df, YM, VIP, n_strain='Strain', n_stress='Stress',
              n_loBo=['F3'], n_upBo=['F4'], option='range', outopt='Series'):
    """
    Refits line values for given rising (i.e. absolute value and R² vor given elastic modulus)

    Parameters
    ----------
    m_df : pd.DataFrame
        Measured data.
    YM : float
        Youngs Modulus (fixed rising of line).
    VIP : pd.Series
        Important points corresponding to measured data.
    n_strain : string
        Name of used strain (have to be in measured data).
    n_stress : string
        Name of used stress (have to be in measured data).
    n_loBo : [string]
        List of lower borders for determination (have to be in VIP). The default is ['F3'].
    n_upBo : [string]
        List of upper borders for determination (have to be in VIP). The default is ['F4'].
    option : string, optional
        Determination range. The default is 'range'.
    outopt : string, optional
        Switch for outupt (Series or list). The default is 'Series'.

    Returns
    -------
    out : list or pd.Series
        Line values and firt parameters for fixed rising.

    """
    m_lim = m_df.loc[min(VIP[n_loBo]):max(VIP[n_upBo])]
    if option == 'range':
        fitm = lmfit.models.LinearModel()
        fitpar = fitm.make_params()
        fitpar.add('slope', value=YM, vary = False)
        fit=fitm.fit(m_lim[n_stress], x=m_lim[n_strain], 
                     params=fitpar,method='leastsq', nan_policy='omit')
        E       = fit.params.valuesdict()['slope']
        Eabs    = fit.params.valuesdict()['intercept']
        Rquad    = 1 - fit.residual.var() / np.nanvar(m_lim['Stress'])
    out=E, Eabs, Rquad, fit
    if outopt=='Series':
        out= pd.Series(out, index=['E','E_abs','Rquad','Fit_result'])
    return out

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

def strain_linfit(stress_ser, YM, YM_abs, strain_offset=0.002):
    """Linearised strain fit corresponding Youngs Modulus and stress"""
    strain_fit_ser=(stress_ser - YM_abs) / YM + strain_offset
    return strain_fit_ser