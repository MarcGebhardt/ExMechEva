# -*- coding: utf-8 -*-
"""
General functions for bending.

@author: MarcGebhardt
"""
import numpy as np

#%% triangle function
def triangle_func_d0(x, xmin,xmax, f_0):
    """Standard triangle function with maximum in middle between min. and max. x coordinate."""
    eq=f_0*(1-2*abs(x)/(xmax-xmin))
    return eq
def triangle_func_d1(x, xmin,xmax, f_0):
    """First derivate of standard triangle function."""
    eq=f_0*(-2*np.sign(x)/(xmax-xmin))
    return eq
def triangle_func_d2(x, xmin,xmax, f_0):
    """Second derivate of standard triangle function (equal 0)."""
    eq=0
    return eq

#%% shear deformation
def Shear_area(Area, CS_type='Rectangle', kappa=None):
    """
    Returns the shear area of a cross section.

    Parameters
    ----------
    Area : float or function or 1
        Cross section area.
    CS_type : string, optional
        Cross section type. The default is 'Rectangle'.
    kappa : float, optional
         Correction factor shear area. Depending to CS_type. The default is None.

    Returns
    -------
    AreaS : same type as Area
        Shear area.

    """
    if  CS_type=='Rectangle': kappa = 5/6
    AreaS = Area * kappa
    return AreaS

def gamma_V_det(poisson, t_mean, Length, CS_type='Rectangle', kappa=None):
    """
    Returns the ratio between shear to entire deformation in mid of bending beam.

    Parameters
    ----------
    poisson : float
        Poisson's ratio.
    t_mean : float
        Mean thickness.
    Length : float
        Distance between load bearings.
    CS_type : string, optional
        Cross section type. The default is 'Rectangle'.
    kappa : float, optional
        Correction factor shear area. Depending to CS_type. The default is None.

    Returns
    -------
    gamma_V : float
        Ratio between shear to entire deformation in mid of bending beam.

    """
    kappa = Shear_area(1, CS_type, kappa)
    
    gamma_V = 2 / kappa * (1 + poisson) * t_mean**2 / Length**2
    return gamma_V