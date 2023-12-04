# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:21:05 2023

@author: mgebhard
"""
import numpy as np
import pandas as pd

#%% tests
def check_empty(x, empty_str=['',' ','  ','   ',
                              '#NV''NaN','nan','NA']):
    """
    Tests if given variable (string or float) can interpreted as empty.

    Parameters
    ----------
    x : None or string or float
        Given variable to test.
    empty_str : list of strings
        Strings determining if variable can interpreted as empty.

    Returns
    -------
    t : bool
        Test result.

    """
    t = (x is None or 
         x in empty_str or
         (isinstance(x, float) and  np.isnan(x)))
    return t

def str_to_bool(x, str_true=["True","true","Yes","1","On"]):
    """
    Interpretes a given string as boolean value

    Parameters
    ----------
    x : string or bool
        Value to be interpreted.
    str_true : list of strings
        List of strings interpreted as True.

    Returns
    -------
    t : bool
        Boolean value result.

    """
    
    if x is True:
        t = x
    elif x in str_true:
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
    rdig = sigdig(x, sd)
    xr =  round(x, rdig)
    return xr

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
    # if r1 == r2:
    #     raise ValueError("Lines are parallel (no intersection)!")
    #     return None
    x = (c2-c1)/(r1-r2)
    if out == 'x':
        return x
    else: 
        y = r1 * x + c1
        if out == 'y':
            return x, y
        elif out == 'xy':
            return x, y
    
def Line_from2P(P1, P2):
    """Determine line from two points (1st argument is x)"""
    slope = (P2[1] - P1[1]) / (P2[0] - P1[0])
    constant = P1[1] - slope * P1[0]
    return slope, constant
    