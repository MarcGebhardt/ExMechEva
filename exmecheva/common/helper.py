# -*- coding: utf-8 -*-
"""
Contains test and basic numeric functionality.

@author: MarcGebhardt
"""
import warnings
import numpy as np

#%% tests
def _test_overwrites(impmods, overwr):
    """
    Tests if function in module. Use while importing.

    Parameters
    ----------
    impmods : list, optional
        Modules to search for function overwrites. The default is [].
    overwr : list, optional
        Function overwrites to check in modules. The default is [].

    Raises
    ------
    ImportWarning
        Function in module and will be overwritten.

    """
    def funcinmod(func, mod):
        if str(func) in dir(mod):
            o = True
        else:
            o = False
        return o
    
    # _overrides=["indlim","meanwoso","stdwoso"]
    # _impmods=[pd.Series,pd.DataFrame]
    for ta in overwr:
        for tb in impmods:
            if funcinmod(func=ta, mod=tb):
                warnings.warn("Function %s in %s and will be overwritten!"%(ta,tb),
                              ImportWarning)


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

