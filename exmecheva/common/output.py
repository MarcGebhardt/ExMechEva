# -*- coding: utf-8 -*-
"""
Functions for output data generation.

@author: MarcGebhardt
"""
import numpy as np
import pandas as pd

from .pd_ext import pd_trapz

def Outvalgetter(mdf, V, VIP, exacts=None, 
                 order=['f','e','U'], add_esuf='_con',
                 n_strain='Strain', n_stress='Stress',
                 use_exacts=True):
    if exacts is None: use_exacts=False
    out=pd.Series([], dtype='float64')
    for i in order:
        if i in ['f','F']:
            name = i+V.lower()
            if use_exacts and (V in exacts.index):
                out[name]=exacts.loc[V,n_stress]
            elif (V in VIP.index):
                out[name]=mdf.loc[VIP[V],n_stress]
            else:
                out[name]=np.nan
        elif i in ['e','s']:
            name = i+V.lower()+add_esuf
            if use_exacts and (V in exacts.index):
                out[name]=exacts.loc[V,n_strain]
            elif (V in VIP.index):
                out[name]=mdf.loc[VIP[V],n_strain]
            else:
                out[name]=np.nan
        elif i in ['U','W']:
            name = i+V.lower()+add_esuf
            if use_exacts and (V in exacts.index):
                tmp = exacts.loc[V,[n_strain,n_stress]]
                tmp.name = exacts.loc[V,'ind_ex']
                tmp = mdf.append(tmp).sort_index()
                out[name]=pd_trapz(tmp.loc[:exacts.loc[V,'ind_ex']],
                                        y=n_stress, x=n_strain,
                                        axis=0, nan_policy='omit')
            elif (V in VIP.index):
                out[name]=pd_trapz(mdf.loc[:VIP[V]],
                                        y=n_stress, x=n_strain,
                                        axis=0, nan_policy='omit')
            else:
                out[name]=np.nan
    return out

def Otvalgetter_Multi(mdf, Vs=['Y','YK','Y0','Y1','U','B'],
                      datasep=['con','opt'], VIPs={'con':None,'opt':None}, 
                      exacts={'con':None,'opt':None}, 
                      orders={'con':['f','e','U'],'opt':['e','U']}, 
                      add_esufs={'con':'_con','opt':'_opt'},
                      n_strains={'con':'Strain','opt':'DStrain'}, 
                      n_stresss={'con':'Stress','opt':'Stress'},
                      use_exacts=True):
    out = pd.Series([],dtype='float64')
    for V in Vs:
        for ds in datasep:
            out=out.append(Outvalgetter(mdf=mdf, V=V, VIP=VIPs[ds], exacts=exacts[ds], 
                                    order=orders[ds], add_esuf=add_esufs[ds],
                                    n_strain=n_strains[ds], n_stress=n_stresss[ds],
                                    use_exacts=use_exacts))
    return out