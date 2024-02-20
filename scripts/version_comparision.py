# -*- coding: utf-8 -*-
"""
Performs comparision of different evaluations

@author: MarcGebhardt
"""
import os
import pandas as pd

def reldev(a,b):
    return (b.sub(a)).div(a)

# Option selection
# option = 'Series'
option = 'Complete'

rel_params=['fy','ey_con','Uy_con','ey_opt','Uy_opt',
            'fu','eu_con','Uu_con','eu_opt','Uu_opt',
            'E_lsq_F_A0Al_E','E_lsq_R_A0Al_E',
            'E_inc_F_D2MGwt_meanwoso','E_inc_R_D2MGwt_meanwoso']

#%% Series comparision
if option == 'Series':
    # Inputs
    # Main path
    # path_main="F:/Messung/009-210120_Becken7-DBV/"
    # path_main="F:/Messung/008-201124_Becken6-ADV/"
    path_main="F:/Messung/008-201117_Becken6-AZV/"
    # Protocol
    # name_protokoll="210120_Becken7-DBV_Protokoll_new.xlsx"
    # name_protokoll="201124_Becken6-ADV_Protokoll_new.xlsx"
    name_protokoll="201117_Becken6-AZV_Protokoll_new.xlsx"
    # Path builder
    path_ausw="Auswertung/"
    # First evaluation path (folder of single hdf-files)
    pv1=path_main+path_ausw+"ExMechEva/"
    # Second evaluation path (folder of single hdf-files)
    pv2=path_main+path_ausw+"Test_py/"
    
    prot=pd.read_excel(path_main+path_ausw+name_protokoll, 
                       header=11, index_col=0, 
                       na_values={"Nan"," Nan","nan",""})
    prot=prot.drop('[-]')
    
    
    prot_lnr=prot.index
    WM="" # suffix für Feuchte
    for i in prot_lnr:
        name_dic=prot.loc[i]['Designation']+WM
        path1=pv1+name_dic+'.h5'
        if os.path.exists(path1):
            HDFst=pd.HDFStore(path1, mode='r')
            out_tab_1 = HDFst['Material_Parameters']
            HDFst.close()
        else:
            print("Not existing: ",path1)
            out_tab_1 = []
        path2=pv2+name_dic+'.h5'
        if os.path.exists(path1):
            HDFst=pd.HDFStore(path2, mode='r')
            out_tab_2 = HDFst['Material_Parameters']
            HDFst.close()
        else:
            print("Not existing: ",path1)
            out_tab_2 = []
    
        if i==prot_lnr[0]:
            out_tab_zsf1=out_tab_1
            out_tab_zsf2=out_tab_2
        else:
            out_tab_zsf1=out_tab_zsf1.append(out_tab_1)
            out_tab_zsf2=out_tab_zsf2.append(out_tab_2)
    
#%% Complete (hdf) comparision
elif option == 'Complete':
    path_main="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/"
    # #TBT:
    # path1=path_main+"240118/TBT/B3-B7_TBT-Summary.h5"
    # path2=path_main+"240219/TBT/B3-B7_TBT-Summary.h5"
    # #ACT:
    # path1=path_main+"240118/ACT/B3-B7_ACT-Summary.h5"
    # path2=path_main+"240219/ACT/B3-B7_ACT-Summary.h5"
    #ATT:
    path1=path_main+"240118/ATT/B3-B7_ATT-Summary.h5"
    path2=path_main+"240219/ATT/B3-B7_ATT-Summary.h5"
    
    HDFst=pd.HDFStore(path1, mode='r')
    out_tab_zsf1 = HDFst['Summary']
    HDFst.close()
    HDFst=pd.HDFStore(path2, mode='r')
    out_tab_zsf2 = HDFst['Summary']
    HDFst.close()
    
#%% Common
# Numeric columns/variables in 1st and 2nd evaluation
zsf1_nc=out_tab_zsf1.select_dtypes(include=['int','float']).columns
zsf2_nc=out_tab_zsf2.select_dtypes(include=['int','float']).columns

# Different variant naming
nc_diff1=zsf1_nc.difference(zsf2_nc)
nc_diff2=zsf2_nc.difference(zsf1_nc)
print('Different naming:')
print(' - 1: {}'.format(nc_diff1))
print(' - 2: {}'.format(nc_diff2))

# PD-DF comparision
# out_tab_comp=out_tab_zsf2[zsf2_nc.difference(nc_diff2)].compare(
#     out_tab_zsf1[zsf1_nc.difference(nc_diff1)]
#     )
out_tab_comp=out_tab_zsf2.compare(out_tab_zsf1)

#relevant Parameters index comp
zsf1_nna=out_tab_zsf1[
    out_tab_zsf1.columns.intersection(rel_params)
    ].dropna(axis=0,how='all')
zsf2_nna=out_tab_zsf2[
    out_tab_zsf2.columns.intersection(rel_params)
    ].dropna(axis=0,how='all')
print('Index count with not all NaN for relevant parameters:')
print(' - 1: {}'.format(zsf1_nna.count().min()))
print(' - 2: {}'.format(zsf2_nna.count().min()))
print('  -> Index difference: {}'.format(
    zsf2_nna.index.difference(zsf1_nna.index)
    ))

# Relative deviation between evaluatuins
out_tab_vgl=reldev(a=out_tab_zsf2[zsf2_nc],b=out_tab_zsf1[zsf1_nc])
# Description (mean, min, max,...) of relative deviation
out_tab_vgl_d=out_tab_vgl.describe().T
# Absolute minimal/maximal values above 1 % relative deviation
out_tab_vgl_dp=out_tab_vgl_d.loc[
    (out_tab_vgl_d[['min','max']].abs()>0.01).any(axis=1)
    ]

# Relevant paramaters
rel_params_int = out_tab_vgl.columns.intersection(rel_params)
out_tab_vgl_rel = out_tab_vgl[rel_params_int]
out_tab_vgl_d_rel = out_tab_vgl_d.loc[rel_params_int]
rel_params_int_dp = out_tab_vgl_dp.index.intersection(rel_params)
out_tab_vgl_rel_dp = out_tab_vgl_dp.loc[rel_params_int_dp]