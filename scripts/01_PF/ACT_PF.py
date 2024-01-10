# -*- coding: utf-8 -*-
"""
Axial compression test for cancellous bone in project PARAFEMM.
"""

import os 
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(-1,'D:\Gebhardt\Programme\DEV\Git\ExMechEva')

nwd = Path.cwd().resolve().parent.parent
os.chdir(nwd)

import exmecheva.eva as eva
import exmecheva.Eva_ACT as emeact

# Global settings
plt.rcParams.update({
    'figure.figsize':[16.0/2.54, 9.0/2.54], 'figure.dpi': 150,
    'font.size': 8.0,
    'lines.linewidth': 1.0, 'lines.markersize': 4.0, 'markers.fillstyle': 'none',
    'axes.grid': True, "axes.axisbelow": True
    })

def main():
    # Set up new DataFrames for paths
    protpaths = pd.DataFrame([],dtype='string')
    combpaths = pd.DataFrame([],dtype='string')   
    
    # Options (uncomment to use):
    ## Evaluate single measurement
    option = 'single'
    # ## Evaluate series of measurements (see protocol table)
    # option = 'series'
    # ## Evaluate series of series
    # option = 'complete'
    # ## Pack all evaluations into single hdf-file (only results and evaluated measurement)
    # option = 'pack'
    # ## Pack all evaluations into single hdf-file with (all results, Warning: high memory requirements!)
    # option = 'pack-all'
    
    #PF:
    ser='B7'
    des='tm41x'
    
    # No Evaluation for list of Assessment Codes
    no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
                   'B01.1','B01.2','B01.3', 'B02.3',
                   'C01.1','C01.2','C01.3', 'C02.3',
                   'D01.1','D01.2','D01.3', 'D02.3',
                   'F01.1','F01.2','F01.3', 'F02.3']
    # Suffix of variants of measurements (p.E. diffferent moistures ["A","B",...])
    var_suffix = [""]
         
    # Path selection
    ## Series dependend paths or file names
    # protpaths.loc['B1','path_main'] = "F:/Messung/003-190822-Becken1-ZDV/"
    # protpaths.loc['B1','name_prot'] = "190822_Becken1_ZDV_Protokoll_new.xlsx"
    # protpaths.loc['B2','path_main'] = "F:/Messung/004-200514-Becken2-ADV/"
    # protpaths.loc['B2','name_prot'] = "200514_Becken2_ADV_Protokoll_new.xlsx"
    protpaths.loc['B3','path_main'] = "F:/Messung/005-200723_Becken3-ADV/"
    protpaths.loc['B3','name_prot'] = "200723_Becken3-ADV_Protokoll_new.xlsx"
    protpaths.loc['B4','path_main'] = "F:/Messung/006-200916_Becken4-ADV/"
    protpaths.loc['B4','name_prot'] = "200916_Becken4-ADV_Protokoll_new.xlsx"
    protpaths.loc['B5','path_main'] = "F:/Messung/007-201013_Becken5-ADV/"
    protpaths.loc['B5','name_prot'] = "201013_Becken5-ADV_Protokoll_new.xlsx"
    protpaths.loc['B6','path_main'] = "F:/Messung/008-201124_Becken6-ADV/"
    protpaths.loc['B6','name_prot'] = "201124_Becken6-ADV_Protokoll_new.xlsx"
    protpaths.loc['B7','path_main'] = "F:/Messung/009-210119_Becken7-ADV/"
    protpaths.loc['B7','name_prot'] = "210119_Becken7-ADV_Protokoll_new.xlsx"

    ## Path extensions for all series
    protpaths.loc[:,'name_opts']    = "ACT_com_eva_opts.json"
    protpaths.loc[:,'path_con']     = "Messdaten/Messkurven/"
    protpaths.loc[:,'path_dic']     = "Messdaten/DIC/"
    protpaths.loc[:,'path_eva1']    = "Auswertung/"
    protpaths.loc[:,'path_eva2']    = "ExMechEva/"
    
    # Path builder 
    combpaths['opts'] = "F:/Messung/Eva_Options/"+protpaths['name_opts']
    combpaths['prot'] = protpaths['path_main']+protpaths['path_eva1']+protpaths['name_prot']
    combpaths['meas'] = protpaths['path_main']+protpaths['path_con']
    combpaths['dic']  = protpaths['path_main']+protpaths['path_dic']
    combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']
        
    # Additional path for hdf-packing
    out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/XXX/ACT/B3-B7_ACT-Summary"  
    if option == 'pack-all': out_path+='-all'  

    # Start evaluation by selector function
    eva.selector(eva_single_func=emeact.ACT_single, 
                 option=option, combpaths=combpaths, no_stats_fc=no_stats_fc,
                 var_suffix=var_suffix, ser=ser, des=des, out_path=out_path,
                 prot_rkws=dict(header=11, skiprows=range(12,13),index_col=0))

if __name__ == "__main__":
    main()