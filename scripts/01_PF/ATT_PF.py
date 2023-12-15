# -*- coding: utf-8 -*-
"""
Axial tensile test for soft tissues in project PARAFEMM.
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
import exmecheva.Eva_ATT as emeatt

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
    ser='B6'
    des='sr13'
    
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
    # protpaths.loc['B1','path_main'] = "F:/Messung/003-190821-Becken1-ZZV/"
    # protpaths.loc['B1','name_prot'] = "190821_Becken1_ZZV_Protokoll_new.xlsx"
    # protpaths.loc['B2','path_main'] = "F:/Messung/004-200512-Becken2-AZV/"
    # protpaths.loc['B2','name_prot'] = "200512_Becken2_AZV_Protokoll_new.xlsx"
    protpaths.loc['B3','path_main'] = "F:/Messung/005-200721_Becken3-AZV/"
    protpaths.loc['B3','name_prot'] = "200721_Becken3-AZV_Protokoll_new.xlsx"
    protpaths.loc['B4','path_main'] = "F:/Messung/006-200910_Becken4-AZV/"
    protpaths.loc['B4','name_prot'] = "200910_Becken4-AZV_Protokoll_new.xlsx"
    protpaths.loc['B5','path_main'] = "F:/Messung/007-201009_Becken5-AZV/"
    protpaths.loc['B5','name_prot'] = "201009_Becken5-AZV_Protokoll_new.xlsx"
    protpaths.loc['B6','path_main'] = "F:/Messung/008-201117_Becken6-AZV/"
    protpaths.loc['B6','name_prot'] = "201117_Becken6-AZV_Protokoll_new.xlsx"
    protpaths.loc['B7','path_main'] = "F:/Messung/009-210114_Becken7-AZV/"
    protpaths.loc['B7','name_prot'] = "210114_Becken7-AZV_Protokoll_new.xlsx"

    ## Path extensions for all series
    protpaths.loc[:,'path_con']     = "Messdaten/Messkurven/"
    protpaths.loc[:,'path_dic']     = "Messdaten/DIC/"
    protpaths.loc[:,'path_eva1']    = "Auswertung/"
    protpaths.loc[:,'path_eva2']    = "ExMechEva/"
    
    # Path builder 
    combpaths['prot'] = protpaths['path_main']+protpaths['path_eva1']+protpaths['name_prot']
    combpaths['meas'] = protpaths['path_main']+protpaths['path_con']
    combpaths['dic']  = protpaths['path_main']+protpaths['path_dic']
    combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']
        
    # Additional path for hdf-packing 
    out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/XXX/ATT/B3-B7_ATT-Summary"
    if option == 'pack-all': out_path+='-all'  

    # Start evaluation by selector function
    eva.selector(eva_single_func=emeatt.ATT_single, 
                 option=option, combpaths=combpaths, no_stats_fc=no_stats_fc,
                 var_suffix=var_suffix, ser=ser, des=des, out_path=out_path,
                 prot_rkws=dict(header=11, skiprows=range(12,13),index_col=0))

if __name__ == "__main__":
    main()