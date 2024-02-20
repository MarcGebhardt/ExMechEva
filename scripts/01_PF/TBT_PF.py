# -*- coding: utf-8 -*-
"""
Three point bending test for cortical bone in project PARAFEMM.
(Material parameters of the human lumbopelvic complex.)
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
import exmecheva.Eva_TBT as emetbt

# Global settings
plt.rcParams.update({
    'figure.figsize':[16.0/2.54, 9.0/2.54], 'figure.dpi': 150,
    'font.size': 8.0,
    'lines.linewidth': 1.0, 'lines.markersize': 4.0, 'markers.fillstyle': 'none',
    'axes.grid': True, "axes.axisbelow": True
    })
log_options={'logfp':None, 'output_lvl':1, 'logopt':True, 'printopt':False}
plt_options={'tight':True, 'show':True, 
             'save':True, 's_types':["pdf"], 
             'clear':True, 'close':True}

def main():
    # Set up new DataFrames for paths
    protpaths = pd.DataFrame([],dtype='string')
    combpaths = pd.DataFrame([],dtype='string')   
    
    # Options (uncomment to use):
    ## Evaluate single measurement
    option = 'single'
    ## Evaluate series of measurements (see protocol table)
    option = 'series'
    ## Evaluate series of series
    option = 'complete'
    ## Pack all evaluations into single hdf-file (only results and evaluated measurement)
    option = 'pack'
    ## Pack all evaluations into single hdf-file with (all results, Warning: high memory requirements!)
    option = 'pack-all'
    
    #PF:
    ser='PT5'
    des='cm31a'
    
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
    # protpaths.loc['PT1','path_main'] = "F:/Messung/003-190822-Becken1-DBV/"
    # protpaths.loc['PT1','name_prot'] = "190822_Becken1_DBV_Protokoll.xlsx"
    # protpaths.loc['PT2','path_main'] = "F:/Messung/004-200515-Becken2-DBV/"
    # protpaths.loc['PT2','name_prot'] = "200515_Becken2_DBV_Protokoll_new.xlsx"
    protpaths.loc['PT3','path_main'] = "F:/Messung/005-200724_Becken3-DBV/"
    protpaths.loc['PT3','name_prot'] = "200724_Becken3-DBV_Protokoll_new.xlsx"
    protpaths.loc['PT4','path_main'] = "F:/Messung/006-200917_Becken4-DBV/"
    protpaths.loc['PT4','name_prot'] = "200917_Becken4-DBV_Protokoll_new.xlsx"
    protpaths.loc['PT5','path_main'] = "F:/Messung/007-201014_Becken5-DBV/"
    protpaths.loc['PT5','name_prot'] = "201014_Becken5-DBV_Protokoll_new.xlsx"
    protpaths.loc['PT6','path_main'] = "F:/Messung/008-201125_Becken6-DBV/"
    protpaths.loc['PT6','name_prot'] = "201125_Becken6-DBV_Protokoll_new.xlsx"
    protpaths.loc['PT7','path_main'] = "F:/Messung/009-210120_Becken7-DBV/"
    protpaths.loc['PT7','name_prot'] = "210120_Becken7-DBV_Protokoll_new.xlsx"

    ## Path extensions for all series
    protpaths.loc[:,'name_opts']    = "TBT_com_eva_opts.json"
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
    out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/240219/TBT/B3-B7_TBT-Summary"
    if option == 'pack-all': out_path+='-all'  

    # Start evaluation by selector function
    eva.selector(eva_single_func=emetbt.TBT_single, 
                 option=option, combpaths=combpaths, no_stats_fc=no_stats_fc,
                 var_suffix=var_suffix, ser=ser, des=des, out_path=out_path,
                 prot_rkws=dict(header=11, skiprows=range(12,13),index_col=0),
                 log_scopt=log_options, plt_scopt=plt_options)

if __name__ == "__main__":
    main()