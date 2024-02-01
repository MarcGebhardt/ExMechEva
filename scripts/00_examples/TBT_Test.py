# -*- coding: utf-8 -*-
"""
Three point bending test for cortical bone.
Simple example using data/TBT for showing evaluation.
"""

import os 
# import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# sys.path.insert(-1,'D:\Gebhardt\Programme\DEV\Git\ExMechEva')

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
    ## Evaluate series of measurements (see protocol table, here only one named 'cl12a')
    option = 'series'
    ## Evaluate series of series (here only one series, named 'TS')
    option = 'complete'
    ## Pack all evaluations into single hdf-file (only results and evaluated measurement)
    option = 'pack'
    ## Pack all evaluations into single hdf-file with (all results, Warning: high memory requirements!)
    option = 'pack-all'

    # Example (Series='TS' and specimen designation='tl21x', see protocol table): 
    ser='TS'
    des='cl12a'
    
    # No evaluation for list of Assessment Codes
    no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
                   'B01.1','B01.2','B01.3', 'B02.3',
                   'C01.1','C01.2','C01.3', 'C02.3',
                   'D01.1','D01.2','D01.3', 'D02.3',
                   'F01.1','F01.2','F01.3', 'F02.3']
    # Suffix of variants of measurements (p.E. different moistures ["A","B",...])
    var_suffix = [""]
    
    # Path selection
    ## Main path from current working directory
    main_path = str(nwd)+"\\data\\Test\\TBT\\" 
    ## Series dependend paths or file names
    protpaths.loc['TS','path_main'] = main_path+"Series_Test\\"
    protpaths.loc['TS','name_prot'] = "TBT_Protocol_Series_Test.xlsx"
    ## Path extensions for all series
    protpaths.loc[:,'name_opts']    = "com_eva_opts.json"
    protpaths.loc[:,'path_con']     = "meas\\conventional\\"
    protpaths.loc[:,'path_dic']     = "meas\\optical\\"
    protpaths.loc[:,'path_eva1']    = "eva\\"
    
    # Path builder 
    combpaths['prot'] = protpaths['path_main']+protpaths['name_prot']
    combpaths['opts'] = protpaths['path_main']+protpaths['name_opts']
    combpaths['meas'] = protpaths['path_main']+protpaths['path_con']
    combpaths['dic']  = protpaths['path_main']+protpaths['path_dic']
    combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']
    
    # Additional path for hdf-packing
    out_path=main_path+"Complete_Eva\\TBT-Summary" 
    if option == 'pack-all': out_path+='-all'  

    # Start evaluation by selector function
    eva.selector(eva_single_func=emetbt.TBT_single, 
                 option=option, combpaths=combpaths, no_stats_fc=no_stats_fc,
                 var_suffix=var_suffix, ser=ser, des=des, out_path=out_path,
                 prot_rkws=dict(header=11, skiprows=range(12,13),index_col=0),
                 log_scopt=log_options, plt_scopt=plt_options)

if __name__ == "__main__":
    main()