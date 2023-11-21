# -*- coding: utf-8 -*-
"""
Axial compression test for cancellous bone.
Simple example using data/ACT for showing evaluation acc. to "bla".
"""

import os 
from pathlib import Path
nwd = Path.cwd().resolve().parent.parent
os.chdir(nwd)

import pandas as pd
import exmecheva.Eva_common as emec
import exmecheva.Eva_ACT as emeact

def ACT_selector(option, combpaths, no_stats_fc,
                 var_suffix=[""], ser='', des='', out_path=''):
    """
    Selects suitable evaluation method acc. to choosen option.

    Parameters
    ----------
    option : str
        DESCRIPTION.
    combpaths : pandas.DataFrame
        DESCRIPTION.
    no_stats_fc : list of str
        DESCRIPTION.
    var_suffix : list of str, optional
        DESCRIPTION. The default is [""].
    ser : str, optional
        DESCRIPTION. The default is ''.
    des : str, optional
        DESCRIPTION. The default is ''.
    out_path : str or Path, optional
        DESCRIPTION. The default is ''.

    Raises
    ------
    NotImplementedError
        Option not implemented.

    Returns
    -------
    None.

    """
    if option == 'single':
        mfile_add = var_suffix[0]
        
        prot=pd.read_excel(combpaths.loc[ser,'prot'],
                           header=11, skiprows=range(12,13),
                           index_col=0)
        _=emeact.ACT_single(prot_ser=prot[prot.Designation==des].iloc[0], 
                     paths=combpaths.loc[ser],
                     mfile_add = mfile_add)
        
    elif option == 'series':
        emeact.ACT_series(paths = combpaths.loc[ser],
                   no_stats_fc = no_stats_fc,
                   var_suffix = var_suffix)
        
    elif option == 'complete':
        for ser in combpaths.index:
            emeact.ACT_series(paths = combpaths.loc[ser],
                       no_stats_fc = no_stats_fc,
                       var_suffix = var_suffix)
            
    elif option == 'pack-complete':
        packpaths = combpaths[['prot','out']]
        packpaths.columns=packpaths.columns.str.replace('out','hdf')
        emec.pack_hdf(in_paths=packpaths, out_path = out_path,
                      hdf_naming = 'Designation', var_suffix = var_suffix,
                      h5_conc = 'Material_Parameters', h5_data = 'Measurement',
                      opt_pd_out = False, opt_hdf_save = True)
        print("Successfully created %s"%out_path)
        
    else:
        raise NotImplementedError('%s not implemented!'%option)    

def main():
    # Options (uncomment to use):
    # Evaluate single measurement
    option = 'single'
    # Evaluate series of measurements (see protocol table, here only one named 'tl21x')
    option = 'series'
    # Evaluate series of series (here only one series, named 'TS')
    option = 'complete'
    # Pack all evaluations into single hdf-file
    option = 'pack-complete'
    
    # Example (Series='TS' and specimen designation='tl21x', see protocol table):    
    ser='TS'
    des='tl21x'
    
    # No Evaluation for list of Assessment Codes
    no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
                   'B01.1','B01.2','B01.3', 'B02.3',
                   'C01.1','C01.2','C01.3', 'C02.3',
                   'D01.1','D01.2','D01.3', 'D02.3',
                   'F01.1','F01.2','F01.3', 'F02.3']
    # Suffix of variants of measurements (p.E. diffferent moistures ["A","B",...])
    var_suffix = [""]
    
    # Path selection
    main_path = str(nwd)+"\\data\\Test\\ACT\\"
    protpaths = pd.DataFrame([],dtype='string')
    combpaths = pd.DataFrame([],dtype='string')    
    protpaths.loc['TS','path_main'] = main_path+"Series_Test\\"
    protpaths.loc['TS','name_prot'] = "ACT_Protocol_Series_Test.xlsx"
    
    protpaths.loc[:,'path_con']     = "meas\\conventional\\"
    protpaths.loc[:,'path_dic']     = "meas\\optical\\"
    protpaths.loc[:,'path_eva1']    = "eva\\"
    
    combpaths['prot'] = protpaths['path_main']+protpaths['name_prot']
    combpaths['meas'] = protpaths['path_main']+protpaths['path_con']
    combpaths['dic']  = protpaths['path_main']+protpaths['path_dic']
    combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']
        
    out_path=main_path+"Complete_Eva\\ACT-Summary" 
    
    # #PF:
    # ser='B7'
    # des='tm41x'
    
    # no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
    #                'B01.1','B01.2','B01.3', 'B02.3',
    #                'C01.1','C01.2','C01.3', 'C02.3',
    #                'D01.1','D01.2','D01.3', 'D02.3',
    #                'F01.1','F01.2','F01.3', 'F02.3']
    # var_suffix = [""] #Suffix of variants of measurements (p.E. diffferent moistures)
        
    # protpaths = pd.DataFrame([],dtype='string')
    # combpaths = pd.DataFrame([],dtype='string')    
    # # protpaths.loc['B1','path_main'] = "F:/Messung/003-190822-Becken1-ZDV/"
    # # protpaths.loc['B1','name_prot'] = "190822_Becken1_ZDV_Protokoll_new.xlsx"
    # # protpaths.loc['B2','path_main'] = "F:/Messung/004-200514-Becken2-ADV/"
    # # protpaths.loc['B2','name_prot'] = "200514_Becken2_ADV_Protokoll_new.xlsx"
    # protpaths.loc['B3','path_main'] = "F:/Messung/005-200723_Becken3-ADV/"
    # protpaths.loc['B3','name_prot'] = "200723_Becken3-ADV_Protokoll_new.xlsx"
    # protpaths.loc['B4','path_main'] = "F:/Messung/006-200916_Becken4-ADV/"
    # protpaths.loc['B4','name_prot'] = "200916_Becken4-ADV_Protokoll_new.xlsx"
    # protpaths.loc['B5','path_main'] = "F:/Messung/007-201013_Becken5-ADV/"
    # protpaths.loc['B5','name_prot'] = "201013_Becken5-ADV_Protokoll_new.xlsx"
    # protpaths.loc['B6','path_main'] = "F:/Messung/008-201124_Becken6-ADV/"
    # protpaths.loc['B6','name_prot'] = "201124_Becken6-ADV_Protokoll_new.xlsx"
    # protpaths.loc['B7','path_main'] = "F:/Messung/009-210119_Becken7-ADV/"
    # protpaths.loc['B7','name_prot'] = "210119_Becken7-ADV_Protokoll_new.xlsx"

    # protpaths.loc[:,'path_con']     = "Messdaten/Messkurven/"
    # protpaths.loc[:,'path_dic']     = "Messdaten/DIC/"
    # protpaths.loc[:,'path_eva1']    = "Auswertung/"
    # protpaths.loc[:,'path_eva2']    = "ExMechEva/"
    
    # combpaths['prot'] = protpaths['path_main']+protpaths['path_eva1']+protpaths['name_prot']
    # combpaths['meas'] = protpaths['path_main']+protpaths['path_con']
    # combpaths['dic']  = protpaths['path_main']+protpaths['path_dic']
    # combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']
        
    # out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/231023/ACT/B3-B7_ACT-Summary"    

    ACT_selector(option=option, combpaths=combpaths, no_stats_fc=no_stats_fc,
                 var_suffix=var_suffix, ser=ser, des=des, out_path=out_path)

if __name__ == "__main__":
    main()