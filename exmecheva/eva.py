# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:23:31 2023

@author: MarcGebhardt
ToDo:
    - series only work with output_lvl>=1
"""
import os
import traceback
import pandas as pd

import exmecheva.Eva_common as Evac #import Eva_common relative?

def series(eva_single_func, paths, no_stats_fc, var_suffix, 
           prot_rkws, output_lvl=1):
    """
    

    Parameters
    ----------
    eva_single_func : TYPE
        DESCRIPTION.
    paths : TYPE
        DESCRIPTION.
    no_stats_fc : TYPE
        DESCRIPTION.
    var_suffix : TYPE
        DESCRIPTION.
    prot_rkws : TYPE
        DESCRIPTION.
    output_lvl : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    prot = pd.read_excel(paths['prot'], **prot_rkws)
    
    logfp = paths['out'] + os.path.basename(paths['prot']).replace('.xlsx','.log')
    if output_lvl>=1: log_mg=open(logfp,'w')
        
    prot.Failure_code  = Evac.list_cell_compiler(prot.Failure_code)
    eva_b = Evac.list_interpreter(prot.Failure_code, no_stats_fc)
    
    Evac.MG_strlog("\n paths:",log_mg,output_lvl,printopt=False)
    for path in paths.index:
        Evac.MG_strlog("\n  %s: %s"%(path,paths[path]),
                       log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n evaluation: %d / %d"%(prot.loc[eva_b].count()[0],prot.count()[0]),
                            log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n%s"%prot.loc[eva_b].Designation.values,
                            log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n not evaluated: %d / %d"%(prot.loc[eva_b==False].count()[0],prot.count()[0]),
                            log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n%s"%prot.loc[eva_b==False].Designation.values,
                            log_mg,output_lvl,printopt=False)

    for eva in prot[eva_b].index:
        for mfile_add in var_suffix:
            Evac.MG_strlog("\n %s"%prot.loc[eva].Designation+mfile_add,
                            log_mg,output_lvl,printopt=False)  
            try:
                cout = eva_single_func(prot_ser = prot.loc[eva],
                                     paths = paths, mfile_add=mfile_add)
                Evac.MG_strlog("\n   Eva_time: %.5f s (Control: %s)"%(cout[0].iloc[-1]-cout[0].iloc[0],cout[1]),
                                log_mg,output_lvl,printopt=False)
            except Exception:
                txt = '\n   Exception:'
                txt+=Evac.str_indent('\n{}'.format(traceback.format_exc()),5)
                Evac.MG_strlog(txt, log_mg,output_lvl,printopt=False)  

    if output_lvl>=1: log_mg.close()

def selector(eva_single_func, option, combpaths, no_stats_fc,
             var_suffix=[""], ser='', des='', out_path='',
             prot_rkws=dict(header=11, skiprows=range(12,13),index_col=0)):
    """
    Selects suitable evaluation method acc. to choosen option.

    Parameters
    ----------
    eva_single_func : function
        Function to perform single evaluation.
    option : str
        Evaluation option. Possible are:
            - 'single': Evaluate single measurement
            - 'series': Evaluate series of measurements (see protocol table)
            - 'complete': Evaluate series of series
            - 'pack': Pack all evaluations into single hdf-file (only results and evaluated measurement)
            - 'pack-all': Pack all evaluations into single hdf-file with (all results, Warning: high memory requirements!)
    combpaths : pandas.DataFrame
        Combined paths for in- and output of evaluations.
    no_stats_fc : list of str
        Assessment codes for excluding from evaluation (searched in protocol variable "Failure_code").
    var_suffix : list of str, optional
        Suffix of variants of measurements (p.E. different moistures ["A","B",...]). 
        The default is [""].
    ser : str, optional
        Accessor for series. Must be as index in combpaths. The default is ''.
    des : str, optional
        Accessor for/Designation of measurement/specimen. 
        Must be as index in combpaths. The default is ''.
    out_path : str or Path, optional
        Additional outputpath for packed evaluation (hdf file). The default is ''.
    prot_rkws : dict, optional
        Dictionary for reading protocol. Must be keyword ind pandas.read_excel.
        The default is dict(header=11, skiprows=range(12,13),index_col=0).

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
        
        prot=pd.read_excel(combpaths.loc[ser,'prot'],**prot_rkws)
        _=eva_single_func(prot_ser=prot[prot.Designation==des].iloc[0], 
                          paths=combpaths.loc[ser],
                          mfile_add = mfile_add)
        
    elif option == 'series':
        series(eva_single_func=eva_single_func,
               paths = combpaths.loc[ser],
               no_stats_fc = no_stats_fc,
               var_suffix = var_suffix,
               prot_rkws=prot_rkws)
        
    elif option == 'complete':
        for ser in combpaths.index:
            series(eva_single_func,
                   paths = combpaths.loc[ser],
                   no_stats_fc = no_stats_fc,
                   var_suffix = var_suffix,
                   prot_rkws=prot_rkws)
            
    elif option == 'pack':
        packpaths = combpaths[['prot','out']]
        packpaths.columns=packpaths.columns.str.replace('out','hdf')
        Evac.pack_hdf(in_paths=packpaths, out_path = out_path,
                      hdf_naming = 'Designation', var_suffix = var_suffix,
                      h5_conc = 'Material_Parameters', h5_data = 'Measurement',
                      opt_pd_out = False, opt_hdf_save = True)
        print("Successfully created %s"%out_path)
    elif option == 'pack-all':
        packpaths = combpaths[['prot','out']]
        packpaths.columns=packpaths.columns.str.replace('out','hdf')
        Evac.pack_hdf_mul(in_paths=packpaths, out_path = out_path,
                          hdf_naming = 'Designation', var_suffix = var_suffix,
                          h5_conc = 'Material_Parameters',
                          opt_pd_out = False, opt_hdf_save = True)
        print("Successfully created %s"%out_path)
        
    else:
        raise NotImplementedError('Option %s not implemented!'%option)