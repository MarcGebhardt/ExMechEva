# -*- coding: utf-8 -*-
"""
Contains functionality for loading and saving files.

@author: MarcGebhardt
"""
import os
import numpy as np
import pandas as pd

def file_namer(fstr, svars='>', svare='<', sform='#'):
    """
    Names a file according given string (sub module of File_namer_interpreter).

    Parameters
    ----------
    fstr : string
        String with replaceable tokens.
    svars : string, optional
        Start character for token. The default is '>'.
    svare : string, optional
        End character for token. The default is '<'.
    sform : string, optional
        Form determinator at end of token. The default is '#'.

    Returns
    -------
    vardf : TYPE
        DESCRIPTION.
        
    Examples
    --------
    - >Number#02<>Variant<.xlsx:
        with Number = 05 (from protocol) and Variant = C (from call) 
        -> 05C.xlsx
    - >Designation<_>Variant<_foo.csv:
        with (from protocol) Designation = Test_1 and Variant = B 
        -> Test_1_B_foo.csv

    """
    if '.' in fstr:
        fstr,fdend=fstr.split('.')
        fdend='.'+fdend
    else:
        fdend=''
    vardf=pd.DataFrame([],columns=['Value','IsVar','HasForm','Form','IsEnd'],dtype='O')
    j=0
    if svars in fstr:
        fstr = fstr.split(svars)
        for fsub in fstr:
            value=fsub
            isvar=False
            hasform=False
            form=''
            if fsub.endswith(svare):
                isvar=True
                if '#' in fsub:
                    hasform=True
                    value,form=fsub.split('#')
                    form=form.replace(svare,'')
                else:
                    value=fsub.replace(svare,'')
            vardf.loc[j]=pd.Series({'Value':value,'IsVar':isvar,
                                    'HasForm':hasform,'Form':form,'IsEnd':False})
            j+=1
    else:
        vardf.loc[j]=pd.Series({'Value':fstr,'IsVar':False,
                                'HasForm':False,'Form':'','IsEnd':False})
    if not fdend == '':
        vardf.loc[j+1]=pd.Series({'Value':fdend,'IsVar':False,
                                  'HasForm':False,'Form':'','IsEnd':True})
    return vardf

def file_namer_interpreter(fstr, prot_ser, path, variant='',
                           expext='.xlsx', svars='>', svare='<', sform='#'):
    """
    Builds a file location according given string (with repleacable tokens).

    Parameters
    ----------
    fstr : string
        String with replaceable tokens.
    prot_ser : pd.Series
        Series form protocoll dataframe with tokens as index.
    path : string
        Path addendum (will be added before adjusted filename).
    variant : string, optional
        Variant (special string for token 'Variant'). The default is ''.
    expext : string, optional
        Expected extension of file. The default is '.xlsx'.
    svars : string, optional
        Start character for token. The default is '>'.
    svare : string, optional
        End character for token. The default is '<'.
    sform : string, optional
        Form determinator at end of token. The default is '#'.

    Raises
    ------
    ValueError
        Token not found in protocoll.

    Returns
    -------
    fname : string
        Adjusted file name.
        
    Examples
    --------
    - >Number#02<>Variant<.xlsx:
        with Number = 05 (from protocol) and Variant = C (from call) 
        -> 05C.xlsx
    - >Designation<_>Variant<_foo.csv:
        with (from protocol) Designation = Test_1 and Variant = B 
        -> Test_1_B_foo.csv
    """
    var_df=file_namer(fstr=fstr,svars=svars,svare=svare,sform=sform)
    fname=path
    for i in var_df.index:
        if var_df.loc[i,'IsEnd']: 
            fdend=var_df.loc[i,'Value']
        else:
            fdend=expext
        if var_df.loc[i,'IsVar']:
            if var_df.loc[i,'Value'] in prot_ser.index:
                val=prot_ser[var_df.loc[i,'Value']]
                if var_df.loc[i,'HasForm']:
                    fname+=("{:%s}"%var_df.loc[i,'Form']).format(val)
                else:
                    fname+=str(val)
            elif var_df.loc[i,'Value'] == 'Variant':
                val=variant
                if var_df.loc[i,'HasForm']:
                    fname+=("{:%s}"%var_df.loc[i,'Form']).format(val)
                else:
                    fname+=str(val)                
            else:
                raise ValueError("%s not in protocol!"%var_df.loc[i,'Value'])
        elif not var_df.loc[i,'IsEnd']:
            fname+=var_df.loc[i,'Value']
    fname+=fdend
    return fname

#%%% HDF
def pack_hdf(in_paths, out_path, 
             hdf_naming = 'Designation', var_suffix = [""],
             h5_conc = 'Material_Parameters', h5_data = 'Measurement',
             opt_pd_out = True, opt_hdf_save = True):
    
    dfc=pd.DataFrame([],dtype='float64')
    dfd=pd.Series([],dtype='object')
    for p in in_paths.index:
        prot=pd.read_excel(in_paths.loc[p,'prot'],
                           header=11, skiprows=range(12,13),
                           index_col=0)
        for ms in var_suffix:
            for i in prot[hdf_naming].loc[np.invert(prot[hdf_naming].isna())]:
                if os.path.exists(in_paths.loc[p,'hdf']+i+ms+'.h5'):
                    data_read = pd.HDFStore(in_paths.loc[p,'hdf']+i+ms+'.h5','r')
                    df1 = data_read.select(h5_conc)
                    df2 = data_read.select(h5_data)
                    data_read.close()
                    dfp=prot.loc[[prot.loc[prot[hdf_naming]==i].index[0]]]
                    dfp.index=dfp.index+ms
                    df1=pd.concat([dfp,df1],axis=1)
                    dfc = dfc.append(df1)
                    # dfd = dfd.append(df2)
                    dfd[prot.loc[prot[hdf_naming]==i].index[0]+ms] = df2
                    del df1,df2
                else: # added 211028
                    dfp=prot.loc[[prot.loc[prot[hdf_naming]==i].index[0]]]
                    dfp.index=dfp.index+ms
                    dfc = dfc.append(dfp)
                    
            
    if opt_hdf_save:
        HDFst = pd.HDFStore(out_path+'.h5')
        HDFst['Summary'] = dfc
        HDFst['Test_Data'] = dfd
        HDFst.close()
    if opt_pd_out:
        return dfc, dfd
    
def pack_hdf_mul(in_paths, out_path, 
                 hdf_naming = 'Designation', var_suffix = [""],
                 h5_conc = 'Material_Parameters', h5_data = 'all',
                 opt_pd_out = True, opt_hdf_save = True):
    
    dfc=pd.DataFrame([],dtype='float64')
    dfd=pd.Series([],dtype='object')
    # dfd={}
    for p in in_paths.index:
        prot=pd.read_excel(in_paths.loc[p,'prot'],
                           header=11, skiprows=range(12,13),
                           index_col=0)
        for ms in var_suffix:
            for i in prot[hdf_naming].loc[np.invert(prot[hdf_naming].isna())]:
                if os.path.exists(in_paths.loc[p,'hdf']+i+ms+'.h5'):
                    data_read = pd.HDFStore(in_paths.loc[p,'hdf']+i+ms+'.h5','r')
                    df1 = data_read.select(h5_conc)
                    didks=data_read.keys()
                    for k in didks:
                        df2 = data_read.select(k)
                        if k not in dfd.keys():
                            dfd[k]=pd.Series([],dtype='object')
                        dfd[k][prot.loc[prot[hdf_naming]==i].index[0]+ms]=df2
                        del df2
                    data_read.close()
                    dfp=prot.loc[[prot.loc[prot[hdf_naming]==i].index[0]]]
                    dfp.index=dfp.index+ms
                    df1=pd.concat([dfp,df1],axis=1)
                    dfc = dfc.append(df1)
                    # dfd = dfd.append(df2)
                    # dfd[prot.loc[prot[hdf_naming]==i].index[0]+ms] = df2
                    del df1
                else: # added 211028
                    dfp=prot.loc[[prot.loc[prot[hdf_naming]==i].index[0]]]
                    dfp.index=dfp.index+ms
                    dfc = dfc.append(dfp)
    if opt_hdf_save:
        HDFst = pd.HDFStore(out_path+'.h5','w')
        HDFst['Summary'] = dfc
        # HDFst['Data'] = dfd
        for k in dfd.keys():
            HDFst['Add_'+k] = dfd[k]
        HDFst.close()
    if opt_pd_out:
        return dfc, dfd