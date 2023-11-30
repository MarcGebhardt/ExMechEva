# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:21:05 2023

@author: mgebhard
"""
import numpy as np
import pandas as pd
import json


def check_empty(x):
    """
    Tests if given variable (string or float) can interpreted as empty.

    Parameters
    ----------
    x : string or float
        DESCRIPTION.

    Returns
    -------
    t : bool
        Test result.

    """
    t = (x == '' or (isinstance(x, float) and  np.isnan(x)))
    return t

def com_option_file_read(file, check_OPT=True):
    """
    Reads common options file with json and return as pandas dataframe.

    Parameters
    ----------
    file : path object or string
        Common options file location.
    check_OPT : bool, optional
        Check if variable name starts with 'OPT_'. 
        If it is true, then use for options dataframe.
        The default is True.

    Returns
    -------
    co : pd.DataFrame
        Common options dataframe.

    """
    with open(file, 'r') as f:
        co_js = json.load(f)
    co = pd.DataFrame(co_js)
    if check_OPT:
        co_io = co.index.str.startswith('OPT_')
        co = co.loc[co_io]
    return co

def com_option_file_write(file, co, indent=1):
    """
    Writes coomon options dataframe to file with json.

    Parameters
    ----------
    file : path object or string
        Common options file location.
    co : pd.DataFrame
        Common options dataframe.
    indent : int, optional
        Indent for json builder. The default is 1.

    Returns
    -------
    None.

    """
    co_d = co.to_dict()
    with open(file, 'w') as f:
        json.dump(co_d, f, indent=indent)

def set_type_by_string(o,t):
    """
    Create new object by given determiner.

    Parameters
    ----------
    o : object
        Object to convert.
    t : string
        Determiner for conversion. Implemented are:
            - "String": String object
            - "Int": Integer object
            - "Bool": Boolean object
            - "Float": Float precision object
            - "Json": Json object (uses json.loads)
            - "Free": No conversion

    Raises
    ------
    NotImplementedError
        Given determiner/type is not implemented.

    Returns
    -------
    o : object
        Converted object.

    """
    if t=="String":
        o=str(o)
    elif t=="Int":
        o=int(o)
    elif t=="Bool":
        o=bool(o)
    elif t=="Float":
        o=float(o)
    elif t=="Json":
        o=json.loads(o)
    elif t=="Free":
        o=o
    else:
        raise NotImplementedError("Type %s not implemented!"%t)
    return o
    
def Option_presetter(opt,mtype,preset,stype=None):
    """
    Presets option by value and types.

    Parameters
    ----------
    opt : object
        Option value.
    mtype : string
        Main type (see set_type_by_string for available types and additional "Array" with stype).
    preset : object
        Preset value.
    stype : string or None, optional
        Sub types, used if mtype == "Array" (see set_type_by_string for available types). 
        The default is None.

    Returns
    -------
    opt : object
        Presetted option value.

    """
    if check_empty(opt):
       opt=preset
    else:
        if mtype == "Array":
            j=0
            opt=str(opt).replace('"','').split(',')
            for s in stype:
                opt[j]=set_type_by_string(opt[j],s)
                j+=1
        else:
            opt=set_type_by_string(opt,mtype)
    return opt

def Option_reader(options, com_opt_df, com_opts=None):
    all_inds=options.index.join(com_opt_df.index, how='outer')
    if not com_opts is None: all_inds=all_inds.join(com_opts.index, how='outer')
    # for o in options.index:
    for o in all_inds:
        if not o in options.index:
            options[o]=np.nan
        o_pres=Option_presetter(opt=options[o],
                                mtype=com_opt_df.loc[o,'mtype'],
                                preset=com_opt_df.loc[o,'preset'],
                                stype=com_opt_df.loc[o,'stype'])
        if not com_opts is None:
            if (o in com_opts.index) and check_empty(options[o]):
                 options[o]=com_opts[o]
            else:
                options[o]=o_pres
        else:
            options[o]=o_pres
    return options

def Option_reader_sel(prot_ser, paths, variant='',
                      option='JFile+Prot',
                      sheet_name="Eva_Options",
                      re_rkws=dict(header=3, skiprows=range(4,5), index_col=0)):
    if option in ['JFile+Prot', 'JFile+Sheet']:
        com_opt_df=com_option_file_read(file=paths['opts'], check_OPT=True)
        
    if option == 'JFile+Prot':
            opts = prot_ser[prot_ser.index.str.startswith('OPT_')]
            opts_out = Option_reader(options=opts, com_opt_df=com_opt_df)
            
    elif option == 'JFile+Sheet': 
        # pandas SettingWithCopyWarning
        prot_opts = pd.read_excel(paths['prot'], sheet_name=sheet_name,**re_rkws)
        if "COMMON" in prot_opts.index:
            com_opts=Option_reader(prot_opts.loc["COMMON"], com_opt_df=com_opt_df)
        else:
            com_opts=None
        search_names=[str(prot_ser.Number),
                      str(prot_ser.Designation),
                      str(prot_ser.name),
                      str(prot_ser.Number)+variant,
                      str(prot_ser.Designation)+variant,
                      str(prot_ser.name)+variant]
        for s in search_names:
            if s in prot_opts.index:
                opts=prot_opts.loc[s]
            else:
                opts=pd.Series(['',]*len(prot_opts.columns),
                                index=prot_opts.columns, dtype='O')
            com_opts=Option_reader(options=opts, com_opt_df=com_opt_df,
                                   com_opts=com_opts)
        opts_out=com_opts
    return opts_out

def File_namer(fstr, svars='>', svare='<', sform='#'):
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

def File_namer_interpreter(fstr, prot_ser, path, variant='',
                           expext='.xlsx', svars='>', svare='<', sform='#'):
    var_df=File_namer(fstr=fstr,svars=svars,svare=svare,sform=sform)
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