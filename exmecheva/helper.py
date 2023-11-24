# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:21:05 2023

@author: mgebhard
"""
import numpy as np
import pandas as pd
import json

def com_option_file_read(file, check_OPT=True):
    with open(file, 'r') as f:
        co_js = json.load(f)
    co = pd.DataFrame(co_js)
    if check_OPT:
        co_io = co.index.str.startswith('OPT_')
        co = co.loc[co_io]
    return co

def com_option_file_write(file, co, indent=1):
    co_d = co.to_dict()
    with open(file, 'w') as f:
        json.dump(co_d, f, indent=indent)


def check_empty(x):
    t = (x == '' or (isinstance(x, float) and  np.isnan(x)))
    return t

def Option_presetter(opt,mtype,preset,stype=None):
    def typer(o,t):
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
    if check_empty(opt):
       opt=preset
    else:
        if mtype == "Array":
            j=0
            opt=str(opt).replace('"','').split(',')
            for s in stype:
                opt[j]=typer(opt[j],s)
                j+=1
        else:
            opt=typer(opt,mtype)
    return opt

def TBT_Option_reader(options, com_opts=None):
    opt_df = pd.DataFrame({'OPT_Testtype':["String",'preload elastic',None],
                          'OPT_File_Meas':["String",'Test',None],
                          'OPT_File_DIC':["String",'Test',None],
                          'OPT_Measurement_file':["Json",'{"header":48,"head_names":["Time","Trigger","F_WZ","L_PM","F_PM","L_IWA1","L_IWA2"], "used_names_dict":{"Time":"Time","Force":"F_WZ","Way":["L_IWA1","L_IWA2"]}}',None],
                          'OPT_Start':["Float",np.nan,None],
                          'OPT_End':["Float",np.nan,None],
                          'OPT_Resampling':["Bool",True,None],
                          'OPT_Resampling_moveave':["Bool",True,None],
                          'OPT_Resampling_Frequency':["Float",5.0,None],
                          'OPT_Springreduction':["Bool",False,None],
                          'OPT_Springreduction_K':["Float",-0.116,None],
                          'OPT_LVDT_failure':["Bool",False,None],
                          'OPT_Compression':["Bool",True,None],
                          'OPT_DIC':["Bool",True,None],
                          'OPT_TimeOS':["String","wo_1st+last",None],
                          'OPT_DIC_Points_device_prefix':["String",'S',None],
                          'OPT_DIC_Points_meas_prefix':["String",'P',None],
                          'OPT_DIC_Points_TBT_device':["Array",['S1','S2','S3'],['String','String','String']],
                          'OPT_DIC_Points_meas_fork':["Array",['P4','P5','P6'],['String','String','String']],
                          'OPT_DIC_Fitting':["Json",'{"Bending_Legion_Builder":"FSE_fixed","Bending_Legion_Name":"FSE fit","Bending_MCFit_opts":{"error_weights_pre":[   1,   0,1000, 100],"error_weights_bend":[   1,  10,1000, 100],"error_weights_pre_inc":[   1,   0,1000, 100],"error_weights_bend_inc":[   1,  10,1000, 100],"fit_max_nfev_pre": 500,"fit_max_nfev_bend": 500,"fit_max_nfev_pre_inc": 500,"fit_max_nfev_bend_inc": 500},"pwargs":{"param_val": {"xmin":-10,"xmax":10,"FP":"pi/(xmax-xmin)","b1":-1.0,"b2":-0.1,"b3":-0.01,"b4":-0.001,"c":0.0,"d":0.0},"param_min": {},"param_max": {"b1":1.0,"b2":1.0,"b3":1.0,"b4":1.0}}}',None],
                          'OPT_DIC_Tester':["Array",[0.005, 50.0, 3, 8],['Float','Float','Int','Int']],
                          'OPT_Poisson_prediction':["Float",0.30,None],
                          'OPT_Determination_Distance':["Int",30,None],
                          'OPT_YM_Determination_range':["Array",[0.25,0.50,'P','U'],['Float','Float','String','String']],
                          'OPT_YM_Determination_refinement':["Array",[[0.15,0.75,'P','U','d_M',True,8]],['Float','Float','String','String','String','Bool','Int']]},
                          index=['mtype','preset','stype']).T
    for o in options.index:
        if not com_opts is None:
            if (o in com_opts.index) and check_empty(options[o]):
                 options[o]=com_opts[o]
            else:
                options[o]=Option_presetter(opt=options[o],
                                            mtype=opt_df.loc[o,'mtype'],
                                            preset=opt_df.loc[o,'preset'],
                                            stype=opt_df.loc[o,'stype'])
        else:
            options[o]=Option_presetter(opt=options[o],
                                        mtype=opt_df.loc[o,'mtype'],
                                        preset=opt_df.loc[o,'preset'],
                                        stype=opt_df.loc[o,'stype'])
    return options

def Option_reader_Excelsheet(prot_ser, paths, variant='',
                             sheet_name="Eva_Options",
                             re_rkws=dict(header=3, skiprows=range(4,5),
                                          index_col=0)):
    # pandas SettingWithCopyWarning
    prot_opts = pd.read_excel(paths['prot'], 
                              sheet_name=sheet_name,
                              **re_rkws)
    if "COMMON" in prot_opts.index:
        opts_com=TBT_Option_reader(prot_opts.loc["COMMON"])
    else:
        opts_com=None
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
            # opts=opts_com
        opts_com=TBT_Option_reader(opts,opts_com)
    return opts_com

def File_namer(fstr,svars='>',svare='<',sform='#'):
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
                           expext='.xlsx', svars='>',svare='<',sform='#'):
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