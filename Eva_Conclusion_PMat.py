# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:54:02 2023

@author: mgebhard
"""

#%% Imports
import os
import copy
from datetime import date
import pandas as pd
idx = pd.IndexSlice
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_tick
from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import Eva_common as Evac

# plt.rcParams['figure.figsize'] = [6.3,3.54]
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['font.size']= 8.0
# #pd.set_option('display.expand_frame_repr', False)
# plt.rcParams['lines.linewidth']= 1.0
# plt.rcParams['lines.markersize']= 4.0
# plt.rcParams['markers.fillstyle']= 'none'
# plt.rcParams['axes.grid']= True

sns.set_theme(context="paper",style="whitegrid",
              font="segoe ui",palette="tab10",
              # font="minion pro",palette="tab10",
              # rc={'figure.dpi': 300.0,'figure.figsize': [16/2.54, 8/2.54],
              rc={'figure.dpi': 600.0,'figure.figsize': [16/2.54, 8/2.54],
                  'font.size':9, 'ytick.alignment': 'center',
                  'axes.titlesize':9, 'axes.titleweight': 'bold',
                  'axes.labelsize':9,
                  'xtick.labelsize': 9,'ytick.labelsize': 9,
                  'legend.title_fontsize': 9,'legend.fontsize': 9,
                  'lines.linewidth': 1.0,'markers.fillstyle': 'none'})

plt_Fig_dict={'tight':True, 'show':True, 
              'save':True, 's_types':["pdf","png"], 
              'clear':True, 'close':True}
MG_logopt={'logfp':None,'output_lvl':1,'logopt':True,'printopt':False}
# MG_logopt={'logfp':None,'output_lvl':1,'logopt':False,'printopt':True}
#%% Functions
def VIP_rebuild(ser):
    """Rebuild VIP Series by Column in Dataframe"""
    tmp=ser[~(ser=='')]
    tmp2=pd.Series(tmp.index, tmp.values, name=ser.name)
    tmp3=pd.Series([], dtype='int64', name=ser.name)
    for i in tmp2.index:
        for j in i.split(','):
            tmp3[j]=tmp2.loc[i]
    return tmp3
def Stress_Strain_fin(messu, Vipser, YM, YMabs,
                      signame='Stress', epsname='Strain',
                      lSname='F3'):
    # strain_dif=messu.loc[Vipser[lSname],signame]/YM
    out=messu.loc[Vipser[lSname]:,[signame,epsname]].copy(deep=True)
    # out[epsname]=out[epsname]-strain_dif
    out[epsname]=out[epsname]-(-YMabs/YM)
    out.loc[0]={signame:0,epsname:0}
    out.sort_index(inplace=True)
    return out

def linRegStats(Y,X,Y_txt,X_txt):
    des_txt = ("\n- Linear relation between %s and %s:"%(Y_txt,X_txt))
    tmp  = Evac.YM_sigeps_lin(Y, X)
    smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
    out=pd.Series([*tmp,smst,des_txt],index=['s','c','Rquad','fit','smstat','Description'])
    return out

def linRegStats_Multi(df, lRd, var_ren={}, var_sym={}, ind=3, addind=3):
    Lin_reg_df =pd.DataFrame(lRd, columns=['var1','var2'])
    Lin_reg_df.index=Lin_reg_df.var1+'-'+Lin_reg_df.var2
    Lin_reg_df[['var1ren','var2ren']]=Lin_reg_df[['var1','var2']].replace(var_ren)
    Lin_reg_df[['var1sym','var2sym']]=Lin_reg_df[['var1','var2']].replace(var_sym)
    tmp=Lin_reg_df.apply(lambda x: linRegStats(df[x.var1], df[x.var2],x.var1ren,x.var2ren), axis=1)
    Lin_reg_df=pd.concat([Lin_reg_df,tmp],axis=1)
    txt=Lin_reg_df.apply(lambda x: Evac.str_indent(x.loc['Description'],ind) + 
                         Evac.str_indent(Evac.fit_report_adder(*x[['fit','Rquad']]),ind+addind) + 
                         Evac.str_indent(x['smstat'],ind+addind),axis=1)
    return Lin_reg_df, txt

def func_lin_str(xl, yl, 
                 a, b, R=None,
                 t_form='{a:.3e},{b:.3e},{R:.3f}'):
    """Return string from a general linear function."""
    if R is None:
        txtc = t_form.format(**{'a':a,'b':b},).split(',')
    else:
        txtc = t_form.format(**{'a':a,'b':b,'R':R},).split(',')
    if a==0:
        txt = r'{y} = {0} {x}'.format(txtc[1],
                                           **{'y':yl,'x':xl},)
    else:
        if b>=0:
            txt = r'{y} = {0} + {1} $\cdot$ {x}'.format(txtc[0],txtc[1],
                                                   **{'y':yl,'x':xl},)
        else:
            txtc[1]=txtc[1][1:]
            txt = r'{y} = {0} - {1} $\cdot$ {x}'.format(txtc[0],txtc[1],
                                                   **{'y':yl,'x':xl},)
    if not R is None:
        txt += r' ($R²$ = {0})'.format(txtc[2])
    return txt
def linRegfunc_fromSt(Lin_reg_df, y, x, t_form='{a:.3e},{b:.3e},{R:.3f}'):
    tmp=Lin_reg_df.loc[y+'-'+x]
    txt= func_lin_str(tmp['var2sym'], tmp['var1sym'], 
                      a=tmp['s'], b=tmp['c'], R=tmp['Rquad'],
                      t_form=t_form)
    return txt

def plt_add_figsubl(text, axo=None, off=(0.02,0.02), xg=0.0, yg=1.0,
                    fontdict=dict(fontsize=9+2, fontweight="bold", fontstyle="normal",
                                  ha='left', va='bottom'),
                    bboxdict=dict(boxstyle="circle", fc="gray", alpha=0.3,
                                  ec="black", lw=1, pad=0.2)):
    if axo is None: axo=plt.gca()
    axo.annotate(text,
                 xy=(xg, yg), xycoords='axes fraction',
                 xytext = off, textcoords='offset points',
                 bbox=bboxdict,**fontdict)
    
def boxplt_dl(pdf, var, ytxt,
              xl, axl, tl, xtxtl,
              xd, axd, td, xtxtd, 
              xltirep={}, xdtirep={},
              bplkws={'showmeans':True, 'meanprops':{"marker":"_", "markerfacecolor":"white",
                                                     "markeredgecolor":"black", "markersize":"12",
                                                     "alpha":0.75}},
              splkws={'dodge':True, 'edgecolor':"black", 'linewidth':.5, 'alpha':.5, 'size':2}):
    df=pd.melt(pdf, id_vars=[xl], value_vars=[var])
    df[xl].replace(xltirep,inplace=True)
    axt = sns.boxplot(x=xl, y="value", data=df, ax=axl, **bplkws)
    axt = sns.swarmplot(x=xl, y="value",data=df, ax=axl, **splkws)
    axl.set_title(tl)
    axl.set_xlabel(xtxtl)
    axl.set_ylabel('')
    axl.tick_params(axis='y',which='both',left=False,labelleft=False)
    df=pd.melt(pdf, id_vars=[xd], value_vars=[var])
    df[xd].replace(xdtirep,inplace=True)
    axt = sns.boxplot(x=xd, y="value", data=df, ax=axd,**bplkws)
    axt = sns.swarmplot(x=xd, y="value",data=df, ax=axd, **splkws)
    axd.set_title(td)
    axd.set_xlabel(xtxtd)
    axd.set_ylabel(ytxt)
    axl.sharey(axd)
    
#%% Einlesen und auswählen
#%%% Main
Version="230920"
ptype="TBT"
ptype="ACT"
# ptype="ATT"

no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
               'B01.1','B01.2','B01.3', 'B02.3',
               'C01.1','C01.2','C01.3', 'C02.3',
               'D01.1','D01.2','D01.3', 'D02.3',
               'F01.1','F01.2','F01.3', 'F02.3',
               'G01.1','G01.2','G01.3', 'G02.3']

VIPar_plt_renamer = {'fy':'$f_{y}$','fu':'$f_{u}$','fb':'$f_{b}$',
                     'ey_con':r'$\epsilon_{y,con}$','eu_con':r'$\epsilon_{u,con}$','eb_con':r'$\epsilon_{b,con}$',
                     'Wy_con':'$W_{y,con}$','Wu_con':'$W_{u,con}$','Wb_con':'$W_{b,con}$',
                     'Uy_con':'$U_{y,con}$','Uu_con':'$U_{u,con}$','Ub_con':'$U_{b,con}$',
                     'ey_opt':r'$\epsilon_{y,opt}$','eu_opt':r'$\epsilon_{u,opt}$','eb_opt':r'$\epsilon_{b,opt}$',
                     'Wy_opt':'$W_{y,opt}$','Wu_opt':'$W_{u,opt}$','Wb_opt':'$W_{b,opt}$',
                     'Uy_opt':'$U_{y,opt}$','Uu_opt':'$U_{u,opt}$','Ub_opt':'$U_{b,opt}$',
                     'E_con': '$E_{con}$','E_opt': '$E_{opt}$',
                     'Density_app': r'$\rho_{app}$',
                     'Length_test': r'$l_{test}$',
                     'MoI_mid': r'$I_{mid}$',
                     'thickness_mean': r'$t_{mean}$', 'width_mean': r'$w_{mean}$',
                     'Area_CS': r'$A_{CS}$', 'Volume': r'$V$',
                     'Fy':'$F_{y}$','Fu':'$F_{u}$','Fb':'$F_{u}$',
                     'sy_con':'$s_{y,con}$','su_con':'$s_{u,con}$','sb_con':'$s_{b,con}$',
                     'D_con':r'$D_{con}$'}
# Donor_dict={"LEIULANA_67-17":"PT3","LEIULANA_57-17":"PT4","LEIULANA_48-17":"PT5",
#             "LEIULANA_22-17":"PT6","LEIULANA_60-17":"PT7"}
# doda.Naming.to_dict()

VIParams_don=["Sex","Age","Storagetime","BMI",
              "ICDCodes","Special_Anamnesis","Note_Anamnesis",
              "Fixation","Note_Fixation"]

#%%% Type
if ptype=="TBT":   
    path = "D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/"+Version+"/TBT/"
    name_in   = "B3-B7_TBT-Summary"
    name_out  = "B3-B7_TBT-Conclusion"
    name_Head = "Compact bone"
    YM_con=['inc','R','A0Al','meanwoso']
    YM_opt=['inc','R','D2Mgwt','meanwoso']
    YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
    YM_opt_str='E_{}_{}_{}_{}'.format(*YM_opt)
    VIParams_gen=["Designation","Origin","Donor"]
    VIParams_geo=["thickness_mean","width_mean",
                  "Area_CS","Volume","geo_MoI_mid","Density_app"]
    VIParams_mat=[YM_con_str,YM_opt_str,
                  "fy","ey_opt","Uy_opt",
                  "fu","eu_opt","Uu_opt",
                  "fb","eb_opt","Ub_opt"]
    VIParams_rename = {'geo_MoI_mid':'MoI_mid',
                       YM_con_str:'E_con',YM_opt_str:'E_opt'}
    VIPar_plt_renamer.update({YM_opt_str:'$E_{opt}$',YM_con_str:'$E_{con}$',
                              'geo_MoI_mid':r'$I_{mid}$'})
elif ptype=="ACT":
    path = "D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/"+Version+"/ACT/"
    name_in   = "B3-B7_ACT-Summary"
    name_out  = "B3-B7_ACT-Conclusion"
    name_Head = "Trabecular bone"
    YM_con=['lsq','R','A0Al','E']
    YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
    # YM_opt_str='E_{}_{}_{}_{}'.format(*YM_opt)
    VIParams_gen=["Designation","Origin","Donor","Direction_test"]
    VIParams_geo=["Length_test",
                  "Area_CS","Volume","Density_app"]
    VIParams_mat=[YM_con_str,
                  "fy","ey_con","Uy_con",
                  "fu","eu_con","Uu_con",
                  "fb","eb_con","Ub_con"]
    VIParams_rename = {YM_con_str:'E_con'}
    VIPar_plt_renamer.update({YM_con_str:'$E_{con}$'})
elif ptype=="ATT":
    path = "D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/"+Version+"/ATT/"
    name_in   = "B3-B7_ATT-Summary"
    name_out  = "B3-B7_ATT-Conclusion"
    name_Head = "Soft tissue"
    YM_con=['lsq','R','A0Al','E']
    YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
    D_con_str='D_{}_{}_{}'.format(*YM_con[:-1])
    VIParams_gen=["Designation","Origin","Donor"]
    VIParams_geo=["Area_CS","Volume","Density_app"]
    VIParams_mat=[YM_con_str,
                  "fy","ey_con","Uy_con",
                  "fu","eu_con","Uu_con",
                  "fb","eb_con","Ub_con",
                  D_con_str,
                  "Fy","sy_con","Wy_con",
                  "Fu","su_con","Wu_con",
                  "Fb","sb_con","Wb_con"]
    VIParams_rename = {D_con_str:'D_con',
                       YM_con_str:'E_con'}
    VIPar_plt_renamer.update({D_con_str:'$D_{con}$',YM_con_str:'$E_{con}$'})
if ptype=="TBT" or ptype=="ACT":  
    relist=[' 4',' 5',' 1',' L',' R',' anterior superior',
            ' proximal',' distal',' ventral',
            ' anterior',' posterior',
            ' supraacetabular',' postacetabular'] # Zu entfernende Worte
    Locdict={'Ala ossis ilii':             'AOIl',
             'Ala ossis ilii superior':    'AOIlS',
             'Ala ossis ilii inferior':    'AOIlI',
             'Corpus ossis ilii':          'COIl',
             'Corpus ossis ischii':        'COIs',
             'Ramus superior ossis pubis': 'ROPu',
             'Ramus ossis ischii':         'ROIs',
             'Corpus vertebrae lumbales':  'CVLu',
             'Corpus vertebrae sacrales':  'CVSa'}
elif ptype == "ATT":
    relist=[' L',' M',' R',
            ' caudal',' medial',' lateral',' cranial',
            ' längs',' quer'] # Zu entfernende Worte
    Locdict={'Ligamentum sacrotuberale':                    'SacTub',
             'Ligamentum sacrospinale':                     'SacSpi',
             'Ligamentum iliolumbale':                      'IliLum',
             'Ligamentum inguinale':                        'Ingui',
             'Ligamentum pectineum':                        'Pecti',
             'Ligamenta sacroiliaca anteriora':             'SaIlia',
             'Ligamenta sacroiliaca posteriora':            'SaIlip',
             'Ligamentum sacroiliacum posterior longum':    'SaIlil',
             'Membrana obturatoria':                        'MObt',
             'Fascia glutea':                               'FGlu',
             'Fascia lumbalis':                             'FLum',
             'Fascia crescent':                             'FCres',
             'Fascia endopelvina':                          'FEnPe',
             'Fascia thoracolumbalis lamina superficalis':  'FTCLls',
             'Fascia thoracolumbalis lamina profunda':      'FTCLlp'}
else: print("Failure ptype!!!")

#%%% Output
out_full= os.path.abspath(path+name_out)
path_doda = os.path.abspath('F:/Messung/000-PARAFEMM_Patientendaten/PARAFEMM_Donordata_full.xlsx')
h5_conc = 'Summary'
h5_data = 'Test_Data'
VIParams = copy.deepcopy(VIParams_geo)
VIParams.extend(VIParams_mat)

MG_logopt['logfp']=open(out_full+'.log','w')
Evac.MG_strlog(name_out, **MG_logopt)
Evac.MG_strlog("\n   Paths:", **MG_logopt)
Evac.MG_strlog("\n   - in:", **MG_logopt)
Evac.MG_strlog("\n         {}".format(os.path.abspath(path+name_in+'.h5')), **MG_logopt)
Evac.MG_strlog("\n   - out:", **MG_logopt)
Evac.MG_strlog("\n         {}".format(out_full), **MG_logopt)
Evac.MG_strlog("\n   - donor:", **MG_logopt)
Evac.MG_strlog("\n         {}".format(path_doda), **MG_logopt)
# Evac.MG_strlog("\n   Donors:"+Evac.str_indent('\n{}'.format(pd.Series(Donor_dict).to_string()),5), **MG_logopt)

#%%% Read
data_read = pd.HDFStore(path+name_in+'.h5','r')
dfa=data_read.select(h5_conc)
dft=data_read.select(h5_data)
data_read.close()
del dfa['No']
doda = pd.read_excel(path_doda, skiprows=range(1,2), index_col=0)

# Add Values
dfa.Failure_code  = Evac.list_cell_compiler(dfa.Failure_code)
dfa['statistics'] = Evac.list_interpreter(dfa.Failure_code, no_stats_fc)

h=dfa.Origin
for i in relist:
    h=h.str.replace(i,'')
for i in h.index:
    if h[i].endswith(' '): h[i]=h[i][:-1]
h2=h.map(Locdict)
if (h2.isna()).any(): print('Locdict have missing/wrong values! \n   (Lnr: %s)'%['{:s}'.format(i) for i in h2.loc[h2.isna()].index])
dfa.insert(3,'Origin_short',h)
dfa.insert(4,'Origin_sshort',h2)
del h, h2

## Add Strain Energy Density
tmp=dfa.loc(axis=1)[dfa.columns.str.startswith('W')].copy(deep=True)
if ptype=="TBT":
    tmp=tmp.div(dfa.thickness_2 * dfa.width_2 * dfa.Length_test,axis=0)*9
else:
    tmp=tmp.div(dfa.Area_CS * dfa.Length_test,axis=0)
tmp.columns=tmp.columns.str.replace('W','U')
dfa=pd.concat([dfa,tmp],axis=1)

doda['Date_Test']=pd.to_datetime(dfa.loc[dfa.statistics].groupby('Donor')['Date_test'].max())
doda['Storagetime']=(doda['Date_Test']-doda['Date_Death']).dt.days

# Select
cs = dfa.loc[dfa.statistics]
cs_num_cols = cs.select_dtypes(include=['int','float']).columns
cs_num_cols = cs_num_cols[np.invert(cs_num_cols.str.startswith('OPT_'))] # exclude Options

cs_short_gen = cs[VIParams_gen]
cs_short_geo = cs[VIParams_geo]
cs_short_mat = cs[VIParams_mat]
cs_short = pd.concat([cs_short_gen,cs_short_geo,cs_short_mat],axis=1)
cols_short = cs_short.columns
cs_short.rename(columns=VIParams_rename,inplace=True)
css_ncols = cs_short.select_dtypes(include=['int','float']).columns
# VIParams_short=[VIParams_rename.get(item,item)  for item in VIParams]
cs_doda = cs.join(doda,on='Donor',how='inner',rsuffix='Don_')

#%%%Eva
agg_funcs=['count','mean',Evac.meanwoso,'median',
           'std', Evac.stdwoso,
           # Evac.coefficient_of_variation, Evac.coefficient_of_variation_woso,Evac.confidence_interval,
           'min','max',Evac.CImin,Evac.CImax]
cs_eva = Evac.pd_agg(cs,agg_funcs,True)
cs_short_eva = Evac.pd_agg(cs_short,agg_funcs,True)
if ptype=="ATT":
    tmp=pd.concat([cs_short,cs['Type']],axis=1).groupby('Type')
    c_short_Type_eva = Evac.pd_agg(tmp,agg_funcs,True).stack()
tmp=pd.concat([cs_short,cs['Origin_short']],axis=1).groupby('Origin_short')
h_short_eva = Evac.pd_agg(tmp,agg_funcs,True).stack()

if ptype=="ACT":
    # h_dir_eva=cs.groupby(['Origin_short','Direction_test'])[cs_num_cols].agg(agg_funcs)
    tmp=pd.concat([cs_short,cs['Origin_short']],axis=1).groupby(['Origin_short','Direction_test'])
    h_dir_eva_short = Evac.pd_agg(tmp,agg_funcs,True)
    h_dir_eva_short = h_dir_eva_short.stack()
    
if ptype == "ATT":
    cs_cfl = cs[['cyc_f_lo','cyc_f_hi']].div(cs.fu,axis=0).dropna()
    
    cs_E = cs.loc(axis=1)[cs.columns.str.startswith('E_lsq')]
    cs_E.columns = cs_E.columns.str.split('_', expand=True)
    cs_E = cs_E.droplevel(0,axis=1)
    cs_E_lsq_m=cs_E.loc(axis=1)[idx['lsq',:,'A0Al','E']].droplevel([0,2,3],axis=1)
    cs_E_lsq_m_pR=cs_E_lsq_m.loc(axis=1)[cs_E_lsq_m.columns.str.contains(r'^C',regex=True)]
    cs_E_lsq_m_pR=cs_E_lsq_m_pR.div(cs_E_lsq_m['F'],axis=0)
    cs_E_lsq_m_pR.dropna(axis=0, inplace=True)
    
    cs_E_lsq_m_pR_rise = cs_E_lsq_m_pR.loc(axis=1)[cs_E_lsq_m_pR.columns.str.contains(r'^C.*\+$',regex=True)]
    cs_E_lsq_m_pR_fall = cs_E_lsq_m_pR.loc(axis=1)[cs_E_lsq_m_pR.columns.str.contains(r'^C.*\-$',regex=True)]

    cs_E_lsq_m_pR_rise.columns=cs_E_lsq_m_pR_rise.columns.str.replace('C','',regex=True).str.replace('\+','',regex=True)
    cs_E_lsq_m_pR_fall.columns=cs_E_lsq_m_pR_fall.columns.str.replace('C','',regex=True).str.replace('\-','',regex=True)

    cs_epl = cs.loc(axis=1)[cs.columns.str.startswith('epl')]
    cs_epl.columns = cs_epl.columns.str.split('_', expand=True)
    cs_epl_m=cs_epl.loc(axis=1)[idx['epl',:,:]].droplevel([0,1],axis=1)
    cs_epl_m.columns = cs_epl_m.columns.str.replace('C','',regex=True)

    cs_E_lsq_m_pR_rise_so=Evac.stat_outliers(cs_E_lsq_m_pR_rise.mean(axis=1))
    cs_E_lsq_m_pR_fall_so=Evac.stat_outliers(cs_E_lsq_m_pR_fall.mean(axis=1))
    cs_epl_m_so = Evac.stat_outliers(cs_epl_m.mean(axis=1))
    cs_E_lsq_m_pR_rise_si=Evac.stat_outliers(cs_E_lsq_m_pR_rise.mean(axis=1),out='inner')
    cs_E_lsq_m_pR_fall_si=Evac.stat_outliers(cs_E_lsq_m_pR_fall.mean(axis=1),out='inner')
    cs_epl_m_si = Evac.stat_outliers(cs_epl_m.mean(axis=1),out='inner')
    
    tmp = cs_E_lsq_m
    tmp.columns = 'E_con_'+tmp.columns
    tmp2 = cs_epl_m
    tmp2.columns = 'epl_con_'+tmp2.columns
    cs_cyc = pd.concat([cs_cfl,tmp2,tmp],axis=1)


    
#%%% Assessment Codes
Evac.MG_strlog("\n "+"="*100,**MG_logopt)
Evac.MG_strlog("\n Assessment Codes Analysies:",**MG_logopt)
# complete Codes
fc_b_df, fc_unique = Evac.Failure_code_bool_df(dfa.Failure_code, 
                                      sep=',',level=2, strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_all_sum = fc_b_df.sum().sort_values(ascending=False)
fc_b_df_fail_sum = fc_b_df[dfa.statistics==False].sum().sort_values(ascending=False)
fc_b_df_nofail_sum = fc_b_df[dfa.statistics==True].sum().sort_values(ascending=False)

txt='Assessment codes frequency:'
Evac.MG_strlog(Evac.str_indent(txt),**MG_logopt)
txt='- all:\n   %s'%fc_b_df_all_sum[fc_b_df_all_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),**MG_logopt)
txt='- fail:\n   %s'%fc_b_df_fail_sum[fc_b_df_fail_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),**MG_logopt)
txt='- no fail:\n   %s'%fc_b_df_nofail_sum[fc_b_df_nofail_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),**MG_logopt)

#%%% Additional Logging
if ptype == "ATT":
    txt="\n "+"="*100
    txt+=("\n Cyclic-loading influence: (%d samples)"%cs.OPT_pre_load_cycles[cs.OPT_pre_load_cycles>0].size)
    txt += Evac.str_indent('- cyclic loading stress level (related to ultimate strength):')
    txt += Evac.str_indent('- preload (aim: 0.10):',6)
    txt += Evac.str_indent('  {0:.3f} ± {1:.3f} ({2:.3f}-{3:.3f})'.format(*cs_cfl.cyc_f_lo.agg(['mean','std','min','max'])),6)
    txt += Evac.str_indent('- cyclic (aim: 0.30):',6)
    txt += Evac.str_indent('  {0:.3f} ± {1:.3f} ({2:.3f}-{3:.3f})'.format(*cs_cfl.cyc_f_hi.agg(['mean','std','min','max'])),6)
    txt += Evac.str_indent('- cyclic related to final Youngs Modulus:')
    txt += Evac.str_indent('- loading (ascending):',6)
    txt += Evac.str_indent(cs_E_lsq_m_pR_rise.agg(agg_funcs).T,9)
    txt += Evac.str_indent('statistical outliers: %d'%len(cs_E_lsq_m_pR_rise_so),9)
    txt += Evac.str_indent(cs.Failure_code[cs_E_lsq_m_pR_rise_so],9)    
    txt += Evac.str_indent('- relaxation (descending):',6)
    txt += Evac.str_indent(cs_E_lsq_m_pR_fall.agg(agg_funcs).T,9)
    txt += Evac.str_indent('statistical outliers: %d'%len(cs_E_lsq_m_pR_fall_so),9)
    txt += Evac.str_indent(cs.Failure_code[cs_E_lsq_m_pR_fall_so],12)
    txt += Evac.str_indent('- plastic strain after cycle:')
    txt += Evac.str_indent(cs_epl_m.agg(agg_funcs).T,9)
    txt += Evac.str_indent('statistical outliers: %d'%len(cs_epl_m_so),9)
    txt += Evac.str_indent(cs.Failure_code[cs_epl_m_so],12)
    Evac.MG_strlog(txt,**MG_logopt)
    
#%% Data-Export
writer = pd.ExcelWriter(out_full+'.xlsx', engine = 'xlsxwriter')
# tmp=dft_comb_rel.rename(VIPar_plt_renamer,axis=1)
dfa.to_excel(writer, sheet_name='Data')
if ptype=="ATT":
    cs_cyc.to_excel(writer, sheet_name='Data-Cyclic')
# tmp=dfa.append(cs_eva,sort=False)
# if ptype=="ATT": tmp=tmp.append(c_short_Type_eva,sort=False)
cs_short_eva.to_excel(writer, sheet_name='Summary')
if ptype=="ATT":
    c_short_Type_eva.loc['Fascia'].to_excel(writer, sheet_name='Summary-Fascia')
    c_short_Type_eva.loc['Ligament'].to_excel(writer, sheet_name='Summary-Ligament')
h_short_eva.to_excel(writer, sheet_name='Location')
if ptype=="ACT": 
    h_dir_eva_short.to_excel(writer, sheet_name='Location-Direction')
tmp=doda.loc[cs.Donor.drop_duplicates(),VIParams_don]
tmp.to_excel(writer, sheet_name='Donor data')

writer.close()


#%% Stat. tests
Evac.MG_strlog("\n\n "+"="*100, **MG_logopt)
Evac.MG_strlog("\n Statistical tests ", **MG_logopt)
#%%% Set
alpha=0.05
stat_ttype_parametric=False # Testtype

if stat_ttype_parametric:
    mpop="ANOVA"
    # Tukey HSD test:
    MComp_kws={'do_mcomp_a':2, 'mcomp':'TukeyHSD', 'mpadj':'', 
                'Ffwalpha':1, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    mcorr="pearson"
else:
    mpop="Kruskal-Wallis H-test"
    # Mann-Whitney U test: (two independent)
    MComp_kws={'do_mcomp_a':2, 'mcomp':'mannwhitneyu', 'mpadj':'holm', 
                'Ffwalpha':1, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    mcorr="spearman"
    MCompdf_kws={'do_mcomp_a':1, 'mcomp':'mannwhitneyu', 'mpadj':'holm', 
                 'Ffwalpha':1, 'mkws':{}}

#%%% Distribution
Evac.MG_strlog("\n\n "+"-"*100, **MG_logopt)
Evac.MG_strlog("\n Distribution tests ", **MG_logopt)
tmp = Evac.Dist_test_multi(cs_short.loc(axis=1)[css_ncols], alpha=alpha)
Evac.MG_strlog(Evac.str_indent(tmp.to_string()), **MG_logopt)

#%%% Variance analyses
Evac.MG_strlog("\n\n "+"-"*100, **MG_logopt)
Evac.MG_strlog("\n %s-Harvesting location: (Groups are significantly different for p < %.3f)"%(mpop,alpha),**MG_logopt)
tmp=pd.concat([cs_short,cs['Origin_short']],axis=1)
tmp=Evac.Multi_conc(df=tmp,group_main='Origin_short', anat='VAwoSg',
               stdict=css_ncols.to_series().to_dict(), 
               met=mpop, alpha=alpha, kws=MCompdf_kws)
Evac.MG_strlog(Evac.str_indent(tmp.loc(axis=1)['DF1':'H0'].to_string()),**MG_logopt)
Evac.MG_strlog("\n  -> Multicomparision (%s)):"%MComp_kws['mcomp'],**MG_logopt)
for i in tmp.loc[tmp.H0 == False].index:
    txt="{}:\n{}".format(i,tmp.loc[i,'MCP'],)
    Evac.MG_strlog(Evac.str_indent(txt,5),**MG_logopt)
Evac.MG_strlog("\n\n   -> Multicomparison relationship interpretation:",**MG_logopt)
tmp2=tmp.loc[tmp.H0 == False]['MCP'].apply(Evac.MComp_interpreter)
tmp2=tmp2.droplevel(0).apply(pd.Series)[0].apply(pd.Series).T
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),5),**MG_logopt)

Evac.MG_strlog("\n\n "+"-"*100, **MG_logopt)
Evac.MG_strlog("\n %s-Donor: (Groups are significantly different for p < %.3f)"%(mpop,alpha),**MG_logopt)
tmp=Evac.Multi_conc(df=cs_short,group_main='Donor', anat='VAwoSg',
               stdict=css_ncols.to_series().to_dict(), 
               met=mpop, alpha=alpha, kws=MCompdf_kws)
Evac.MG_strlog(Evac.str_indent(tmp.loc(axis=1)['DF1':'H0'].to_string()),**MG_logopt)
Evac.MG_strlog("\n  -> Multicomparision (%s)):"%MComp_kws['mcomp'],**MG_logopt)
for i in tmp.loc[tmp.H0 == False].index:
    txt="{}:\n{}".format(i,tmp.loc[i,'MCP'],)
    Evac.MG_strlog(Evac.str_indent(txt,5),**MG_logopt)
Evac.MG_strlog("\n\n   -> Multicomparison relationship interpretation:",**MG_logopt)
tmp2=tmp.loc[tmp.H0 == False]['MCP'].apply(Evac.MComp_interpreter)
tmp2=tmp2.droplevel(0).apply(pd.Series)[0].apply(pd.Series).T
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),5),**MG_logopt)


#%%% Correlations
cs_short_corr = cs_short.corr(method=mcorr)

doda_corr_d=doda.loc(axis=1)[VIParams_don]
doda_corr_d['Sex'] = doda_corr_d['Sex'].map({'f':1,'m':-1})
doda_corr_d_num_cols=doda_corr_d.select_dtypes(include=['int','float']).columns
doda_corr_i= Evac.ICD_bool_df(doda.ICDCodes,**{'level':0})
# doda_corr_i['SP'] = np.invert(doda['Special_Anamnesis'].isna()).astype(int)
doda_di = pd.concat([doda_corr_d.loc(axis=1)[doda_corr_d_num_cols],doda_corr_i],axis=1)
cs_doda_short=pd.merge(left=cs_short,right=doda_di,left_on='Donor',right_index=True)
doda_corr=cs_doda_short.corr(method=mcorr)

#%%% Regressions
if ptype == "TBT":
    lR=[[YM_con_str,'fu'],[YM_con_str,'Density_app'],
        [YM_opt_str,'fu'],[YM_opt_str,'Density_app'],
        ['fu','Density_app'],['Density_app','Age']]
    lRvar_ren={YM_con_str:'elastic modulus (conventional)',
               YM_opt_str:'elastic modulus (optical)',
               'fu':'ultimate strength',
               'Density_app':'apparent density',
               'Age':'donor age'}
else:
    lR=[[YM_con_str,'fu'],[YM_con_str,'Density_app'],
        ['fu','Density_app'],['Density_app','Age']]
    lRvar_ren={YM_con_str:'elastic modulus (conventional)',
               'fu':'ultimate strength',
               'Density_app':'apparent density',
               'Age':'donor age'}

Lin_reg_df, txt = linRegStats_Multi(cs_doda, lR, lRvar_ren, VIPar_plt_renamer)

Evac.MG_strlog("\n\n "+"-"*100, **MG_logopt)
Evac.MG_strlog("\n Linear regressions:", **MG_logopt)
Evac.MG_strlog(('').join(txt.values), **MG_logopt)

#%% Plots
#%%% Paper
# Plose-One-Dimensions: 300-600 dpi
    # Minimum width 6.68cm-2.63in
    # Maximum width 19.05cm-7.5in-2,250 pixels (text column: 5.2 inches (13.2 cm))
    # Height maximum (wocaption) 22.23cm-8.75in
subfiglabeldict=dict(off=(0,5.5), xg=0.0, yg=1.0,
                     fontdict=dict(fontsize=9, 
                                   fontweight="bold", 
                                   fontstyle="normal",
                                   ha='left', va='bottom'),
                     bboxdict=dict(boxstyle="circle", 
                                   fc="gray", alpha=0.3,
                                   ec="black", lw=1, pad=0.2))
#%%%% Overview E TBT + ACT
if ptype in ["TBT","ACT"]:
    gs_kw = dict(width_ratios=[0.715, 1.0, 0.285], height_ratios=[1.5, 1])
    fig, ax = plt.subplot_mosaic([['Donor','Location','Pelvis'],
                                  ['Reg','Reg','Reg']],
                                  gridspec_kw=gs_kw,
                                  # empty_sentinel='lower mid',
                                  figsize=(16/2.54, 12/2.54),
                                  constrained_layout=True)
    if ptype == "TBT":
        pltvartmp=YM_opt_str
        with get_sample_data("D:/Gebhardt/Veröffentlichungen/2023-08-03_ Präp/IMG/05-Results/Loc-Kort_wb.png") as file:
            arr_img = plt.imread(file)
    elif ptype == "ACT":
        pltvartmp=YM_con_str
        with get_sample_data("D:/Gebhardt/Veröffentlichungen/2023-08-03_ Präp/IMG/05-Results/Loc-Kort_wb.png") as file:
            arr_img = plt.imread(file)
    imagebox = OffsetImage(arr_img, zoom=0.105)
    imagebox.image.axes = ax['Pelvis']
    ab = AnnotationBbox(imagebox, (0,0),
                        xybox=(0.43, 0.5),
                        xycoords='axes fraction',frameon=False)
    ax['Pelvis'].add_artist(ab)
    ax['Pelvis'].grid(False)
    ax['Pelvis'].axis('off')
    
    df=pd.melt(cs, id_vars=['Origin_sshort'], value_vars=[pltvartmp])
    axt = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax['Location'], 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax['Location'], dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax['Location'].set_title('By harvesting region')
    ax['Location'].set_xlabel('Region')
    ax['Location'].set_ylabel('')
    ax['Location'].tick_params(axis='y',which='both',left=False,labelleft=False)
    plt_add_figsubl(text='b', axo=ax['Location'],**subfiglabeldict)
    
    df=pd.melt(cs, id_vars=['Donor'], value_vars=[pltvartmp])
    df.Donor.replace(doda.Naming,inplace=True)
    axt = sns.boxplot(x="Donor", y="value", data=df, ax=ax['Donor'], 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Donor", y="value",
                        data=df, ax=ax['Donor'], dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax['Donor'].set_title('By cadaver')
    ax['Donor'].set_xlabel('Cadaver')
    ax['Donor'].set_ylabel('Elastic modulus in MPa')
    ax['Location'].sharey(ax['Donor'])
    plt_add_figsubl(text='a', axo=ax['Donor'],**subfiglabeldict)
    
    axt = sns.regplot(x="Density_app", y=pltvartmp, data=cs,
                      ax=ax['Reg'], color = sns.color_palette()[0], scatter_kws={'s':2})
    axtmp = ax['Reg'].twiny()
    axt = sns.regplot(x="fu", y=pltvartmp, data=cs,
                      ax=axtmp, color = sns.color_palette()[1], scatter_kws={'s':2})
    # ax['Reg'].set_title('Linear Regression')
    ax['Reg'].set_xlabel('Apparent density in g/cm³',color=sns.color_palette()[0])
    ax['Reg'].set_ylabel('Elastic modulus in MPa')
    ax['Reg'].tick_params(axis='x', colors=sns.color_palette()[0])
    axtmp.set_xlabel('Ultimate strength in MPa',color=sns.color_palette()[1])
    axtmp.tick_params(axis='x', colors=sns.color_palette()[1])
    plt_add_figsubl(text='c', axo=ax['Reg'],**subfiglabeldict)
    fig.suptitle(None)
    Evac.plt_handle_suffix(fig,path=path+"Paper-OV",**plt_Fig_dict)

#%%%% Overview E ATT
if ptype == "ATT":
    gs_kw = dict(width_ratios=[0.715, 1.0, 0.285], height_ratios=[1.5, 1.5, 1])
    fig, ax = plt.subplot_mosaic([['DonorF','LocationF','PelvisF'],
                                  ['DonorL','LocationL','PelvisL'],
                                  ['Reg','Reg','Reg']],
                                  gridspec_kw=gs_kw,
                                  # empty_sentinel='lower mid',
                                  figsize=(16/2.54, 1.6*12/2.54),
                                  constrained_layout=True)

    pltvartmp=YM_con_str
    with get_sample_data("D:/Gebhardt/Veröffentlichungen/2023-08-03_ Präp/IMG/05-Results/Loc-Kort_wb.png") as file:
        arr_img = plt.imread(file)
    imagebox = OffsetImage(arr_img, zoom=0.105)
    imagebox.image.axes = ax['PelvisF']
    ab = AnnotationBbox(imagebox, (0,0),
                        xybox=(0.43, 0.5),
                        xycoords='axes fraction',frameon=False)
    ax['PelvisF'].add_artist(ab)
    ax['PelvisF'].grid(False)
    ax['PelvisF'].axis('off')
    
    df=pd.melt(cs[cs.Type=='Fascia'], id_vars=['Origin_sshort'], value_vars=[pltvartmp])
    axt = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax['LocationF'], 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax['LocationF'], dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax['LocationF'].set_title('Fascia by harvesting region')
    ax['LocationF'].set_xlabel('Region')
    ax['LocationF'].set_ylabel('')
    ax['LocationF'].tick_params(axis='y',which='both',left=False,labelleft=False)
    plt_add_figsubl(text='b', axo=ax['LocationF'],**subfiglabeldict)
    
    df=pd.melt(cs[cs.Type=='Fascia'], id_vars=['Donor'], value_vars=[pltvartmp])
    df.Donor.replace(doda.Naming,inplace=True)
    axt = sns.boxplot(x="Donor", y="value", data=df, ax=ax['DonorF'], 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Donor", y="value",
                        data=df, ax=ax['DonorF'], dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax['DonorF'].set_title('Fascia by cadaver')
    ax['DonorF'].set_xlabel('Cadaver')
    ax['DonorF'].set_ylabel('Elastic modulus in MPa')
    ax['LocationF'].sharey(ax['DonorF'])
    plt_add_figsubl(text='a', axo=ax['DonorF'],**subfiglabeldict)
    
    
    with get_sample_data("D:/Gebhardt/Veröffentlichungen/2023-08-03_ Präp/IMG/05-Results/Loc-Kort_wb.png") as file:
        arr_img = plt.imread(file)
    imagebox = OffsetImage(arr_img, zoom=0.105)
    imagebox.image.axes = ax['PelvisL']
    ab = AnnotationBbox(imagebox, (0,0),
                        xybox=(0.43, 0.5),
                        xycoords='axes fraction',frameon=False)
    ax['PelvisL'].add_artist(ab)
    ax['PelvisL'].grid(False)
    ax['PelvisL'].axis('off')
    
    df=pd.melt(cs[cs.Type=='Ligament'], id_vars=['Origin_sshort'], value_vars=[pltvartmp])
    axt = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax['LocationL'], 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax['LocationL'], dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax['LocationL'].set_title('Ligaments by harvesting region')
    ax['LocationL'].set_xlabel('Region')
    ax['LocationL'].set_ylabel('')
    ax['LocationL'].tick_params(axis='y',which='both',left=False,labelleft=False)
    plt_add_figsubl(text='d', axo=ax['LocationL'],**subfiglabeldict)
    
    df=pd.melt(cs[cs.Type=='Ligament'], id_vars=['Donor'], value_vars=[pltvartmp])
    df.Donor.replace(doda.Naming,inplace=True)
    axt = sns.boxplot(x="Donor", y="value", data=df, ax=ax['DonorL'], 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Donor", y="value",
                        data=df, ax=ax['DonorL'], dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax['DonorL'].set_title('Ligaments by cadaver')
    ax['DonorL'].set_xlabel('Cadaver')
    ax['DonorL'].set_ylabel('Elastic modulus in MPa')
    ax['LocationL'].sharey(ax['DonorL'])
    plt_add_figsubl(text='c', axo=ax['DonorL'],**subfiglabeldict)
    
    axt = sns.regplot(x="Density_app", y=pltvartmp, data=cs,
                      ax=ax['Reg'], color = sns.color_palette()[0], scatter_kws={'s':2})
    axtmp = ax['Reg'].twiny()
    axt = sns.regplot(x="fu", y=pltvartmp, data=cs,
                      ax=axtmp, color = sns.color_palette()[1], scatter_kws={'s':2})
    # ax['Reg'].set_title('Linear Regression')
    ax['Reg'].set_xlabel('Apparent density in g/cm³',color=sns.color_palette()[0])
    ax['Reg'].set_ylabel('Elastic modulus in MPa')
    ax['Reg'].tick_params(axis='x', colors=sns.color_palette()[0])
    axtmp.set_xlabel('Ultimate strength in MPa',color=sns.color_palette()[1])
    axtmp.tick_params(axis='x', colors=sns.color_palette()[1])
    plt_add_figsubl(text='e', axo=ax['Reg'],**subfiglabeldict)
    fig.suptitle(None)
    Evac.plt_handle_suffix(fig,path=path+"Paper-OVS",**plt_Fig_dict)


#%%%% Directions ACT
if ptype == "ACT":
    gs_kw = dict(width_ratios=[0.75, 0.25], height_ratios=[1, 1])
    fig, ax = plt.subplot_mosaic([['E', 'Pelvis'],
                                  ['fu','Pelvis']],
                                  gridspec_kw=gs_kw,
                                  # empty_sentinel='lower mid',
                                  figsize=(16/2.54, 12/2.54),
                                  constrained_layout=True)
    with get_sample_data("D:/Gebhardt/Veröffentlichungen/2023-08-03_ Präp/IMG/05-Results/Loc-Kort_wb.png") as file:
        arr_img = plt.imread(file)
    imagebox = OffsetImage(arr_img, zoom=0.105)
    imagebox.image.axes = ax['Pelvis']
    ab = AnnotationBbox(imagebox, xy = (0,0), box_alignment=(0, 0),
                        # xybox=(0.43, 0.5),
                        xycoords='axes fraction',frameon=False)
    ax['Pelvis'].add_artist(ab)
    ax['Pelvis'].grid(False)
    ax['Pelvis'].axis('off')
    df=pd.melt(cs, id_vars=['Origin_sshort','Direction_test'], value_vars=[YM_con_str])
    df.sort_values(['Origin_sshort','Direction_test'],inplace=True)
    axt = sns.boxplot(x="Origin_sshort", y="value", hue="Direction_test",
                     # data=df, ax=ax['E'], palette={'x':'r','y':'g','z':'b'}, saturation=.5,
                     data=df, ax=ax['E'], 
                     palette={'x':sns.color_palette()[3],
                              'y':sns.color_palette()[2],
                              'z':sns.color_palette()[0]}, 
                     showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                "markeredgecolor":"black",
                                                "markersize":"8","alpha":0.75})
    axt = sns.swarmplot(x="Origin_sshort", y="value", hue="Direction_test",
                       data=df, ax=ax['E'], 
                       palette={'x':sns.color_palette()[3],
                                'y':sns.color_palette()[2],
                                'z':sns.color_palette()[0]}, 
                       dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
    ax['E'].set_title('Elastic modulus by harvesting region and testing direction')
    ax['E'].set_xlabel(None)
    ax['E'].set_ylabel('Elastic modulus in MPa')
    ax['E'].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
    plt_add_figsubl(text='a', axo=ax['E'],**subfiglabeldict)
    handles, _ = ax['E'].get_legend_handles_labels()
    fig.legend(handles[0:3], ['$x$','$y$','$z$'], title="Direction", 
               loc="upper right", bbox_to_anchor=(0.94, 0.96))
    ax['E'].legend().set_visible(False)    
    df=pd.melt(cs, id_vars=['Origin_sshort','Direction_test'], value_vars=['fu'])
    df.sort_values(['Origin_sshort','Direction_test'],inplace=True)
    axt = sns.boxplot(x="Origin_sshort", y="value", hue="Direction_test",
                     data=df, ax=ax['fu'], 
                       palette={'x':sns.color_palette()[3],
                                'y':sns.color_palette()[2],
                                'z':sns.color_palette()[0]},
                     showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                "markeredgecolor":"black",
                                                "markersize":"8","alpha":0.75})
    axt = sns.swarmplot(x="Origin_sshort", y="value", hue="Direction_test",
                       data=df, ax=ax['fu'], palette={'x':'r','y':'g','z':'b'},
                       dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
    ax['fu'].set_title('Ultimate strength by harvesting region and testing direction')
    ax['fu'].set_xlabel('Region')
    ax['fu'].set_ylabel('Ultimate strength in MPa')
    ax['fu'].sharey(ax['fu'])
    ax['fu'].legend().set_visible(False)    
    plt_add_figsubl(text='b', axo=ax['fu'],**subfiglabeldict)
    fig.suptitle(None)
    Evac.plt_handle_suffix(fig,path=path+"Paper-Dir",**plt_Fig_dict)
    

#%%%% Cyclic loading ATT
if ptype == "ATT":
    gs_kw = dict(width_ratios=[1], height_ratios=[1, 1])
    fig, ax = plt.subplot_mosaic([['slvl'],
                                  ['Ecyc']],
                                  gridspec_kw=gs_kw,
                                  # empty_sentinel='lower mid',
                                  figsize=(16/2.54, 12/2.54),
                                  constrained_layout=True)
    df=pd.concat([cs_cfl,cs_E_lsq_m_pR_rise],axis=1)
    kws = {"marker":"d", "s": 40, "facecolor": "none", "edgecolor":"black",
           "linewidth": 0.5}
    sns.scatterplot(data=df.loc[cs_E_lsq_m_pR_rise_so],x='cyc_f_hi',y='1',
                    label='Statisitcal outliers', ax=ax['slvl'], **kws)
    sns.scatterplot(data=df.loc[cs_E_lsq_m_pR_rise_so],x='cyc_f_hi',y='2',
                    legend=None,ax=ax['slvl'],**kws)
    sns.scatterplot(data=df.loc[cs_E_lsq_m_pR_rise_so],x='cyc_f_hi',y='5',
                    legend=None,ax=ax['slvl'],**kws)
    sns.scatterplot(data=df.loc[cs_E_lsq_m_pR_rise_so],x='cyc_f_hi',y='10',
                    legend=None,ax=ax['slvl'],**kws)
    sns.scatterplot(data=df,x='cyc_f_hi',y='1', label='Cycle 1', ax=ax['slvl'])
    sns.scatterplot(data=df,x='cyc_f_hi',y='2', label='Cycle 2', ax=ax['slvl'])
    sns.scatterplot(data=df,x='cyc_f_hi',y='5', label='Cycle 5', ax=ax['slvl'])
    sns.scatterplot(data=df,x='cyc_f_hi',y='10',label='Cycle 10',ax=ax['slvl'])
    ax['slvl'].axvline(0.3, color='green', label='Aim')
    ax['slvl'].axhline(1.0, color='green',)
    ax['slvl'].set_title('')
    ax['slvl'].set_xlabel('Ratio of upper cyclic to ultimate stress')
    ax['slvl'].set_ylabel('Cyclic related to\nfinal elastic modulus')
    ax['slvl'].legend(ncol=3)
    plt_add_figsubl(text='a', axo=ax['slvl'],**subfiglabeldict)
    
    sns.lineplot(data=cs_E_lsq_m_pR_rise.loc[cs_E_lsq_m_pR_rise_si].T,palette=('cyan',),
                 linestyle='dashed',linewidth=0.25,legend=False,ax=ax['Ecyc'])
    sns.lineplot(data=cs_E_lsq_m_pR_fall.loc[cs_E_lsq_m_pR_fall_si].T,palette=('orange',),
                 linestyle='dashed',linewidth=0.25,legend=False,ax=ax['Ecyc'])
    lns = sns.lineplot(data=cs_E_lsq_m_pR_rise.loc[cs_E_lsq_m_pR_rise_si].mean(),color='blue',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='Loading',ax=ax['Ecyc'])
    lns = sns.lineplot(data=cs_E_lsq_m_pR_fall.loc[cs_E_lsq_m_pR_fall_si].mean(),color='red',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='Relaxation',ax=ax['Ecyc'])
    ax['Ecyc'].legend_.remove()
    ln = lns.get_legend_handles_labels()[0]
    la = lns.get_legend_handles_labels()[1]
    ax2=ax['Ecyc'].twinx()
    sns.lineplot(data=cs_epl_m.loc[cs_epl_m_si].T,palette=('limegreen',),
                 linestyle='dashed',linewidth=0.25,legend=False,ax=ax2)
    lns = sns.lineplot(data=cs_epl_m.loc[cs_epl_m_si].mean(),color='green',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='Plastic strain',ax=ax2)
    ln += lns.get_legend_handles_labels()[0]
    la += lns.get_legend_handles_labels()[1]
    ax2.legend_.remove()
    ax['Ecyc'].set_xlabel('Cycle')
    ax['Ecyc'].set_xticklabels(cs_epl_m.columns.str.replace('epl_con_',''))
    ax['Ecyc'].set_ylabel('Cyclic related to\nfinal elastic modulus')
    ax2.set_ylabel('Plastic strain')
    ax2.legend(ln,la,loc='center right')
    plt_add_figsubl(text='b', axo=ax['Ecyc'],**subfiglabeldict)
    fig.suptitle(None)
    Evac.plt_handle_suffix(fig,path=path+"Paper-Cyc",**plt_Fig_dict)   

#%%% SupMat
figsize_sup=(16.0/2.54, 22.0/2.54)
#%%%% Assesment codes
fc_b_df_mat = np.dot(fc_b_df.T, fc_b_df)
fc_b_df_freq = pd.DataFrame(fc_b_df_mat, columns = fc_unique, index = fc_unique)
ind = [(True if x in no_stats_fc else False) for x in fc_unique]
fc_b_df_hm = fc_b_df_freq.loc[ind]

gs_kw = dict(width_ratios=[1.0,0.05], height_ratios=[.15, 1],
                 wspace=0.1, hspace=0.1)
fig, ax = plt.subplot_mosaic([['excl', 'exclcb'],
                              ['all' , 'allcb' ]],
                              gridspec_kw=gs_kw,
                              figsize=figsize_sup,
                              constrained_layout=True)
ax['excl'].set_title('Excluding assessment code combination frequency')
sns.heatmap(fc_b_df_hm.loc(axis=1)[np.invert((fc_b_df_hm==0).all())],
            annot_kws={"size":8, 'rotation':0}, cmap = "Reds", 
            annot=True, ax=ax['excl'],cbar_ax=ax['exclcb'])
# ax['excl'].set_xlabel('Assessment code')
# ax['excl'].set_ylabel('Excluding assessment code')
ax['excl'].tick_params(axis='x', labelrotation=90, labelsize=8)
ax['excl'].tick_params(axis='y', labelrotation=0, labelsize=8)
ax['exclcb'].tick_params(axis='y', labelsize=8)
ax['exclcb'].yaxis.set_major_formatter(plt_tick.FormatStrFormatter('%d'))
ax['excl'].tick_params(left=True, bottom=True)
ax['all'].set_title('Assessment code combination frequency')
sns.heatmap(fc_b_df_freq, xticklabels=1, 
            cmap = "Reds", annot=False, ax=ax['all'],cbar_ax=ax['allcb'])
# ax['all'].set_xlabel('Assessment code')
# ax['all'].set_ylabel('Assessment code')
ax['all'].tick_params(axis='x', labelrotation=90, labelsize=8)
ax['all'].tick_params(axis='y', labelrotation=0, labelsize=8)
ax['allcb'].tick_params(axis='y', labelsize=8)
ax['allcb'].yaxis.set_major_formatter(plt_tick.FormatStrFormatter('%d'))
ax['all'].tick_params(left=True, bottom=True)
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-AC",**plt_Fig_dict)

#%%%% Data visualization
if ptype == 'TBT':
    subj='0000-0001-8378-3108_LEIULANA_60-17_LuPeCo_cl32b'
    epsu="Strain_opt_d_M"
    Viupu='VIP_d'
elif ptype == 'ACT':
    subj='0000-0001-8378-3108_LEIULANA_60-17_LuPeCo_tm21y'
    epsu="Strain"
    Viupu='VIP_m'
elif ptype == 'ATT':
    subj='0000-0001-8378-3108_LEIULANA_60-17_LuPeCo_sr03a'
    epsu="Strain"
    Viupu='VIP_m'
tmp=dft[subj]
tmp2=VIP_rebuild(tmp[Viupu])
tmp3=dfa.loc[subj]
tmp4=Evac.YM_sigeps_lin(tmp.Stress, tmp[epsu], 
                        ind_S=tmp2['F3'], ind_E=tmp2['F4'])
colp=sns.color_palette()
tmpf=Stress_Strain_fin(tmp, tmp2, tmp4[0],tmp4[1],epsname=epsu)
gs_kw = dict(width_ratios=[1.0], height_ratios=[1, 1, 1],
                 wspace=0.1, hspace=0.1)
fig, ax = plt.subplot_mosaic([['fotot'],
                              ['sigeps'],
                              ['sigepsf']],
                              gridspec_kw=gs_kw,
                              figsize=figsize_sup,
                              constrained_layout=True)
ax['fotot'].set_title('Analyzed measured force')
ax['fotot'].set_xlabel('Time in s')
ax['fotot'].set_ylabel('Force in N', color=colp[0])
ax['fotot'].plot(tmp.Time, tmp.Force, '-', color=colp[0], label='Measurement')
a, b=tmp.Time[tmp2],tmp.Force[tmp2]
j=np.int64(-1)
ax['fotot'].plot(a, b, 'x', color=colp[3], label='Points of interest')
for x in tmp2.index:
    j+=1
    if j%2: c=(6,-6)
    else:   c=(-6,6)
    ax['fotot'].annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), 
                         xycoords='data', xytext=c, 
                         ha="center", va="center", 
                         textcoords='offset points')
ax['fotot'].tick_params(axis='y', labelcolor=colp[0])
ax2 = ax['fotot'].twinx() 
ax2.set_ylabel('Rise in N/s and curvature in N/s²', color=colp[1])
ax2.plot(tmp.Time, tmp.driF, ':', color=colp[1], label='Rise')
ax2.plot(tmp.Time, tmp.dcuF, ':', color=colp[2], label='Curvature')
ax2.tick_params(axis='y', labelcolor=colp[1])
ln, la = ax['fotot'].get_legend_handles_labels()
ln += ax2.get_legend_handles_labels()[0]
la += ax2.get_legend_handles_labels()[1]
ax['fotot'].legend(ln,la,loc='lower right')

ax['sigeps'].set_title('Stress-strain-curve with labels')
ax['sigeps'].set_xlabel('Strain')
ax['sigeps'].set_ylabel('Stress in MPa')
ind_E='E'
ax['sigeps'].plot(tmp.loc[:tmp2[ind_E]][epsu],
                  tmp.loc[:tmp2[ind_E]]['Stress'], 
                  '-', color=colp[0], label='Measurement')
a,b = Evac.stress_linfit_plt(tmp[epsu], tmp2[['F3','F4']],
                             tmp4[0], tmp4[1], strain_offset=0, ext=0.2)
ax['sigeps'].plot(a, b, 'g-',label='Elastic modulus')
VIP_mwoC=tmp2[np.invert(tmp2.index.str.contains('C'))]
a, b=tmp[epsu][VIP_mwoC[:ind_E]],tmp.Stress[VIP_mwoC[:ind_E]]
j=np.int64(-1)
ax['sigeps'].plot(a, b, 'x', color=colp[3], label='Points of interest')
for x in VIP_mwoC[:ind_E].index:
    j+=1
    if j%2: c=(6,-6)
    else:   c=(-6,6)
    ax['sigeps'].annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                  xytext=c, ha="center", va="center", textcoords='offset points')
ax['sigeps'].legend()

ax['sigepsf'].set_title('Stress-strain-curve (start linearized)')
ax['sigepsf'].set_xlabel('Strain')
ax['sigepsf'].set_ylabel('Stress in MPa')
ind_E='B'
ax['sigepsf'].plot(tmpf.loc[:tmp2[ind_E]][epsu],
                   tmpf.loc[:tmp2[ind_E]]['Stress'], 
                   '-', color=colp[0],label='Measurement')
ax['sigepsf'].plot(tmpf.iloc[:2][epsu],
                   tmpf.iloc[:2]['Stress'], 
                   '-', color=colp[2],label='Linearization')
ax['sigepsf'].fill_between(tmpf.loc[:tmp2['Y']][epsu],
                           tmpf.loc[:tmp2['Y']]['Stress'],
                            **dict(color=colp[1], hatch='||', alpha= 0.2), 
                            label='$U_{y}$')
ax['sigepsf'].fill_between(tmpf.loc[:tmp2['U']][epsu],
                           tmpf.loc[:tmp2['U']]['Stress'],
                            **dict(color=colp[1], hatch='//', alpha= 0.2), 
                            label='$U_{u}$')
ax['sigepsf'].fill_between(tmpf.loc[:tmp2['B']][epsu],
                           tmpf.loc[:tmp2['B']]['Stress'],
                            **dict(color=colp[1], hatch='..', alpha= 0.2), 
                            label='$U_{b}$')
VIP_mwoC=tmp2[tmp2.index.str.contains('F3|F4|Y|U|B')]
a, b=tmpf[epsu][VIP_mwoC[:ind_E]],tmp.Stress[VIP_mwoC[:ind_E]]
j=np.int64(-1)
ax['sigepsf'].plot(a, b, 'x', color=colp[3], label='Points of interest')
for x in VIP_mwoC[:ind_E].index:
    j+=1
    if j%2: c=(6,-6)
    else:   c=(-6,6)
    ax['sigepsf'].annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), 
                            xycoords='data', xytext=c, 
                            ha="center", va="center", 
                            textcoords='offset points')
ax['sigepsf'].legend(ncol=2, loc='upper left', columnspacing=0.6)
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-DV",**plt_Fig_dict)

#%%%% Distribution
cs_hist = cs[VIParams_geo+VIParams_mat]/cs[VIParams_geo+VIParams_mat].max(axis=0)
#cs_hist.rename(columns=VIParams_rename,inplace=True)
tmp=pd.Series(VIParams_mat)
gs_kw = dict(width_ratios=[1], height_ratios=[1, 1, 1, 1, 1],
             wspace=0.1, hspace=0.1)
fig, ax = plt.subplot_mosaic([['Geo'],
                              ['E'],
                              ['eps'],
                              ['sig'],
                              ['U']],
                              gridspec_kw=gs_kw,
                              # empty_sentinel='lower mid',
                              figsize=figsize_sup,
                              constrained_layout=True)
sns.histplot(cs_hist[VIParams_geo], stat='count', bins=20,
                  ax=ax['Geo'], kde=True, color=sns.color_palette()[0],
                  edgecolor=None, legend = True, alpha=0.25)
old_legend = ax['Geo'].legend_
handles = old_legend.legendHandles
labels = pd.Series([t.get_text() for t in old_legend.get_texts()])
labels.replace(VIPar_plt_renamer,inplace=True)
for i in range(len(handles)):
    t=list(handles[i].get_facecolor())
    t[-1]=0.75
    handles[i].set_facecolor(tuple(t))
ax['Geo'].legend(handles, labels, loc='best', ncol=3)
#ax['Geo'].set_title('Geometry paramter')
#ax['Geo'].set_xlabel('Normed parameter')
ax['Geo'].set_ylabel('Count')
ax['Geo'].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
ax['Geo'].set_xlim((0,1))

sns.histplot(cs_hist[tmp[tmp.str.startswith('E_')].values], 
             stat='count', bins=20,
             ax=ax['E'], kde=True, color=sns.color_palette()[0],
             edgecolor=None, legend = True, alpha=0.25)
old_legend = ax['E'].legend_
handles = old_legend.legendHandles
labels = pd.Series([t.get_text() for t in old_legend.get_texts()])
labels.replace(VIPar_plt_renamer,inplace=True)
for i in range(len(handles)):
    t=list(handles[i].get_facecolor())
    t[-1]=0.75
    handles[i].set_facecolor(tuple(t))
ax['E'].legend(handles, labels, loc='best', ncol=3)
#ax['E'].set_title('Elastic modulus')
#ax['E'].set_xlabel('Normed parameter')
ax['E'].set_ylabel('Count')
ax['E'].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
ax['E'].sharex(ax['Geo'])

sns.histplot(cs_hist[tmp[tmp.str.startswith('e')].values], 
             stat='count', bins=20,
             ax=ax['eps'], kde=True, color=sns.color_palette()[0],
             edgecolor=None, legend = True, alpha=0.25)
old_legend = ax['eps'].legend_
handles = old_legend.legendHandles
labels = pd.Series([t.get_text() for t in old_legend.get_texts()])
labels.replace(VIPar_plt_renamer,inplace=True)
for i in range(len(handles)):
    t=list(handles[i].get_facecolor())
    t[-1]=0.75
    handles[i].set_facecolor(tuple(t))
ax['eps'].legend(handles, labels, loc='best', ncol=3)
#ax['eps'].set_title('Strain')
#ax['eps'].set_xlabel('Normed parameter')
ax['eps'].set_ylabel('Count')
ax['eps'].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
ax['eps'].sharex(ax['E'])

sns.histplot(cs_hist[tmp[tmp.str.startswith('f')].values], 
             stat='count', bins=20,
             ax=ax['sig'], kde=True, color=sns.color_palette()[0],
             edgecolor=None, legend = True, alpha=0.25)
old_legend = ax['sig'].legend_
handles = old_legend.legendHandles
labels = pd.Series([t.get_text() for t in old_legend.get_texts()])
labels.replace(VIPar_plt_renamer,inplace=True)
for i in range(len(handles)):
    t=list(handles[i].get_facecolor())
    t[-1]=0.75
    handles[i].set_facecolor(tuple(t))
ax['sig'].legend(handles, labels, loc='best', ncol=3)
#ax['sig'].set_title('Strength')
#ax['sig'].set_xlabel('Normed parameter')
ax['sig'].set_ylabel('Count')
ax['sig'].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
ax['sig'].sharex(ax['eps'])

sns.histplot(cs_hist[tmp[tmp.str.startswith('U')].values], 
             stat='count', bins=20,
             ax=ax['U'], kde=True, color=sns.color_palette()[0],
             edgecolor=None, legend = True, alpha=0.25)
old_legend = ax['U'].legend_
handles = old_legend.legendHandles
labels = pd.Series([t.get_text() for t in old_legend.get_texts()])
labels.replace(VIPar_plt_renamer,inplace=True)
for i in range(len(handles)):
    t=list(handles[i].get_facecolor())
    t[-1]=0.75
    handles[i].set_facecolor(tuple(t))
ax['U'].legend(handles, labels, loc='best', ncol=3)
#ax['U'].set_title('Strain energy')
ax['U'].set_xlabel('Maximum normed parameter')
ax['U'].set_ylabel('Count')
ax['U'].sharex(ax['sig'])
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-OH",**plt_Fig_dict)

#%%%% Add Eva
if ptype == "TBT":
    pltvartmp=YM_opt_str
    tmp=cs[['Donor','Origin_sshort']].agg(lambda x: x.drop_duplicates().count())
    gs_kw_width_ratios=[tmp[0]/tmp[1], 1.0]
    pltvarco='_opt'
elif ptype in ["ACT","ATT"]:
    pltvarco='_con'
    pltvartmp=YM_con_str
    tmp=cs[['Donor','Origin_sshort']].agg(lambda x: x.drop_duplicates().count())
    gs_kw_width_ratios=[tmp[0]/tmp[1], 1.0]

gs_kw = dict(width_ratios=gs_kw_width_ratios, 
             height_ratios=[1, 1, 1, 1],
             wspace=0.1, hspace=0.1)
fig, ax = plt.subplot_mosaic([['ED','EL'],
                              ['fD','fL'],
                              ['eD','eL'],
                              ['UD','UL']],
                              gridspec_kw=gs_kw,
                              # empty_sentinel='lower mid',
                              figsize=figsize_sup,
                              constrained_layout=True)
boxplt_dl(pdf=cs, var=pltvartmp, ytxt='Elastic modulus in MPa',
          xl='Origin_sshort', axl=ax['EL'], tl='By harvesting region', xtxtl='Region',
          xd='Donor', axd=ax['ED'], td='By cadaver', xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
boxplt_dl(pdf=cs, var='fy', ytxt='Yield strength in MPa',
          xl='Origin_sshort', axl=ax['fL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['fD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
boxplt_dl(pdf=cs, var='ey'+pltvarco, ytxt='Strain at yield stress',
          xl='Origin_sshort', axl=ax['eL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['eD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
boxplt_dl(pdf=cs, var='Uy'+pltvarco, ytxt='Yield strain energy in mJ/mm³',
          xl='Origin_sshort', axl=ax['UL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['UD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-OV1",**plt_Fig_dict) 
   

gs_kw = dict(width_ratios=gs_kw_width_ratios, 
             height_ratios=[1, 1, 1],
             wspace=0.1, hspace=0.1)
fig, ax = plt.subplot_mosaic([['fuD','fuL'],
                              ['euD','euL'],
                              ['UuD','UuL']],
                              gridspec_kw=gs_kw,
                              # empty_sentinel='lower mid',
                              figsize=figsize_sup,
                              constrained_layout=True)
boxplt_dl(pdf=cs, var='fu', ytxt='Ultimate strength in MPa',
          xl='Origin_sshort', axl=ax['fuL'], tl='By harvesting region', xtxtl='Region',
          xd='Donor', axd=ax['fuD'], td='By cadaver', xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
boxplt_dl(pdf=cs, var='eu'+pltvarco, ytxt='Strain at ultimate stress',
          xl='Origin_sshort', axl=ax['euL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['euD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
boxplt_dl(pdf=cs, var='Uu'+pltvarco, ytxt='Ultimate strain energy in mJ/mm³',
          xl='Origin_sshort', axl=ax['UuL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['UuD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-OV2",**plt_Fig_dict) 

gs_kw = dict(width_ratios=gs_kw_width_ratios, 
             height_ratios=[1, 1, 1],
             wspace=0.1, hspace=0.1)
fig, ax = plt.subplot_mosaic([['fbD','fbL'],
                              ['ebD','ebL'],
                              ['UbD','UbL']],
                              gridspec_kw=gs_kw,
                              # empty_sentinel='lower mid',
                              figsize=figsize_sup,
                              constrained_layout=True)
boxplt_dl(pdf=cs, var='fb', ytxt='Fracture strength in MPa',
          xl='Origin_sshort', axl=ax['fbL'], tl='By harvesting region', xtxtl='Region',
          xd='Donor', axd=ax['fbD'], td='By cadaver', xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
boxplt_dl(pdf=cs, var='eb'+pltvarco, ytxt='Strain at fracture stress',
          xl='Origin_sshort', axl=ax['ebL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['ebD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
boxplt_dl(pdf=cs, var='Ub'+pltvarco, ytxt='Fracture strain energy in mJ/mm³',
          xl='Origin_sshort', axl=ax['UbL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['UbD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming)
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-OV3",**plt_Fig_dict)

#%%%% Correlation
cs_short_corr1, cs_short_corr2 = Evac.Corr_ext(cs_short[css_ncols], method=mcorr, 
              sig_level={0.001:'$^a$',0.01:'$^b$',0.05:'$^c$',0.10:'$^d$'},
              corr_round=2)
gs_kw = dict(width_ratios=[1.0,0.05], height_ratios=[1],
                 wspace=0.1)
fig, ax = plt.subplot_mosaic([['d', 'b']],
                              gridspec_kw=gs_kw,
                              figsize=figsize_sup,
                              constrained_layout=True)
axt=sns.heatmap(cs_short_corr1, annot = cs_short_corr2, fmt='',
              center=0, vmin=-1,vmax=1, annot_kws={"size":8, 'rotation':90},
              xticklabels=1, ax=ax['d'],cbar_ax=ax['b'])
Evac.tick_label_renamer(ax=axt, renamer=VIPar_plt_renamer, axis='both')
ax['d'].tick_params(left=True, bottom=True)
ax['b'].yaxis.set_major_formatter(plt_tick.FormatStrFormatter('%0.1f'))
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-MC",**plt_Fig_dict)

doda_corr1, doda_corr2, = Evac.Corr_ext(cs_doda_short, method=mcorr,
              sig_level={0.001:'$^a$',0.01:'$^b$',0.05:'$^c$',0.10:'$^d$'},
              corr_round=2)
gs_kw = dict(width_ratios=[1.0,0.05], height_ratios=[1,1],
                 wspace=0.1)
fig, ax = plt.subplot_mosaic([['d', 'b'],
                              ['di','bi']],
                              gridspec_kw=gs_kw,
                              figsize=figsize_sup,
                              constrained_layout=True)
sns.heatmap(doda_corr1.loc[doda_corr_d_num_cols,css_ncols], 
            annot = doda_corr2.loc[doda_corr_d_num_cols,css_ncols], fmt='',
            center=0, annot_kws={"size":8, 'rotation':90},
            xticklabels=1, ax=ax['d'], cbar_ax=ax['b'])
Evac.tick_label_renamer(ax=ax['d'], renamer=VIPar_plt_renamer, axis='both')
ax['d'].set_ylabel('Donor data')
ax['d'].set_xlabel('Material parameters')
ax['d'].tick_params(axis='y', labelrotation=90)
ax['d'].tick_params(left=True, bottom=True)
ax['b'].yaxis.set_major_formatter(plt_tick.FormatStrFormatter('%0.1f'))
plt.setp(ax['d'].get_yticklabels(), va="center")

sns.heatmap(doda_corr1.loc[doda_corr_i.columns,css_ncols], 
            annot = doda_corr2.loc[doda_corr_i.columns,css_ncols], fmt='',
            center=0, annot_kws={"size":8, 'rotation':90},
            xticklabels=1, ax=ax['di'], cbar_ax=ax['bi'])
Evac.tick_label_renamer(ax=ax['di'], renamer=VIPar_plt_renamer, axis='both')
ax['di'].set_ylabel('Donor ICD-codes')
ax['di'].tick_params(axis='y', labelrotation=90)
ax['di'].tick_params(left=True, bottom=True)
ax['bi'].yaxis.set_major_formatter(plt_tick.FormatStrFormatter('%0.1f'))
plt.setp(ax['di'].get_yticklabels(), va="center")
ax['di'].set_xlabel('Material parameters')
ax['di'].tick_params(axis='x', labelrotation=90)
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-DC",**plt_Fig_dict)

#%%%% Regressions

gs_kw = dict(width_ratios=[1], 
             height_ratios=[1, 1, 1, 1],
             wspace=0.1, hspace=0.1)
fig, ax = plt.subplot_mosaic([['Efu'],['ERho'],['fuRho'],['RhoAge']],
                              gridspec_kw=gs_kw,
                              # empty_sentinel='lower mid',
                              figsize=figsize_sup,
                              constrained_layout=True)
txt=linRegfunc_fromSt(Lin_reg_df,pltvartmp,'fu','{a:.3e},{b:.3e},{R:.3f}')
sns.regplot(x="fu", y=pltvartmp, data=cs_doda,
            ax=ax['Efu'], label=txt)
ax['Efu'].legend()
ax['Efu'].set_ylabel('Elastic modulus in MPa')
ax['Efu'].set_xlabel('Ultimate strength in MPa')

txt=linRegfunc_fromSt(Lin_reg_df,pltvartmp,'Density_app','{a:.3e},{b:.3e},{R:.3f}')
sns.regplot(x="Density_app", y=pltvartmp, data=cs_doda,
            ax=ax['ERho'], label=txt)
ax['ERho'].legend()
ax['ERho'].set_ylabel('Elastic modulus in MPa')
ax['ERho'].set_xlabel('Apparent density in g/cm³')

txt=linRegfunc_fromSt(Lin_reg_df,'fu','Density_app','{a:.3e},{b:.3e},{R:.3f}')
sns.regplot(x="Density_app", y='fu', data=cs_doda,
            ax=ax['fuRho'], label=txt)
ax['fuRho'].legend()
ax['fuRho'].set_ylabel('Ultimate strength in MPa')
ax['fuRho'].set_xlabel('Apparent density in g/cm³')

txt=linRegfunc_fromSt(Lin_reg_df,'Density_app','Age','{a:.3e},{b:.3e},{R:.3f}')
sns.regplot(x="Age", y='Density_app', data=cs_doda,
            ax=ax['RhoAge'], label=txt)
ax['RhoAge'].legend()
ax['RhoAge'].set_ylabel('Apparent density in g/cm³')
ax['RhoAge'].set_xlabel('Age in years')
ax['RhoAge'].set_xlim((50,90))
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=path+"SM-LR",**plt_Fig_dict) 
#%% Close Log
MG_logopt['logfp'].close()