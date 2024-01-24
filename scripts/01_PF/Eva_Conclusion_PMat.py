# -*- coding: utf-8 -*-
"""
Conclusion of axial tensile (ATT), axial compression (ACT) and three point 
bending tests (TBT) for soft tissue, cancellous bone and cortical bone in 
project PARAFEMM. (Material parameters of the human lumbopelvic complex.)

@author: MarcGebhardt
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

from pathlib import Path
nwd = Path.cwd().resolve().parent.parent
os.chdir(nwd)
import exmecheva.common as emec

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

# , "axes.axisbelow": True

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

def linRegStats(Y, X, Y_txt, X_txt):
    des_txt = ("\n- Linear relation between %s and %s:"%(Y_txt,X_txt))
    tmp  = emec.fitting.YM_sigeps_lin(Y, X)
    smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
    out=pd.Series([*tmp,smst,des_txt],
                  index=['s','c','Rquad','fit','smstat','Description'])
    return out

def linRegStats_Multi(df, lRd, var_ren={}, var_sym={}, ind=3, addind=3):
    Lin_reg_df = pd.DataFrame(lRd, columns=['var1','var2'])
    Lin_reg_df.index = Lin_reg_df.var1+'-'+Lin_reg_df.var2
    Lin_reg_df[['var1ren','var2ren']]=Lin_reg_df[['var1','var2']].replace(var_ren)
    Lin_reg_df[['var1sym','var2sym']]=Lin_reg_df[['var1','var2']].replace(var_sym)
    tmp=Lin_reg_df.apply(
        lambda x: linRegStats(df[x.var1], df[x.var2], x.var1ren, x.var2ren),
        axis=1
        )
    Lin_reg_df=pd.concat([Lin_reg_df,tmp],axis=1)
    txt=Lin_reg_df.apply(
        lambda x: emec.output.str_indent(x.loc['Description'],ind) + 
        emec.output.str_indent(
            emec.fitting.fit_report_adder(*x[['fit','Rquad']]), ind+addind
            ) + emec.output.str_indent(x['smstat'],ind+addind), axis=1
        )
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
              hue=None, htirep={}, hn=None, 
              orderl=None, orderd=None, hue_order=None,
              bplkws={
                  'showmeans':True, 
                  'meanprops':{
                      "marker":"_", "markerfacecolor":"white", "alpha":0.75,
                      "markeredgecolor":"black", "markersize":"12"
                      }},
              splkws={
                  'dodge':True, 'edgecolor':"black", 
                  'linewidth':.5, 'alpha':.5, 'size':2
                  }):
    """
    Returns two axes containing box and overlaid swarm plot generated by 
    seaborn with shared y-axis.

    Parameters
    ----------
    pdf : pd.DataFrame
        Input data.
    var : string or None
        Column name of variable.
    ytxt : string or None
        Label of y-axis.
    xl : string or None
        Identifier of data on 2nd x-axis
    axl : plt.subplot.axis or None
        2nd axis to plot.
    tl : string or None
        Legend title of 2nd axis.
    xtxtl : string or None
        Label of 2nd x-axis.
    xd : string or None
        Identifier of data on 1st x-axis.
    axd : plt.subplot.axis or None
        1st axis to plot.
    td : string or None
        Legend title of 2nd axis.
    xtxtd : string or None
        Label of 1st x-axis.
    xltirep : dict, optional
        Dictionary for x-axis label replacement. The default is {}.
    xdtirep : TYPE, optional
        DESCRIPTION. The default is {}.
    hue : string or None, optional
        Variable for categorization. The default is None.
    htirep : dict, optional
        Dictionary for hue label replacement. The default is {}.
    hn : string or None, optional
        Title for hue categization. The default is None.
    orderl : None or list of strings, optional
        Order of levels. The default is None.
    orderd : TYPE, optional
        DESCRIPTION. The default is None.
    hue_order : None or list of strings, optional
        Order of hue levels. The default is None.
    bplkws : dict, optional
        Dictionary with additional boxplot keywords. 
        The default is {'showmeans':True,
                        'meanprops':{"marker":"_", "markerfacecolor":"white",
                                     "alpha":0.75, "markeredgecolor":"black",
                                     "markersize":"12"}}.
    splkws : dict, optional
        Dictionary with additional swarmplot keywords. 
        The default is {'dodge':True, 'edgecolor':"black", 'alpha':.5,
                        'linewidth':.5, 'size':2}.
        
    Returns
    -------
    None.

    """
    if hue is None:
        dfl=pd.melt(pdf, id_vars=[xl], value_vars=[var])
        dfd=pd.melt(pdf, id_vars=[xd], value_vars=[var])
    else:
        dfl=pd.melt(pdf, id_vars=[xl,hue], value_vars=[var])
        dfd=pd.melt(pdf, id_vars=[xd,hue], value_vars=[var])
        dfl[hue].replace(htirep,inplace=True)
        dfd[hue].replace(htirep,inplace=True)
    dfl[xl].replace(xltirep,inplace=True)
    dfd[xd].replace(xdtirep,inplace=True)
    if hue is None:
        axt = sns.boxplot(x=xl, y="value", data=dfl, order=orderl, ax=axl, **bplkws)
        axt = sns.swarmplot(x=xl, y="value",data=dfl, order=orderl,  ax=axl, **splkws)
    else:
        axt = sns.boxplot(x=xl, y="value", hue=hue, data=dfl, 
                          order=orderl, hue_order=hue_order, ax=axl, **bplkws)
        axt = sns.swarmplot(x=xl, y="value", hue=hue, data=dfl, 
                           order=orderl, hue_order=hue_order, ax=axl, **splkws)  
        h, l = axl.get_legend_handles_labels()
        axl.legend(h[0:dfl[hue].unique().size], l[0:dfl[hue].unique().size], title=hn)      
    axl.set_title(tl)
    axl.set_xlabel(xtxtl)
    axl.set_ylabel('')
    axl.tick_params(axis='y',which='both',left=False,labelleft=False)
    if hue is None:
        axt = sns.boxplot(x=xd, y="value", data=dfd, order=orderd, ax=axd,**bplkws)
        axt = sns.swarmplot(x=xd, y="value",data=dfd, order=orderd, ax=axd, **splkws)
    else:
        axt = sns.boxplot(x=xd, y="value", hue=hue, data=dfd, 
                          order=orderd, hue_order=hue_order, ax=axd,**bplkws)
        axt = sns.swarmplot(x=xd, y="value", hue=hue, data=dfd, 
                            order=orderd, hue_order=hue_order, ax=axd, **splkws)
        h, l = axd.get_legend_handles_labels()
        axd.legend(h[0:dfd[hue].unique().size], l[0:dfd[hue].unique().size], title=hn)
    axd.set_title(td)
    axd.set_xlabel(xtxtd)
    axd.set_ylabel(ytxt)
    axl.sharey(axd)
    
def boxplt_ext(
        pdf, var=None, ytxt=None, xl=None, axl=None, 
        tl=None, xtxtl=None, xltirep={},
        hue=None, htirep={}, hn=None, orderl=None, hue_order=None,
        bplkws={
            'showmeans':True,
            'meanprops':{
                "marker":"_", "markerfacecolor":"white","alpha":0.75,
                "markeredgecolor":"black", "markersize":"12"
                }},
        splkws={
            'dodge':True, 'edgecolor':"black", 'alpha':.5, 
            'linewidth':.5, 'size':2
            }):
    """
    Returns an axis containing box and overlaid swarm plot generated by seaborn.

    Parameters
    ----------
    pdf : pd.DataFrame
        Input data.
    var : string or None
        Column name of variable. The default is None.
    ytxt : string or None
        Label of y-axis.
    xl : string or None
        Identifier of data on x-axis. The default is None.
    axl : plt.subplot.axis or None
        Axis to plot. The default is None (get current axis).
    tl : string or None
        Legend title. The default is None.
    xtxtl : string or None
        Label of x-axis. The default is None.
    xltirep : dict, optional
        Dictionary for x-axis label replacement. The default is {}.
    hue : string or None, optional
        Variable for categorization. The default is None.
    htirep : dict, optional
        Dictionary for hue label replacement. The default is {}.
    hn : string or None, optional
        Title for hue categization. The default is None.
    orderl : None or list of strings, optional
        Order of levels. The default is None.
    hue_order : None or list of strings, optional
        Order of hue levels. The default is None.
    bplkws : dict, optional
        Dictionary with additional boxplot keywords. 
        The default is {'showmeans':True,
                        'meanprops':{"marker":"_", "markerfacecolor":"white",
                                     "alpha":0.75, "markeredgecolor":"black",
                                     "markersize":"12"}}.
    splkws : dict, optional
        Dictionary with additional swarmplot keywords. 
        The default is {'dodge':True, 'edgecolor':"black", 'alpha':.5,
                        'linewidth':.5, 'size':2}.

    Returns
    -------
    None.

    """
    if axl is None: axl = plt.gca()
    yinplmelt="value"
    if hue is None:
        if xl is None and not(var is None):
            dfl=pd.melt(pdf, value_vars=[var])
        elif not(xl is None) and var is None:
            dfl=pd.melt(pdf, id_vars=[xl])
        elif xl is None and var is None:
            dfl=pdf.copy()
            yinplmelt=var
        else:
            dfl=pd.melt(pdf, id_vars=[xl], value_vars=[var])
    else:
        if xl is None and not(var is None):
            dfl=pd.melt(pdf, id_vars=[hue], value_vars=[var])
        elif not(xl is None) and var is None:
            dfl=pd.melt(pdf, id_vars=[xl,hue])
        elif xl is None and var is None:
            dfl=pd.melt(pdf, id_vars=[hue])
        else:
            dfl=pd.melt(pdf, id_vars=[xl,hue], value_vars=[var])
        dfl[hue].replace(htirep,inplace=True)
    if not(xl is None):
        dfl[xl].replace(xltirep,inplace=True)
    if hue is None:
        axt = sns.boxplot(x=xl, y=yinplmelt, data=dfl, 
                          order=orderl, ax=axl, **bplkws)
        axt = sns.swarmplot(x=xl, y=yinplmelt,data=dfl, 
                            order=orderl,  ax=axl, **splkws)
    else:
        axt = sns.boxplot(x=xl, y=yinplmelt, hue=hue, data=dfl, 
                          order=orderl, hue_order=hue_order, ax=axl, **bplkws)
        axt = sns.swarmplot(x=xl, y=yinplmelt, hue=hue, data=dfl, 
                           order=orderl, hue_order=hue_order, ax=axl, **splkws)  
        h, l = axl.get_legend_handles_labels()
        axl.legend(h[0:dfl[hue].unique().size], l[0:dfl[hue].unique().size], title=hn)      
    axl.set_title(tl)
    axl.set_xlabel(xtxtl)
    axl.set_ylabel(ytxt)
    
#%% Einlesen und auswählen
#%%% Main
Version="240118"
ptype="TBT"
# ptype="ACT"
ptype="ATT"

img_p_mpath="D:/Gebhardt/Veröffentlichungen/2022-X-X_MatParams_Pelvis/IMG/04_build/"
img_p={'TBT':'fig2_cortical_location.png',
       'ACT':'fig2_trabecular_location.png',
       'ACT_dir':'fig2_trabecular_direction.png',
       'ATT_F':'fig2_fas.png',
       'ATT_L':'fig2_lig.png',}

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
    # YM_con=['inc','R','A0Al','meanwoso']
    YM_con=['lsq','R','A0Al','E']
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
if ptype=="TBT":  
    relist_woLR=[' L',' M',' R',
                 ' proximal',' distal',' ventral'] # Zu entfernende Worte
    relist=[' 4',' 5',' 1',' L',' R',' anterior superior',
            ' proximal',' distal',' ventral',
            ' anterior',' posterior',
            ' supraacetabular',' postacetabular'] # Zu entfernende Worte
    Locdict={'Ala ossis ilii superior':    'AOIlS',
             'Ala ossis ilii inferior':    'AOIlI',
             'Corpus ossis ilii':          'COIl',
             'Corpus ossis ischii':        'COIs',
             'Ramus superior ossis pubis': 'ROPu',
             'Ramus ossis ischii':         'ROIs',
             'Corpus vertebrae lumbales':  'CVLu'}
elif ptype=="ACT":  
    relist_woLR=[' L',' M',' R',
                 ' proximal',' distal',' ventral'] # Zu entfernende Worte
    relist=[' 4',' 5',' 1',' L',' R',' anterior superior',
            ' proximal',' distal',' ventral',
            ' anterior',' posterior',
            ' supraacetabular',' postacetabular'] # Zu entfernende Worte
    Locdict={'Ala ossis ilii inferior':    'AOIlI',
             'Corpus ossis ilii':          'COIl',
             'Corpus ossis ischii':        'COIs',
             'Corpus vertebrae lumbales':  'CVLu',
             'Corpus vertebrae sacrales':  'CVSa'}
elif ptype == "ATT":
    relist_woLR=[' L',' M',' R'] # Zu entfernende Worte
    relist=[' L',' M',' R',
            ' caudal',' medial',' lateral',' cranial',
            ' längs',' quer'] # Zu entfernende Worte
    Locdict_F={'Membrana obturatoria':                        'MObt',
               'Fascia glutea':                               'FGlu',
               # 'Fascia lumbalis':                             'FLum',
               'Fascia crescent':                             'FCre',
               'Fascia endopelvina':                          'FEnP',
               'Fascia thoracolumbalis lamina superficalis':  'FTLs',
               'Fascia thoracolumbalis lamina profunda':      'FTLp'}
    Locdict_L={'Ligamentum sacrotuberale':                    'SaTu',
               'Ligamentum sacrospinale':                     'SaSp',
               'Ligamentum iliolumbale':                      'IlLu',
               'Ligamentum inguinale':                        'Ingu',
               'Ligamentum pectineum':                        'Pect',
               'Ligamenta sacroiliaca anteriora':             'SaIla',
               'Ligamenta sacroiliaca posteriora':            'SaIlp',
               'Ligamentum sacroiliacum posterior longum':    'SaIll'}
    Locdict=pd.concat([pd.Series(Locdict_F),pd.Series(Locdict_L)]).to_dict()
else: print("Failure ptype!!!")

#%%% Output
out_full= os.path.abspath(path+name_out)
path_doda = os.path.abspath('F:/Messung/000-PARAFEMM_Patientendaten/PARAFEMM_Donordata_full.xlsx')
h5_conc = 'Summary'
h5_data = 'Test_Data'
VIParams = copy.deepcopy(VIParams_geo)
VIParams.extend(VIParams_mat)

MG_logopt['logfp']=open(out_full+'.log','w')
emec.output.str_log(name_out, **MG_logopt)
emec.output.str_log("\n   Paths:", **MG_logopt)
emec.output.str_log("\n   - in:", **MG_logopt)
emec.output.str_log("\n         {}".format(os.path.abspath(path+name_in+'.h5')), **MG_logopt)
emec.output.str_log("\n   - out:", **MG_logopt)
emec.output.str_log("\n         {}".format(out_full), **MG_logopt)
emec.output.str_log("\n   - donor:", **MG_logopt)
emec.output.str_log("\n         {}".format(path_doda), **MG_logopt)
# emec.output.str_log("\n   Donors:"+emec.output.str_indent('\n{}'.format(pd.Series(Donor_dict).to_string()),5), **MG_logopt)

#%%% Read
data_read = pd.HDFStore(path+name_in+'.h5','r')
dfa=data_read.select(h5_conc)
dft=data_read.select(h5_data)
data_read.close()
del dfa['No']
doda = pd.read_excel(path_doda, skiprows=range(1,2), index_col=0)

# Add Values
dfa.Failure_code  = emec.list_ops.list_cell_compiler(dfa.Failure_code)
dfa['statistics'] = emec.list_ops.list_interpreter(dfa.Failure_code, no_stats_fc)

h=dfa.Origin
for i in relist_woLR:
    h=h.str.replace(i,'')
dfa.insert(3,'Origin_woLRpd',h)
h=dfa.Origin
for i in relist:
    h=h.str.replace(i,'')
for i in h.index:
    if h[i].endswith(' '): h[i]=h[i][:-1]
h2=h.map(Locdict)
if (h2.isna()).any(): print('Locdict have missing/wrong values! \n   (Lnr: %s)'%['{:s}'.format(i) for i in h2.loc[h2.isna()].index])
dfa.insert(4,'Origin_short',h)
dfa.insert(5,'Origin_sshort',h2)
del h, h2

def containsetter(a, sd={' L':'L',' R':'R'}):
    out=''
    for i in sd.keys():
        if i in a:
            if out=='': 
                out = sd[i]
            else:
                out += sd[i]
    return out
tmp=dfa.Origin.apply(containsetter, sd={' L':'L',' R':'R'})
dfa.insert(6,'Side_LR',tmp)
if ptype == 'TBT':
    tmp=dfa.Origin.apply(containsetter, sd={' proximal':'p',' distal':'d'})
    dfa.insert(7,'Side_pd',tmp)
    
# ## Add Strain Energy Density
# tmp=dfa.loc(axis=1)[dfa.columns.str.startswith('W')].copy(deep=True)
# if ptype=="TBT":
#     tmp=tmp.div(dfa.thickness_2 * dfa.width_2 * dfa.Length_test,axis=0)*9
# else:
#     tmp=tmp.div(dfa.Area_CS * dfa.Length_test,axis=0)
# tmp.columns=tmp.columns.str.replace('W','U')
# dfa=pd.concat([dfa,tmp],axis=1)

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
agg_funcs=['count','mean',emec.stat_ext.meanwoso,'median',
           'std', emec.stat_ext.stdwoso,
           # Evac.coefficient_of_variation, Evac.coefficient_of_variation_woso,Evac.confidence_interval,
           'min','max',emec.stat_ext.CImin,emec.stat_ext.CImax]
cs_eva = emec.stat_ext.pd_agg(cs,agg_funcs,True)
cs_short_eva = emec.stat_ext.pd_agg(cs_short,agg_funcs,True)
if ptype=="ATT":
    tmp=pd.concat([cs_short,cs['Type']],axis=1).groupby('Type')
    c_short_Type_eva = emec.stat_ext.pd_agg(tmp,agg_funcs,True).stack()
tmp=pd.concat([cs_short,cs['Origin_short']],axis=1).groupby('Origin_short')
h_short_eva = emec.stat_ext.pd_agg(tmp,agg_funcs,True).stack()

if ptype=="ACT":
    # h_dir_eva=cs.groupby(['Origin_short','Direction_test'])[cs_num_cols].agg(agg_funcs)
    tmp=pd.concat([cs_short,cs['Origin_short']],axis=1).groupby(['Origin_short','Direction_test'])
    h_dir_eva_short = emec.stat_ext.pd_agg(tmp,agg_funcs,True)
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
    cs_E_lsq_m_sR=cs_E_lsq_m.loc(axis=1)[cs_E_lsq_m.columns.str.contains(r'^C',regex=True)]
    cs_E_lsq_m_sR=cs_E_lsq_m_sR.sub(cs_E_lsq_m['F'],axis=0)
    cs_E_lsq_m_sR.dropna(axis=0, inplace=True)
    cs_E_lsq_m_dR=cs_E_lsq_m.loc(axis=1)[cs_E_lsq_m.columns.str.contains(r'^C',regex=True)]
    cs_E_lsq_m_dR=(cs_E_lsq_m_dR.sub(cs_E_lsq_m['F'],axis=0)).div(cs_E_lsq_m['F'],axis=0)
    cs_E_lsq_m_dR.dropna(axis=0, inplace=True)
    
    cs_E_lsq_m_pR_rise = cs_E_lsq_m_pR.loc(axis=1)[cs_E_lsq_m_pR.columns.str.contains(r'^C.*\+$',regex=True)]
    cs_E_lsq_m_pR_fall = cs_E_lsq_m_pR.loc(axis=1)[cs_E_lsq_m_pR.columns.str.contains(r'^C.*\-$',regex=True)]
    cs_E_lsq_m_pR_rise.columns=cs_E_lsq_m_pR_rise.columns.str.replace('C','',regex=True).str.replace('\+','',regex=True)
    cs_E_lsq_m_pR_fall.columns=cs_E_lsq_m_pR_fall.columns.str.replace('C','',regex=True).str.replace('\-','',regex=True)

    cs_epl = cs.loc(axis=1)[cs.columns.str.startswith('epl')]
    cs_epl.columns = cs_epl.columns.str.split('_', expand=True)
    cs_epl_m=cs_epl.loc(axis=1)[idx['epl',:,:]].droplevel([0,1],axis=1)
    cs_epl_m.columns = cs_epl_m.columns.str.replace('C','',regex=True)

    cs_E_lsq_m_pR_rise_so=emec.stat_ext.stat_outliers(cs_E_lsq_m_pR_rise.mean(axis=1))
    cs_E_lsq_m_pR_fall_so=emec.stat_ext.stat_outliers(cs_E_lsq_m_pR_fall.mean(axis=1))
    cs_epl_m_so = emec.stat_ext.stat_outliers(cs_epl_m.mean(axis=1))
    cs_E_lsq_m_pR_rise_si=emec.stat_ext.stat_outliers(cs_E_lsq_m_pR_rise.mean(axis=1),out='inner')
    cs_E_lsq_m_pR_fall_si=emec.stat_ext.stat_outliers(cs_E_lsq_m_pR_fall.mean(axis=1),out='inner')
    cs_epl_m_si = emec.stat_ext.stat_outliers(cs_epl_m.mean(axis=1),out='inner')
    
    tmp = cs_E_lsq_m.copy()
    tmp.columns = 'E_con_'+tmp.columns
    tmp2 = cs_epl_m.copy()
    tmp2.columns = 'epl_con_'+tmp2.columns
    cs_cyc = pd.concat([cs_cfl,tmp2,tmp],axis=1)
    
#%%% Assessment Codes
emec.output.str_log("\n "+"="*100,**MG_logopt)
emec.output.str_log("\n Assessment Codes Analysies:",**MG_logopt)
# complete Codes
fc_b_df, fc_unique = emec.list_ops.Failure_code_bool_df(
    dfa.Failure_code, sep=',',level=2, strength=[1,2,3],
    drop_duplicates=True, sort_values=True, exclude =['nan'],
    replace_whitespaces=True, as_int=True
    )
fc_b_df_all_sum = fc_b_df.sum().sort_values(ascending=False)
fc_b_df_fail_sum = fc_b_df[dfa.statistics==False].sum().sort_values(ascending=False)
fc_b_df_nofail_sum = fc_b_df[dfa.statistics==True].sum().sort_values(ascending=False)

txt='Assessment codes frequency:'
emec.output.str_log(emec.output.str_indent(txt),**MG_logopt)
txt='- all:\n   %s'%fc_b_df_all_sum[fc_b_df_all_sum!=0].to_dict()
emec.output.str_log(emec.output.str_indent(txt,5),**MG_logopt)
txt='- fail:\n   %s'%fc_b_df_fail_sum[fc_b_df_fail_sum!=0].to_dict()
emec.output.str_log(emec.output.str_indent(txt,5),**MG_logopt)
txt='- no fail:\n   %s'%fc_b_df_nofail_sum[fc_b_df_nofail_sum!=0].to_dict()
emec.output.str_log(emec.output.str_indent(txt,5),**MG_logopt)
    
#%% Data-Export
writer = pd.ExcelWriter(out_full+'.xlsx', engine = 'xlsxwriter')
# tmp=dft_comb_rel.rename(VIPar_plt_renamer,axis=1)
dfa.to_excel(writer, sheet_name='Data-All')
tmp1=['Date_test','Temperature_test','Humidity_test','Failure_code','statistics','Date_eva']
tmp=dfa[VIParams_gen + VIParams_geo + tmp1 + VIParams_mat].copy()
tmp.rename(columns=VIParams_rename,inplace=True)
tmp.to_excel(writer, sheet_name='Data')
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
emec.output.str_log("\n\n "+"="*100, **MG_logopt)
emec.output.str_log("\n Statistical tests ", **MG_logopt)
#%%% Set
alpha=0.05
# stat_ttype_parametric=True # Testtype
stat_ttype_parametric=False # Testtype

if stat_ttype_parametric:
    mcomp_ind="ttest_ind"
    mcomp_rel="ttest_rel"
    mcomp_ph="TukeyHSD"
    mpop="ANOVA"
    # Tukey HSD test:
    MComp_kws={'do_mcomp_a':2, 'mcomp':mcomp_ind, 'mpadj':'', 
                'Ffwalpha':1, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    MCompdf_kws={'do_mcomp_a':1, 'mcomp':mcomp_ph, 'mpadj':'holm', 
                 'Ffwalpha':1, 'mkws':{}, 'check_resnorm':True}
    mcorr="pearson"
else:
    mcomp_ind="mannwhitneyu"
    mcomp_rel="wilcoxon"
    # mcomp_ph="mannwhitneyu"
    mcomp_ph="Dunn"
    mpop="Kruskal-Wallis H-test"
    # Mann-Whitney U test: (two independent)
    MComp_kws={'do_mcomp_a':2, 'mcomp':mcomp_ph, 'mpadj':'holm', 
                'Ffwalpha':1, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    MCompdf_kws={'do_mcomp_a':1, 'mcomp':mcomp_ph, 'mpadj':'holm', 
                 'Ffwalpha':1, 'mkws':{}, 'check_resnorm':True}
    mcorr="spearman"

#%%% Distribution
emec.output.str_log("\n\n "+"-"*100, **MG_logopt)
emec.output.str_log("\n Distribution tests: (HO=sample looks Gaussian)", 
                    **MG_logopt)
tmp = emec.stat_ext.Dist_test_multi(cs_short.loc(axis=1)[css_ncols],
                                    alpha=alpha)
emec.output.str_log(emec.output.str_indent(tmp.to_string()), **MG_logopt)

#%%% Additional Logging
if ptype == "ATT":
    txt="\n "+"="*100
    txt+=("\n Cyclic-loading influence: (%d samples)"%cs.OPT_pre_load_cycles[cs.OPT_pre_load_cycles>0].size)
    txt += emec.output.str_indent('- cyclic loading stress level (related to ultimate strength):')
    txt += emec.output.str_indent('- preload (aim: 0.10):',6)
    tmp=cs_cfl.cyc_f_lo.agg(agg_funcs)[['mean','std','min','max','CImin','CImax']]
    txt += emec.output.str_indent('  {0:.5f} ± {1:.5f} ({2:.5f}-{3:.5f}, CI: {4:.5f}-{5:.5f})'.format(*tmp),6)
    txt += emec.output.str_indent('- cyclic (aim: 0.30):',6)
    tmp=cs_cfl.cyc_f_hi.agg(agg_funcs)[['mean','std','min','max','CImin','CImax']]
    txt += emec.output.str_indent('  {0:.5f} ± {1:.5f} ({2:.5f}-{3:.5f}, CI: {4:.5f}-{5:.5f})'.format(*tmp),6)
    txt += emec.output.str_indent('- cyclic related to final Youngs Modulus:')
    txt += emec.output.str_indent('- loading (ascending):',6)
    txt += emec.output.str_indent(cs_E_lsq_m_pR_rise.agg(agg_funcs).T.to_string(),9)
    txt += emec.output.str_indent('Hypothesis test equality:',9)
    tmp=cs_E_lsq_m.dropna(axis=0)
    tmp=tmp.loc(axis=1)['R':].apply(lambda x: stats.wilcoxon(x,tmp['F']))
    tmp.index=['F','p']
    tmp.loc['H0']=tmp.loc['p'].apply(lambda x: False if x <= alpha else True)
    tmp1=tmp.columns.str.contains(r'^C.*\-$',regex=True)
    txt += emec.output.str_indent(tmp.loc(axis=1)[~tmp1].to_string(),9)
    txt += emec.output.str_indent('statistical outliers: %d'%len(cs_E_lsq_m_pR_rise_so),9)
    txt += emec.output.str_indent(cs.Failure_code[cs_E_lsq_m_pR_rise_so],9)
    txt += emec.output.str_indent('- unloading (descending):',6)
    txt += emec.output.str_indent(cs_E_lsq_m_pR_fall.agg(agg_funcs).T.to_string(),9)
    txt += emec.output.str_indent('Mean over all:',9)
    txt += emec.output.str_indent(cs_E_lsq_m_pR_fall.stack().agg(agg_funcs).to_string(),9)
    txt += emec.output.str_indent('Hypothesis test equality:',9)
    txt += emec.output.str_indent(tmp.loc(axis=1)[tmp1].to_string(),9)
    txt += emec.output.str_indent('statistical outliers: %d'%len(cs_E_lsq_m_pR_fall_so),9)
    txt += emec.output.str_indent(cs.Failure_code[cs_E_lsq_m_pR_fall_so],12)
    txt += emec.output.str_indent('- plastic strain after cycle:')
    txt += emec.output.str_indent(cs_epl_m.agg(agg_funcs).T.to_string(),9)
    txt += emec.output.str_indent('statistical outliers: %d'%len(cs_epl_m_so),9)
    txt += emec.output.str_indent(cs.Failure_code[cs_epl_m_so],12)
    emec.output.str_log(txt,**MG_logopt)
    
#%%% Variance analyses
emec.output.str_log("\n\n "+"-"*100, **MG_logopt)
emec.output.str_log("\n %s-Harvesting location: (Groups are significantly different for p < %.3f)"%(mpop,alpha),**MG_logopt)
tmp=pd.concat([cs_short,cs['Origin_short']],axis=1)
tmp=emec.stat_ext.Multi_conc(df=tmp,group_main='Origin_short', anat='VAwoSg',
               stdict=css_ncols.to_series().to_dict(), 
               met=mpop, alpha=alpha, kws=MCompdf_kws)
emec.output.str_log(emec.output.str_indent(tmp.loc(axis=1)['DF1':'H0'].to_string()),**MG_logopt)
emec.output.str_log("\n  -> Multicomparision (%s)):"%MComp_kws['mcomp'],**MG_logopt)
for i in tmp.loc[tmp.H0 == False].index:
    txt="{}:\n{}".format(i,tmp.loc[i,'MCP'],)
    emec.output.str_log(emec.output.str_indent(txt,5),**MG_logopt)
emec.output.str_log("\n\n   -> Multicomparison relationship interpretation:",**MG_logopt)
tmp2=tmp.loc[tmp.H0 == False]['MCP'].apply(emec.stat_ext.MComp_interpreter)
tmp2=tmp2.droplevel(0).apply(pd.Series)[0].apply(pd.Series).T
emec.output.str_log(emec.output.str_indent(tmp2.to_string(),5),**MG_logopt)

emec.output.str_log("\n\n "+"-"*100, **MG_logopt)
emec.output.str_log("\n %s-Donor: (Groups are significantly different for p < %.3f)"%(mpop,alpha),**MG_logopt)
tmp=emec.stat_ext.Multi_conc(df=cs_short,group_main='Donor', anat='VAwoSg',
               stdict=css_ncols.to_series().to_dict(), 
               met=mpop, alpha=alpha, kws=MCompdf_kws)
emec.output.str_log(emec.output.str_indent(tmp.loc(axis=1)['DF1':'H0'].to_string()),**MG_logopt)
emec.output.str_log("\n  -> Multicomparision (%s)):"%MComp_kws['mcomp'],**MG_logopt)
for i in tmp.loc[tmp.H0 == False].index:
    txt="{}:\n{}".format(i,tmp.loc[i,'MCP'],)
    emec.output.str_log(emec.output.str_indent(txt,5),**MG_logopt)
emec.output.str_log("\n\n   -> Multicomparison relationship interpretation:",**MG_logopt)
tmp2=tmp.loc[tmp.H0 == False]['MCP'].apply(emec.stat_ext.MComp_interpreter)
tmp2=tmp2.droplevel(0).apply(pd.Series)[0].apply(pd.Series).T
emec.output.str_log(emec.output.str_indent(tmp2.to_string(),5),**MG_logopt)

#%%% Hyphotesis tests
#%%%% Side Left Right
emec.output.str_log("\n\n "+"-"*100, **MG_logopt)
emec.output.str_log("\n Hypothesis test - body side (left/right): (significantly different for p < %.3f)"%(alpha),**MG_logopt)
emec.output.str_log("\n    (%s: all values, %s: Only values which are available at donor and location on both sides)"%(mcomp_ind, mcomp_rel),**MG_logopt)
if ptype == 'TBT':
    stat_dd_ind='raise'
    stat_dd_rel=['Donor','Origin_woLRpd','Side_pd']
    # stat_dd_ind='mean'
    # stat_dd_rel=['Donor','Origin_short','Side_pd']
elif ptype == 'ACT':
    stat_dd_ind='raise'
    stat_dd_rel=['Donor','Origin_woLRpd','Direction_test']
    # stat_dd_ind='mean'
    # stat_dd_rel=['Donor','Origin_short','Direction_test']
elif ptype == 'ATT':
    stat_dd_ind='mean'
    stat_dd_rel=['Donor','Origin_short']
else:
    stat_dd_ind='raise'
# tmp=pd.concat([cs_short,cs[['Side_LR']+stat_dd_rel]],axis=1)
# tmp=tmp.query("Side_LR =='L' or Side_LR =='R'")
tmp=cs.query("Side_LR =='L' or Side_LR =='R'")
tmp1=emec.stat_ext.Hypo_test_multi(tmp, group_main='Side_LR', group_sub=None, 
                     ano_Var=VIParams_geo+VIParams_mat, 
                     rel=False, rel_keys=[], 
                     mcomp=mcomp_ind,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
tmp2=emec.stat_ext.Hypo_test_multi(tmp, group_main='Side_LR', group_sub=None, 
                     ano_Var=VIParams_geo+VIParams_mat, 
                     rel=True, rel_keys=stat_dd_rel, 
                     mcomp=mcomp_rel,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
tmp3=pd.concat([tmp1, tmp2], axis=1, keys=[mcomp_ind, mcomp_rel])
emec.output.str_log(emec.output.str_indent(tmp3.to_string()),**MG_logopt)

if ptype == 'ATT':
    emec.output.str_log("\n Hypothesis test - type (Fascia/Ligament): (%s, significantly different for p < %.3f)"%(mcomp_ind,alpha),**MG_logopt)
    tmp1=emec.stat_ext.Hypo_test_multi(cs, group_main='Type', group_sub=None, 
                     ano_Var=VIParams_geo+VIParams_mat, 
                     rel=False, rel_keys=[], 
                     mcomp=mcomp_ind,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    emec.output.str_log(emec.output.str_indent(tmp1.to_string()),**MG_logopt)

#%%%% Side proximal distal (TBT only)
if ptype == 'TBT':
    emec.output.str_log("\n\n "+"-"*100, **MG_logopt)
    emec.output.str_log("\n Hypothesis test - body side (proximal/distal): (significantly different for p < %.3f)"%(alpha),**MG_logopt)
    emec.output.str_log("\n    (%s: all values, %s: Only values which are available at donor and location on both sides)"%(mcomp_ind, mcomp_rel),**MG_logopt)
    # tmp=pd.concat([cs_short,cs[['Side_pd','Donor','Origin_short']]],axis=1)
    tmp=cs.query("Side_pd =='p' or Side_pd =='d'")
    tmp1=emec.stat_ext.Hypo_test_multi(tmp, group_main='Side_pd', group_sub=None, 
                         ano_Var=VIParams_geo+VIParams_mat, 
                         rel=False, rel_keys=[], 
                         mcomp=mcomp_ind,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    tmp2=emec.stat_ext.Hypo_test_multi(tmp, group_main='Side_pd', group_sub=None, 
                         ano_Var=VIParams_geo+VIParams_mat, 
                         rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                         mcomp=mcomp_rel,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    tmp3=pd.concat([tmp1, tmp2], axis=1, keys=[mcomp_ind, mcomp_rel])
    emec.output.str_log(emec.output.str_indent(tmp3.to_string()),**MG_logopt)
    
#%%%% Direction of test (ACT only)
if ptype == 'ACT':
    emec.output.str_log("\n\n "+"-"*100, **MG_logopt)
    emec.output.str_log("\n Hypothesis test - direction (x/y/z): (significantly different for p < %.3f)"%(alpha),**MG_logopt)
    emec.output.str_log("\n    (%s: all values, %s: Only values which are available at donor and location on both sides)"%(mcomp_ind, mcomp_rel),**MG_logopt)
    tmp=emec.stat_ext.Multi_conc(df=cs,group_main='Direction_test', anat='VAwoSg',
               stdict=pd.Series(VIParams_geo+VIParams_mat, index=VIParams_geo+VIParams_mat).to_dict(), 
               met=mpop, alpha=alpha, kws=MCompdf_kws)
    emec.output.str_log(emec.output.str_indent(tmp.loc(axis=1)['DF1':'H0'].to_string()),**MG_logopt)
    # tmp=pd.concat([cs_short,cs[['Direction_test','Donor','Origin_short']]],axis=1)
    emec.output.str_log("\n  - x to y:",**MG_logopt)
    tmp4=cs.query("Direction_test =='x' or Direction_test =='y'")
    tmp1=emec.stat_ext.Hypo_test_multi(tmp4, group_main='Direction_test', group_sub=None, 
                         ano_Var=VIParams_geo+VIParams_mat, 
                         rel=False, rel_keys=[], 
                         mcomp=mcomp_ind,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    tmp2=emec.stat_ext.Hypo_test_multi(tmp4, group_main='Direction_test', group_sub=None, 
                         ano_Var=VIParams_geo+VIParams_mat, 
                         rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                         mcomp=mcomp_rel,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    tmp3=pd.concat([tmp1, tmp2], axis=1, keys=[mcomp_ind, mcomp_rel])
    emec.output.str_log(emec.output.str_indent(tmp3.to_string()),**MG_logopt)
    emec.output.str_log("\n  - x to z:",**MG_logopt)
    tmp4=cs.query("Direction_test =='x' or Direction_test =='z'")
    tmp1=emec.stat_ext.Hypo_test_multi(tmp4, group_main='Direction_test', group_sub=None, 
                         ano_Var=VIParams_geo+VIParams_mat, 
                         rel=False, rel_keys=[], 
                         mcomp=mcomp_ind,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    tmp2=emec.stat_ext.Hypo_test_multi(tmp4, group_main='Direction_test', group_sub=None, 
                         ano_Var=VIParams_geo+VIParams_mat, 
                         rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                         mcomp=mcomp_rel,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    tmp3=pd.concat([tmp1, tmp2], axis=1, keys=[mcomp_ind, mcomp_rel])
    emec.output.str_log(emec.output.str_indent(tmp3.to_string()),**MG_logopt)
    emec.output.str_log("\n  - y to z:",**MG_logopt)
    tmp4=cs.query("Direction_test =='y' or Direction_test =='z'")
    tmp1=emec.stat_ext.Hypo_test_multi(tmp4, group_main='Direction_test', group_sub=None, 
                         ano_Var=VIParams_geo+VIParams_mat, 
                         rel=False, rel_keys=[], 
                         mcomp=mcomp_ind,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    tmp2=emec.stat_ext.Hypo_test_multi(tmp4, group_main='Direction_test', group_sub=None, 
                         ano_Var=VIParams_geo+VIParams_mat, 
                         rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                         mcomp=mcomp_rel,  alpha=alpha, deal_dupl_ind=stat_dd_ind)
    tmp3=pd.concat([tmp1, tmp2], axis=1, keys=[mcomp_ind, mcomp_rel])
    emec.output.str_log(emec.output.str_indent(tmp3.to_string()),**MG_logopt)
    
    emec.output.str_log("\n  - Corpus vertebrae lumbales (CVLu, x/z equal/greater y):",**MG_logopt)
    tmp1=pd.DataFrame([])
    tmp2=cs.query("Origin_sshort=='CVLu'")
    tmp1['yx-wi-eq']=emec.stat_ext.Hypo_test(tmp2, groupby='Direction_test',
                        ano_Var=YM_con_str, mkws={'alternative':'two-sided'},
                        rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                        mcomp=mcomp_rel,  alpha=alpha,
                        deal_dupl_ind=stat_dd_ind, group_ord=['y','x'], add_out='Series')
    tmp1['yx-wi-gr']=emec.stat_ext.Hypo_test(tmp2, groupby='Direction_test',
                        ano_Var=YM_con_str, mkws={'alternative':'greater'},
                        rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                        mcomp=mcomp_rel,  alpha=alpha,
                        deal_dupl_ind=stat_dd_ind, group_ord=['y','x'], add_out='Series')
    tmp1['yz-wi-eq']=emec.stat_ext.Hypo_test(tmp2, groupby='Direction_test',
                        ano_Var=YM_con_str, mkws={'alternative':'two-sided'},
                        rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                        mcomp=mcomp_rel,  alpha=alpha,
                        deal_dupl_ind=stat_dd_ind, group_ord=['y','z'], add_out='Series')
    tmp1['yz-wi-gr']=emec.stat_ext.Hypo_test(tmp2, groupby='Direction_test',
                        ano_Var=YM_con_str, mkws={'alternative':'greater'},
                        rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                        mcomp=mcomp_rel,  alpha=alpha,
                        deal_dupl_ind=stat_dd_ind, group_ord=['y','z'], add_out='Series')      
    emec.output.str_log(emec.output.str_indent(tmp1.to_string()),**MG_logopt)
    emec.output.str_log("\n  - Corpus vertebrae sacrales (CVSa, x/y equal/greater z):",**MG_logopt)
    tmp1=pd.DataFrame([])
    tmp2=cs.query("Origin_sshort=='CVSa'")
    tmp1['zx-wi-eq']=emec.stat_ext.Hypo_test(tmp2, groupby='Direction_test',
                        ano_Var=YM_con_str, mkws={'alternative':'two-sided'},
                        rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                        mcomp=mcomp_rel,  alpha=alpha,
                        deal_dupl_ind=stat_dd_ind, group_ord=['z','x'], add_out='Series')
    tmp1['zx-wi-le']=emec.stat_ext.Hypo_test(tmp2, groupby='Direction_test',
                        ano_Var=YM_con_str, mkws={'alternative':'greater'},
                        rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                        mcomp=mcomp_rel,  alpha=alpha,
                        deal_dupl_ind=stat_dd_ind, group_ord=['z','x'], add_out='Series')
    tmp1['zy-wi-eq']=emec.stat_ext.Hypo_test(tmp2, groupby='Direction_test',
                        ano_Var=YM_con_str, mkws={'alternative':'two-sided'},
                        rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                        mcomp=mcomp_rel,  alpha=alpha,
                        deal_dupl_ind=stat_dd_ind, group_ord=['z','y'], add_out='Series')
    tmp1['zy-wi-le']=emec.stat_ext.Hypo_test(tmp2, groupby='Direction_test',
                        ano_Var=YM_con_str, mkws={'alternative':'greater'},
                        rel=True, rel_keys=['Donor','Origin_woLRpd','Side_LR'], 
                        mcomp=mcomp_rel,  alpha=alpha,
                        deal_dupl_ind=stat_dd_ind, group_ord=['z','y'], add_out='Series')    
    emec.output.str_log(emec.output.str_indent(tmp1.to_string()),**MG_logopt)
    
#%%% Correlations
cs_short_corr = cs_short.corr(method=mcorr)

doda_corr_d=doda.loc(axis=1)[VIParams_don]
doda_corr_d['Sex'] = doda_corr_d['Sex'].map({'f':1,'m':-1})
doda_corr_d_num_cols=doda_corr_d.select_dtypes(include=['int','float']).columns
doda_corr_i= emec.list_ops.ICD_bool_df(doda.ICDCodes,**{'level':0})
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

emec.output.str_log("\n\n "+"-"*100, **MG_logopt)
emec.output.str_log("\n Linear regressions:", **MG_logopt)
emec.output.str_log(('').join(txt.values), **MG_logopt)

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
        with get_sample_data(img_p_mpath+img_p['TBT']) as file:
            arr_img = plt.imread(file)
    elif ptype == "ACT":
        pltvartmp=YM_con_str
        with get_sample_data(img_p_mpath+img_p['ACT']) as file:
            arr_img = plt.imread(file)
    imagebox = OffsetImage(arr_img, zoom=0.16)
    imagebox.image.axes = ax['Pelvis']
    ab = AnnotationBbox(imagebox, (0,0),
                        xybox=(0.4, 0.5),
                        xycoords='axes fraction',frameon=False)
    ax['Pelvis'].add_artist(ab)
    ax['Pelvis'].grid(False)
    ax['Pelvis'].axis('off')
    
    df=pd.melt(cs, id_vars=['Origin_sshort'], value_vars=[pltvartmp])
    axt = sns.boxplot(x="Origin_sshort", y="value", 
                      data=df, ax=ax['Location'], 
                      order=Locdict.values(),
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax['Location'], 
                        order=Locdict.values(),
                        dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax['Location'].set_title('By harvesting region')
    ax['Location'].set_xlabel('Region')
    ax['Location'].set_ylabel('')
    ax['Location'].tick_params(axis='y',which='both',left=False,labelleft=False)
    plt_add_figsubl(text='b', axo=ax['Location'],**subfiglabeldict)
    
    df=pd.melt(cs, id_vars=['Donor'], value_vars=[pltvartmp])
    df.Donor.replace(doda.Naming,inplace=True)
    axt = sns.boxplot(x="Donor", y="value", 
                      data=df, ax=ax['Donor'], 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Donor", y="value",
                        data=df, ax=ax['Donor'], 
                        dodge=True, edgecolor="black",
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
    emec.plotting.plt_handle_suffix(fig,path=path+"Paper-OV",**plt_Fig_dict)

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
    with get_sample_data(img_p_mpath+img_p['ATT_F']) as file:
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
                      order=Locdict_F.values(),
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Origin_sshort", y="value", data=df, ax=ax['LocationF'], 
                        order=Locdict_F.values(),
                        dodge=True, edgecolor="black",
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
    
    
    with get_sample_data(img_p_mpath+img_p['ATT_L']) as file:
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
                      order=Locdict_L.values(),
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                 "markeredgecolor":"black", "markersize":"12",
                                                 "alpha":0.75})
    axt = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax['LocationL'], 
                        order=Locdict_L.values(),
                        dodge=True, edgecolor="black",
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
    axt = sns.swarmplot(x="Donor", y="value", data=df, ax=ax['DonorL'], 
                        dodge=True, edgecolor="black",
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
    emec.plotting.plt_handle_suffix(fig,path=path+"Paper-OVS",**plt_Fig_dict)


#%%%% Directions ACT
if ptype == "ACT":
    gs_kw = dict(width_ratios=[0.75, 0.25], height_ratios=[1, 1])
    fig, ax = plt.subplot_mosaic([['E', 'Pelvis'],
                                  ['fu','Pelvis']],
                                  gridspec_kw=gs_kw,
                                  # empty_sentinel='lower mid',
                                  figsize=(16/2.54, 12/2.54),
                                  constrained_layout=True)
    with get_sample_data(img_p_mpath+img_p['ACT_dir']) as file:
        arr_img = plt.imread(file)
    imagebox = OffsetImage(arr_img, zoom=0.28)
    imagebox.image.axes = ax['Pelvis']
    ab = AnnotationBbox(imagebox, xy = (-0.09,-0.05), box_alignment=(0, 0),
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
    fig.legend(handles[0:3], ['$x$','$y$','$z$'], 
               ncol=3, columnspacing=0.5, handlelength=1.5,
               title="Direction", 
               loc="upper right", bbox_to_anchor=(0.98, 0.918))
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
                       data=df, ax=ax['fu'], 
                       # palette={'x':'r','y':'g','z':'b'},
                       palette={'x':sns.color_palette()[3],
                                'y':sns.color_palette()[2],
                                'z':sns.color_palette()[0]},
                       dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
    ax['fu'].set_title('Ultimate strength by harvesting region and testing direction')
    ax['fu'].set_xlabel('Region')
    ax['fu'].set_ylabel('Ultimate strength in MPa')
    ax['fu'].sharey(ax['fu'])
    ax['fu'].legend().set_visible(False)    
    plt_add_figsubl(text='b', axo=ax['fu'],**subfiglabeldict)
    fig.suptitle(None)
    emec.plotting.plt_handle_suffix(fig,path=path+"Paper-Dir",**plt_Fig_dict)
    
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
    ax['slvl'].set_title('Loading elastic modulus ratio vs. cyclic stress level')
    ax['slvl'].set_xlabel('Ratio of upper cyclic to ultimate stress')
    # ax['slvl'].set_ylabel('Cyclic related to\nfinal elastic modulus')
    ax['slvl'].set_ylabel('Elastic modulus ratio')
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
                 label='Unloading',ax=ax['Ecyc'])
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
    ax['Ecyc'].set_title('Elastic modulus ratio and plastic strain vs. applied cycles')
    ax['Ecyc'].set_xlabel('Cycle')
    ax['Ecyc'].set_xticklabels(cs_epl_m.columns.str.replace('epl_con_',''))
    # ax['Ecyc'].set_ylabel('Cyclic related to\nfinal elastic modulus')
    ax['Ecyc'].set_ylabel('Elastic modulus ratio')
    ax2.set_ylabel('Plastic strain')
    ax2.legend(ln,la,loc='center right')
    plt_add_figsubl(text='b', axo=ax['Ecyc'],**subfiglabeldict)
    fig.suptitle(None)
    emec.plotting.plt_handle_suffix(fig,path=path+"Paper-Cyc",**plt_Fig_dict)   

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
emec.plotting.plt_handle_suffix(fig,path=path+"SM-AC",**plt_Fig_dict)

#%%%% Data visualization
if ptype == 'TBT':
    subj='0000-0001-8378-3108_LEIULANA_60-17_LuPeCo_cl32b'
    epsu="Strain_opt_d_M"
    Viupu='VIP_d'
    leps="leos_opt"
    YMu=YM_opt_str
    ind_E1='E'
    ind_E2='E'
    ind_E3='B'
    Vsr={'F3':'L'}
    Vs1=['S','L','YK','Y','U','B','E','SU','SE']
    Vs2=['S','L','YK','Y0','Y1','Y2','Y','U','B','E']
    Vs3=['L','Y','U','B']
elif ptype == 'ACT':
    subj='0000-0001-8378-3108_LEIULANA_60-17_LuPeCo_tm21y'
    epsu="Strain"
    Viupu='VIP_m'
    leps="leos_con"
    YMu=YM_con_str
    ind_E1='SE'
    ind_E2='E'
    ind_E3='B'
    Vsr={'F3':'L'}
    Vs1=['S','L','YK','Y','U','B','E','SU','SE']
    Vs2=['S','L','YK','Y0','Y1','Y2','Y','U','B','E']
    Vs3=['L','Y','U','B']
elif ptype == 'ATT':
    subj='0000-0001-8378-3108_LEIULANA_60-17_LuPeCo_sl03a'
    epsu="Strain"
    Viupu='VIP_m'
    leps="leos_con"
    YMu=YM_con_str
    ind_E1='E'
    ind_E2='E'
    ind_E3='B'
    Vsr={'F3':'L'}
    # Vs1=['S','L','YK','Y','U','B','E','SU','SE']
    Vs1=['S','C1+', 'C1-', 'C2+', 'C2-', 'C3+', 'C3-', 'C4+', 'C4-', 'C5+',
       'C5-', 'C6+', 'C6-', 'C7+', 'C7-', 'C8+', 'C8-', 'C9+', 'C9-', 'C10+',
       'C10-','L','YK','Y','U','B','E','SU','SE']
    Vs2=['S','L','YK','Y0','Y1','Y2','Y','U','B','E']
    Vs3=['L','Y','U','B']
tmp=dft[subj]
tmp2=VIP_rebuild(tmp[Viupu])
tmp2=tmp2.rename(Vsr)
tmp3=dfa.loc[subj]
tmp4=tmp[[epsu,'Stress']].copy()
tmp4[epsu]=tmp4[epsu]-tmp3.loc[leps]
tmpf,_=emec.mc_man.DetFinSSC(mdf=tmp, YM=tmp3.loc[leps], 
                       iS=tmp2['L'], iLE=None,
                       StressN='Stress', StrainN=epsu, 
                       addzero=True, izero=0, option='SO')
colp=sns.color_palette()
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
tmp21=tmp2.loc[tmp2.index.to_series().apply(lambda x: x in Vs1)]
ax['fotot'].plot(tmp.Time[:tmp21[ind_E1]], tmp.Force[:tmp21[ind_E1]], 
                 '-', color=colp[0], label='Measurement')
a, b=tmp.Time[tmp21[:ind_E1]],tmp.Force[tmp21[:ind_E1]]
ax['fotot'].plot(a, b, 'x', color=colp[3], label='Points of interest')
j=np.int64(-1)
for x in tmp21.index:
    j+=1
    if j%2: c=(8,-8)
    else:   c=(-8,8)
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
# ax['fotot'].legend(ln,la,loc='lower right')
ax['fotot'].legend(ln, la, loc='lower center', bbox_to_anchor=(0.7, 0.0))

ax['sigeps'].set_title('Stress-strain-curve with labels')
ax['sigeps'].set_xlabel('Strain')
ax['sigeps'].set_ylabel('Stress in MPa')
tmp22=tmp2.loc[tmp2.index.to_series().apply(lambda x: x in Vs2)]
ax['sigeps'].plot(tmp4.loc[:tmp22[ind_E2]][epsu],
                  tmp4.loc[:tmp22[ind_E2]]['Stress'], 
                  '-', color=colp[0], label='Measurement')
ax['sigeps'].plot([0,tmp4.loc[tmp22['YK']][epsu]], 
                  [0,tmp4.loc[tmp22['YK']]['Stress']], 
                  'g-',label='Elastic modulus')
a, b=tmp4[epsu][tmp22],tmp4.Stress[tmp22]
j=np.int64(-1)
ax['sigeps'].plot(a, b, 'x', color=colp[3], label='Points of interest')
for x in tmp22.index:
    j+=1
    if j%2: c=(8,-8)
    else:   c=(-8,8)
    ax['sigeps'].annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                  xytext=c, ha="center", va="center", textcoords='offset points')
ax['sigeps'].legend(loc='lower center', bbox_to_anchor=(0.7, 0.0))

ax['sigepsf'].set_title('Stress-strain-curve (start linearized)')
ax['sigepsf'].set_xlabel('Strain')
ax['sigepsf'].set_ylabel('Stress in MPa')
tmp23=tmp2.loc[tmp2.index.to_series().apply(lambda x: x in Vs3)]
ax['sigepsf'].plot(tmpf.loc[:tmp23[ind_E3]][epsu],
                   tmpf.loc[:tmp23[ind_E3]]['Stress'], 
                   '-', color=colp[0],label='Measurement')
ax['sigepsf'].plot(tmpf.iloc[:2][epsu],
                   tmpf.iloc[:2]['Stress'], 
                   '-', color=colp[2],label='Linearization')
ax['sigepsf'].fill_between(tmpf.loc[:tmp23['Y']][epsu],
                           tmpf.loc[:tmp23['Y']]['Stress'],
                            **dict(color=colp[1], hatch='|||',
                            edgecolor='black', alpha= 0.2, lw=0), 
                            label='$U_{y}$')
ax['sigepsf'].fill_between(tmpf.loc[:tmp23['U']][epsu],
                           tmpf.loc[:tmp23['U']]['Stress'],
                            **dict(color=colp[1], hatch='///',
                            edgecolor='black', alpha= 0.2, lw=0),  
                            label='$U_{u}$')
ax['sigepsf'].fill_between(tmpf.loc[:tmp23['B']][epsu],
                           tmpf.loc[:tmp23['B']]['Stress'],
                            **dict(color=colp[1], hatch='...',
                            edgecolor='black', alpha= 0.2, lw=0),  
                            label='$U_{b}$')
a, b=tmpf[epsu][tmp23[:ind_E3]],tmp.Stress[tmp23[:ind_E3]]
j=np.int64(-1)
ax['sigepsf'].plot(a, b, 'x', color=colp[3], label='Points of interest')
for x in tmp23[:ind_E3].index:
    j+=1
    if j%2: c=(-8,8)
    else:   c=(-8,8)
    ax['sigepsf'].annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), 
                            xycoords='data', xytext=c, 
                            ha="center", va="center", 
                            textcoords='offset points')
# ax['sigepsf'].legend(ncol=2, loc='upper left', columnspacing=0.6)
ax['sigepsf'].legend(ncol=2, loc='lower center', columnspacing=0.6)
fig.suptitle(None)
emec.plotting.plt_handle_suffix(fig,path=path+"SM-DV",**plt_Fig_dict)

#%%%% DV - DIC
if ptype == 'ACT':
    subj='0000-0001-8378-3108_LEIULANA_48-17_LuPeCo_tl22y'
    imgp='F:/Messung/007-201013_Becken5-ADV/Messdaten/DIC/Export/sl22y-Cam/'
    tmp=dft[subj]
    gs_kw = dict(width_ratios=[1,1,1,1,1], height_ratios=[1, 1, 1, 1],
                     wspace=0.1, hspace=0.1)
    tmp1=pd.DataFrame({
        'Name':['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'],
        'Time':[19.9,25,30,35,40,47,50,57.2,65,70,75,80,90,100,112.6]
        })
    tmp1.index=tmp.loc[tmp.Time.apply(lambda x: x in tmp1.Time.values)].index
    
    fig, ax = plt.subplot_mosaic([['FT','FT','FT','FT','FT'],
                                  tmp1.Name[:5].values,
                                  tmp1.Name[5:10].values,
                                  tmp1.Name[10:15].values],
                                  gridspec_kw=gs_kw,
                                  figsize=figsize_sup,
                                  constrained_layout=True)
    ax['FT'].set_title('Measured force versus time')
    ax['FT'].set_xlabel('Time in s')
    ax['FT'].set_ylabel('Force in N')
    ax['FT'].plot(tmp.Time, tmp.Force, 
                  '-', color=colp[0], label='Measurement')
    a, b=tmp.Time[tmp1.index],tmp.Force[tmp1.index]
    ax['FT'].plot(a, b, 'x', color=colp[3], label='Points of interest')
    j=0
    for i in tmp1.index:
        tmp2=tmp1.Name[i]
        if j%2: c=(0,8)
        else:   c=(0,8)
        ax['FT'].annotate('%s' % tmp2, 
                         xy=(a.iloc[j],b.iloc[j]), 
                         xycoords='data', xytext=c, 
                         ha="center", va="center", 
                         textcoords='offset points')
        imgpt=imgp+'cam_0_step_{}.tiff'.format(
            "{:3d}".format(int(tmp1.Time[i]/.25))
            )
        ax[tmp2].imshow(plt.imread(imgpt), cmap=plt.cm.gray)
        ax[tmp2].set_title("{} ({:.2f} s)".format(tmp2, tmp1.Time[i]))
        ax[tmp2].grid(False)
        ax[tmp2].axis('off')
        j+=1
    fig.suptitle(None)
    emec.plotting.plt_handle_suffix(fig,path=path+"SM-DIC",**plt_Fig_dict)
    
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
emec.plotting.plt_handle_suffix(fig,path=path+"SM-OH",**plt_Fig_dict)

#%%%% Add Eva Overview
if ptype == "TBT":
    pltvarco='_opt'
    pltvartmp=YM_opt_str
    tmp=cs[['Donor','Origin_sshort']].agg(lambda x: x.drop_duplicates().count())
    gs_kw_width_ratios=[tmp[0]/tmp[1], 1.0]
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
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
boxplt_dl(pdf=cs, var='fy', ytxt='Yield strength in MPa',
          xl='Origin_sshort', axl=ax['fL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['fD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
boxplt_dl(pdf=cs, var='ey'+pltvarco, ytxt='Strain at yield stress',
          xl='Origin_sshort', axl=ax['eL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['eD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
boxplt_dl(pdf=cs, var='Uy'+pltvarco, ytxt='Yield strain energy in mJ/mm³',
          xl='Origin_sshort', axl=ax['UL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['UD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
fig.suptitle(None)
emec.plotting.plt_handle_suffix(fig,path=path+"SM-OV1",**plt_Fig_dict) 
   

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
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
boxplt_dl(pdf=cs, var='eu'+pltvarco, ytxt='Strain at ultimate stress',
          xl='Origin_sshort', axl=ax['euL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['euD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
boxplt_dl(pdf=cs, var='Uu'+pltvarco, ytxt='Ultimate strain energy in mJ/mm³',
          xl='Origin_sshort', axl=ax['UuL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['UuD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
fig.suptitle(None)
emec.plotting.plt_handle_suffix(fig,path=path+"SM-OV2",**plt_Fig_dict) 

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
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
boxplt_dl(pdf=cs, var='eb'+pltvarco, ytxt='Strain at fracture stress',
          xl='Origin_sshort', axl=ax['ebL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['ebD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
boxplt_dl(pdf=cs, var='Ub'+pltvarco, ytxt='Fracture strain energy in mJ/mm³',
          xl='Origin_sshort', axl=ax['UbL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['UbD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values())
fig.suptitle(None)
emec.plotting.plt_handle_suffix(fig,path=path+"SM-OV3",**plt_Fig_dict)


#%%%% Add Eva Yield
if ptype == "TBT":
    tmp=pd.DataFrame({'YK':['fyk','eyk_opt','Uyk_opt','KLA'],
                      'Y0':['fy0','ey0_opt','Uy0_opt','0.0 % plso.'],
                      'Y1':['fy1','ey1_opt','Uy1_opt','0.007 % plso.'],
                      'Y2':['fy2','ey2_opt','Uy2_opt','0.1 % plso.'],
                      'Y': ['fy', 'ey_opt', 'Uy_opt', '0.2 % plso.'],
                      'U': ['fu', 'eu_opt', 'Uu_opt', 'u']},
                      # 'B': ['fb', 'eb_opt', 'Ub_opt', 'b']}, 
                     index=['f','e','U','name'])  
elif ptype == "ACT":
    tmp=pd.DataFrame({'YK':['fyk','eyk_con','Uyk_con','KLA'],
                      'Y0':['fy0','ey0_con','Uy0_con','0.0 % plso.'],
                      'Y1':['fy1','ey1_con','Uy1_con','0.05 % plso.'],
                      'Y2':['fy2','ey2_con','Uy2_con','0.1 % plso.'],
                      'Y': ['fy', 'ey_con', 'Uy_con', '0.2 % plso.'],
                      'U': ['fu', 'eu_con', 'Uu_con', 'u']},
                      # 'B': ['fb', 'eb_con', 'Ub_con', 'b']}, 
                     index=['f','e','U','name'])
elif ptype == "ATT":
    tmp=pd.DataFrame({'YK':['fyk','eyk_con','Uyk_con','KLA'],
                      'Y0':['fy0','ey0_con','Uy0_con','0.0 % plso.'],
                      'Y1':['fy1','ey1_con','Uy1_con','0.1 % plso.'],
                      'Y': ['fy', 'ey_con', 'Uy_con', '0.2 % plso.'],
                      'Y2':['fy2','ey2_con','Uy2_con','0.5 % plso.'],
                      'U': ['fu', 'eu_con', 'Uu_con', 'u']},
                      # 'B': ['fb', 'eb_con', 'Ub_con', 'b']},  
                     index=['f','e','U','name'])
    
gs_kw = dict(width_ratios=[1], 
             height_ratios=[1, 1, 1],
             wspace=0.1, hspace=0.1)
fig, ax = plt.subplot_mosaic([['f'],
                              ['e'],
                              ['U']],
                              gridspec_kw=gs_kw,
                              # empty_sentinel='lower mid',
                              figsize=figsize_sup,
                              constrained_layout=True)
tmp2=cs[tmp.loc['f']].copy()
tmp3=tmp2[tmp.loc['f','U']]
tmp2.drop(tmp.loc['f','U'], axis=1, inplace=True)
tmp4=(tmp2.sub(tmp3,axis=0)).div(tmp3,axis=0)
# axt = sns.boxplot(data=tmp4, ax=ax['f'],
#                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
#                                             "markeredgecolor":"black",
#                                             "markersize":"20","alpha":0.75})
# ax['f'].set_ylabel('Deviation to ultimate strength')
boxplt_ext(pdf=tmp4, axl=ax['f'], 
           ytxt='Deviation to ultimate strength')
ax['f'].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
tmp2=cs[tmp.loc['e']].copy()
tmp3=tmp2[tmp.loc['e','U']]
tmp2.drop(tmp.loc['e','U'], axis=1, inplace=True)
tmp4=(tmp2.sub(tmp3,axis=0)).div(tmp3,axis=0)
# axt = sns.boxplot(data=tmp4, ax=ax['e'],
#                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
#                                             "markeredgecolor":"black",
#                                             "markersize":"20","alpha":0.75})
# ax['e'].set_ylabel('Deviation to strain at ultimate strength')
boxplt_ext(pdf=tmp4, axl=ax['e'], 
           ytxt='Deviation to strain at ultimate strength')
ax['e'].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
ax['e'].sharex(ax['f'])
tmp2=cs[tmp.loc['U']].copy()
tmp3=tmp2[tmp.loc['U','U']]
tmp2.drop(tmp.loc['U','U'], axis=1, inplace=True)
tmp4=(tmp2.sub(tmp3,axis=0)).div(tmp3,axis=0)
# axt = sns.boxplot(data=tmp4, ax=ax['U'],
#                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
#                                             "markeredgecolor":"black",
#                                             "markersize":"20","alpha":0.75})
# ax['U'].set_xlabel('Types')
# ax['U'].set_ylabel('Deviation to ultimate strain energy density')
boxplt_ext(pdf=tmp4, axl=ax['U'], xtxtl='Types',
           ytxt='Deviation to ultimate strain energy density')
ax['U'].sharex(ax['e'])
ax['U'].set_xticklabels(tmp.loc['name',~(tmp.columns=='U')].to_list())
fig.suptitle(None)
emec.plotting.plt_handle_suffix(fig,path=path+"SM-YD",**plt_Fig_dict)

#%%%% Add Eva side dependence
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
tmp=cs.query("Side_LR =='L' or Side_LR =='R'")
# tmp=tmp.sort_values(['Side_LR','Donor','Origin_sshort'])
# tmp=tmp.sort_values('Side_LR')
boxplt_dl(pdf=tmp, var=pltvartmp, ytxt='Elastic modulus in MPa',
          xl='Origin_sshort', axl=ax['EL'], tl='By harvesting region', xtxtl='Region',
          xd='Donor', axd=ax['ED'], td='By cadaver', xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values(),
          hue='Side_LR', htirep={'L':'Left','R':'Right'}, hn=None)
boxplt_dl(pdf=tmp, var='fy', ytxt='Yield strength in MPa',
          xl='Origin_sshort', axl=ax['fL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['fD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values(),
          hue='Side_LR', htirep={'L':'Left','R':'Right'}, hn=None)
boxplt_dl(pdf=tmp, var='ey'+pltvarco, ytxt='Strain at yield stress',
          xl='Origin_sshort', axl=ax['eL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['eD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values(),
          hue='Side_LR', htirep={'L':'Left','R':'Right'}, hn=None)
boxplt_dl(pdf=tmp, var='Uy'+pltvarco, ytxt='Yield strain energy in mJ/mm³',
          xl='Origin_sshort', axl=ax['UL'], tl=None, xtxtl='Region',
          xd='Donor', axd=ax['UD'], td=None, xtxtd='Cadaver', 
          xltirep={}, xdtirep=doda.Naming, orderl=Locdict.values(),
          hue='Side_LR', htirep={'L':'Left','R':'Right'}, hn=None)
fig.suptitle(None)
emec.plotting.plt_handle_suffix(fig,path=path+"SM-SLR",**plt_Fig_dict)

if ptype == "TBT":
    pltvarco='_opt'
    pltvartmp=YM_opt_str
    tmp=cs.query("Side_pd =='p' or Side_pd =='d'")

    gs_kw = dict(width_ratios=[1], 
                 height_ratios=[1, 1, 1, 1],
                 wspace=0.1, hspace=0.1)
    fig, ax = plt.subplot_mosaic([['EL'],
                                  ['fL'],
                                  ['eL'],
                                  ['UL']],
                                  gridspec_kw=gs_kw,
                                  # empty_sentinel='lower mid',
                                  figsize=figsize_sup,
                                  constrained_layout=True)
    boxplt_ext(pdf=tmp, var=pltvartmp, ytxt='Elastic modulus in MPa',
              xl='Origin_sshort', axl=ax['EL'], tl=None, xtxtl='Region',
              hue='Side_pd',htirep={'d':'distal','p':'proximal'},
              orderl=Locdict.values())
    boxplt_ext(pdf=tmp, var='fy', ytxt='Yield strength in MPa',
              xl='Origin_sshort', axl=ax['fL'], tl=None, xtxtl='Region',
              hue='Side_pd',htirep={'d':'distal','p':'proximal'},
              orderl=Locdict.values())
    boxplt_ext(pdf=tmp, var='ey'+pltvarco, ytxt='Strain at yield stress',
              xl='Origin_sshort', axl=ax['eL'], tl=None, xtxtl='Region',
              hue='Side_pd',htirep={'d':'distal','p':'proximal'},
              orderl=Locdict.values())
    boxplt_ext(pdf=tmp, var='Uy'+pltvarco, ytxt='Yield strain energy in mJ/mm³',
              xl='Origin_sshort', axl=ax['UL'], tl=None, xtxtl='Region',
              hue='Side_pd',htirep={'d':'distal','p':'proximal'},
              orderl=Locdict.values())
    fig.suptitle(None)
    emec.plotting.plt_handle_suffix(fig,path=path+"SM-Spd",**plt_Fig_dict)

if ptype == "ACT":
    pltvarco='_con'
    pltvartmp=YM_con_str
    tmp=cs.query("Direction_test =='x' or Direction_test =='y' or Direction_test =='z'")
    tmp1=dict(palette={'x':sns.color_palette()[3],
                       'y':sns.color_palette()[2],
                       'z':sns.color_palette()[0]}, 
              showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                         "markeredgecolor":"black",
                                         "markersize":"8","alpha":0.75})
    tmp2=dict(palette={'x':sns.color_palette()[3],
                       'y':sns.color_palette()[2],
                       'z':sns.color_palette()[0]}, 
              dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
    gs_kw = dict(width_ratios=[1], 
                 height_ratios=[1, 1, 1, 1],
                 wspace=0.1, hspace=0.1)
    fig, ax = plt.subplot_mosaic([['EL'],
                                  ['fL'],
                                  ['eL'],
                                  ['UL']],
                                  gridspec_kw=gs_kw,
                                  # empty_sentinel='lower mid',
                                  figsize=figsize_sup,
                                  constrained_layout=True)
    boxplt_ext(pdf=tmp, var=pltvartmp, ytxt='Elastic modulus in MPa',
              xl='Origin_sshort', axl=ax['EL'], tl=None, xtxtl='Region',
              hue='Direction_test', orderl=Locdict.values(),
              bplkws=tmp1, splkws=tmp2)
    boxplt_ext(pdf=tmp, var='fy', ytxt='Yield strength in MPa',
              xl='Origin_sshort', axl=ax['fL'], tl=None, xtxtl='Region',
              hue='Direction_test', orderl=Locdict.values(),
              bplkws=tmp1, splkws=tmp2)
    boxplt_ext(pdf=tmp, var='ey'+pltvarco, ytxt='Strain at yield stress',
              xl='Origin_sshort', axl=ax['eL'], tl=None, xtxtl='Region',
              hue='Direction_test', orderl=Locdict.values(),
              bplkws=tmp1, splkws=tmp2)
    boxplt_ext(pdf=tmp, var='Uy'+pltvarco, ytxt='Yield strain energy in mJ/mm³',
              xl='Origin_sshort', axl=ax['UL'], tl=None, xtxtl='Region',
              hue='Direction_test', orderl=Locdict.values(),
              bplkws=tmp1, splkws=tmp2)
    fig.suptitle(None)
    emec.plotting.plt_handle_suffix(fig,path=path+"SM-SlD",**plt_Fig_dict)
    
#%%%% Correlation
cs_short_corr1, cs_short_corr2 = emec.stat_ext.Corr_ext(
    cs_short[css_ncols], method=mcorr, 
    sig_level={0.001:'$^a$',0.01:'$^b$',0.05:'$^c$',0.10:'$^d$'},
    corr_round=2
    )
gs_kw = dict(width_ratios=[1.0,0.05], height_ratios=[1], wspace=0.1)
fig, ax = plt.subplot_mosaic([['d', 'b']],
                              gridspec_kw=gs_kw,
                              figsize=figsize_sup,
                              constrained_layout=True)
axt=sns.heatmap(cs_short_corr1, annot = cs_short_corr2, fmt='',
              center=0, vmin=-1,vmax=1, annot_kws={"size":8, 'rotation':90},
              xticklabels=1, ax=ax['d'],cbar_ax=ax['b'])
emec.plotting.tick_label_renamer(ax=axt, renamer=VIPar_plt_renamer, axis='both')
ax['d'].tick_params(left=True, bottom=True)
ax['b'].yaxis.set_major_formatter(plt_tick.FormatStrFormatter('%0.1f'))
fig.suptitle(None)
emec.plotting.plt_handle_suffix(fig,path=path+"SM-MC",**plt_Fig_dict)

doda_corr1, doda_corr2, = emec.stat_ext.Corr_ext(
    cs_doda_short, method=mcorr,
    sig_level={0.001:'$^a$',0.01:'$^b$',0.05:'$^c$',0.10:'$^d$'},
    corr_round=2
    )
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
emec.plotting.tick_label_renamer(ax=ax['d'], renamer=VIPar_plt_renamer, axis='both')
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
emec.plotting.tick_label_renamer(ax=ax['di'], renamer=VIPar_plt_renamer, axis='both')
ax['di'].set_ylabel('Donor ICD-codes')
ax['di'].tick_params(axis='y', labelrotation=90)
ax['di'].tick_params(left=True, bottom=True)
ax['bi'].yaxis.set_major_formatter(plt_tick.FormatStrFormatter('%0.1f'))
plt.setp(ax['di'].get_yticklabels(), va="center")
ax['di'].set_xlabel('Material parameters')
ax['di'].tick_params(axis='x', labelrotation=90)
fig.suptitle(None)
emec.plotting.plt_handle_suffix(fig,path=path+"SM-DC",**plt_Fig_dict)

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
emec.plotting.plt_handle_suffix(fig,path=path+"SM-LR",**plt_Fig_dict) 
#%% Close Log
MG_logopt['logfp'].close()