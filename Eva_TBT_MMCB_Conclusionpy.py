# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:43:41 2022

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
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import Eva_common as Evac

plt.rcParams['figure.figsize'] = [6.3,3.54]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size']= 8.0
#pd.set_option('display.expand_frame_repr', False)
plt.rcParams['lines.linewidth']= 1.0
plt.rcParams['lines.markersize']= 4.0
plt.rcParams['markers.fillstyle']= 'none'
plt.rcParams['axes.grid']= True


plt_Fig_dict={'tight':True, 'show':True, 
              'save':True, 's_types':["pdf","png"], 
              'clear':True, 'close':True}
plt_Fig_dict={'tight':True, 'show':True, 
              'save':False, 's_types':["pdf","png"], 
              'clear':True, 'close':True}
#%% Functions
def relDev(pdo, pdr):
    out = (pdo - pdr)/pdr
    return out

def CD_rep(pdo, groupby='Series', var='DEFlutoB', 
           det_met='SM-RRT', outtype='txt', tnform='{:.3e}'):
    """ Calculate critical differences and compare to given.
    """
    # grsagg=pdo.groupby(groupby)[var].agg(['mean','std','count'])
    grsagg=pdo[[groupby,var]].groupby(groupby).agg(['mean','std','count'])
    if isinstance(var,tuple):
        grsagg=grsagg.droplevel([0,1],axis=1)
    grs_sumhrn=grsagg['count'].apply(lambda x: 1/(2*x)).sum()
    grs_MD=grsagg['mean'].max()-grsagg['mean'].min()
    
    CDF= 1.96 * 2**0.5 # 2.8 only implemented for 95%-probability level (only two groups?)
    if det_met=='SM-RRT': #https://www.methodensammlung-bvl.de/resource/blob/208066/e536126ed1723145e51fc90b12736f5e/planung-und-statistische-auswertung-data.pdf
        allagg=pdo[var].agg(['mean','std','count'])
        CD=CDF*allagg['std']*np.sqrt(grs_sumhrn)
    elif det_met=='CV-DC': #https://flexikon.doccheck.com/de/Kritische_Differenz
        allagg=pdo[var].agg(['max',Evac.coefficient_of_variation])
        CD=abs(allagg['coefficient_of_variation']*CDF*allagg['max'])
    elif det_met=='CV': #https://link.springer.com/chapter/10.1007/978-3-662-48986-4_887
        allagg=pdo[var].agg([Evac.coefficient_of_variation])
        CD=abs(allagg['coefficient_of_variation'])*CDF
    elif det_met=='SD': #https://edoc.hu-berlin.de/bitstream/handle/18452/11713/cclm.1982.20.11.817.pdf?sequence=1
        allagg=pdo[var].agg(['std'])
        CD=abs(allagg['std'])*CDF
    else:
        raise NotImplementedError("Method %s not implemented!"%det_met)
        
    eta = grs_MD/CD
    # MD_l_CD=True if abs(grs_MD) < CD else False
    MD_l_CD=False
    t_grs_MD=tnform.format(grs_MD)
    t_CD    =tnform.format(CD)
    txt="{} >  {} -> Repeatability not verified! (eta={})".format(t_grs_MD,t_CD,eta,)
    if abs(grs_MD) <= CD:
        MD_l_CD=True
        txt="{} <= {} -> Repeatability verified! (eta={})".format(t_grs_MD,t_CD,eta,)
    
    if outtype=='Ser_all':
        out= pd.Series({'df_groups_agg':grsagg,'df_all_agg':allagg,
                        'groups_MD':grs_MD, 'CD':CD, 'eta':eta, 'MDlCD':MD_l_CD})
    elif outtype=='Tuple_all':
        out=grsagg,allagg,grs_MD,CD,eta,MD_l_CD
    elif outtype=='txt':
        out= txt
    elif outtype=='Series':
        out= pd.Series({'MD':grs_MD,'CD':CD, 'eta':eta, 'H0': MD_l_CD})
    return out


def CD_test_multi(df, group_main='Series', group_sub=['A'],
                    ano_Var=['WC_vol'],
                    det_met='SM-RRT',
                    Transpose=True):
    df_out=pd.DataFrame([],dtype='O')
    # for sg in group_sub:
    #     for av in ano_Var:
    for av in ano_Var:
        for sg in group_sub:
            tmp=CD_rep(pdo=df, groupby=(group_main,sg), 
                          var=(av,sg), det_met=det_met,
                          outtype = 'Series')
            # name='_'.join([group_main,av,sg])
            name='_'.join([av,sg])
            df_out[name]=tmp
    if Transpose:
        df_out=df_out.T
    return df_out

def group_ANOVA_MComp_multi(df, group_main='Series', group_sub=['A'],
                            ano_Var=['WC_vol'],
                            mpop = "ANOVA", alpha=0.05, group_ren={},
                            do_mcomp_a=0, mcomp='TukeyHSD', mpadj='bonf', Ffwalpha=1,
                            mkws={},
                            Transpose=True):
    df_out=pd.DataFrame([],dtype='O')
    for av in ano_Var:
        if len(group_sub) == 0:
            tmp=Evac.group_ANOVA_MComp(df=df, groupby=group_main, 
                                       ano_Var=av,
                                       group_str=group_main,
                                       ano_str=av,
                                       mpop=mpop, alpha=alpha,
                                       do_mcomp_a=do_mcomp_a, 
                                       mcomp=mcomp, 
                                       mpadj=mpadj, Ffwalpha=Ffwalpha, mkws=mkws,
                                       add_out = 'Series')
            # name='_'.join([group_main,av,sg])
            name=av
            df_out[name]=tmp
        else:
            for sg in group_sub:
                tmp=Evac.group_ANOVA_MComp(df=df, groupby=(group_main,sg), 
                                           ano_Var=(av,sg),
                                           group_str=group_main,
                                           ano_str='_'.join([av,sg]),
                                           mpop=mpop, alpha=alpha,
                                           do_mcomp_a=do_mcomp_a, 
                                           mcomp=mcomp, 
                                           mpadj=mpadj, Ffwalpha=Ffwalpha, mkws=mkws,
                                           add_out = 'Series')
                # name='_'.join([group_main,av,sg])
                name='_'.join([av,sg])
                df_out[name]=tmp
    if Transpose:
        df_out=df_out.T
    return df_out

from functools import partial, wraps
def Hypo_test(df, groupby, ano_Var,
              group_str=None, ano_str=None,
              alpha=0.05, group_ren={},
              mcomp='TukeyHSD', mkws={},
              rel=False, rel_keys=[],
              add_T_ind=3, add_out = False):
    if ano_str is None:
        ano_str = ano_Var
    if group_str is None:
        group_str = groupby
    dft = df.copy(deep=True)
    if rel and rel_keys!=[]:
        # dft.index = pd.MultiIndex.from_frame(dft.loc(axis=1)[np.concatenate((rel_keys,
        #                                                                      [groupby]))])
        dft.index = pd.MultiIndex.from_frame(dft.loc(axis=1)[rel_keys+[groupby]])

    else:
        dft.index = pd.MultiIndex.from_arrays([dft.index,
                                               dft.loc(axis=1)[groupby]])
    dft = dft[ano_Var]
    dft = dft.unstack(level=-1)
    if rel: dft=dft.dropna(axis=0)
    
    dfgr=dft.columns.values
    if not len(dfgr)==2:
        raise ValueError('More than two groups (%s)!'%dfgr)
    a = dft[dfgr[0]].dropna()
    b = dft[dfgr[1]].dropna()
    ano_df2=a.count()+b.count()-2 #Freiheitsgrad = Testpersonen pro Gruppe - 1
    
    if mcomp=='TukeyHSD':
        stats_test  = wraps(partial(stats.tukey_hsd, **mkws))(stats.tukey_hsd)
    elif mcomp=='ttest_ind':
        stats_test  = wraps(partial(stats.ttest_ind, **mkws))(stats.ttest_ind)
    elif mcomp=='ttest_rel':
        stats_test  = wraps(partial(stats.ttest_rel, **mkws))(stats.ttest_rel)
    elif mcomp=='mannwhitneyu':
        stats_test  = wraps(partial(stats.mannwhitneyu, **mkws))(stats.mannwhitneyu)
    elif mcomp=='wilcoxon':
        stats_test  = wraps(partial(stats.wilcoxon, **mkws))(stats.wilcoxon)
    else:
        raise NotImplementedError('Method %s for multi comparison not implemented!'%mcomp)
    t = stats_test(a,b)
    F = t.statistic
    p = t.pvalue
    if p < alpha:
        rtxt = 'H0 rejected!'
        H0=False
    else:
        rtxt = 'Fail to reject H0!'
        H0=True
    txt=("- F(%d) = %.3e, p = %.3e, for %s to %s (%s)"%(ano_df2,
                                                          F,p,
                                                          ano_str,group_str,
                                                          rtxt)) # Gruppen sind signifikant verschieden bei p<0.05
    if add_out is True:
        return txt, t
    elif add_out=='Series':
        return pd.Series({'DF2':ano_df2, 'Stat':F, 'p':p, 'H0':H0})
    elif add_out=='Test':
        return dft, pd.Series({'DF2':ano_df2, 'Stat':F, 'p':p, 'H0':H0}), txt
    else:
        return txt

def Hypo_test_multi(df, group_main='Series', group_sub=['A'],
                    ano_Var=['WC_vol'],
                    mcomp='mannwhitneyu', alpha=0.05, mkws={},
                    rel=False, rel_keys=[],
                    Transpose=True):
    df_out=pd.DataFrame([],dtype='O')
    # for sg in group_sub:
    #     for av in ano_Var:
    for av in ano_Var:
        for sg in group_sub:
            tmp=Hypo_test(df=df, groupby=(group_main,sg), 
                          ano_Var=(av,sg), alpha=alpha,
                          mcomp=mcomp, mkws=mkws,
                          # rel=rel, rel_keys=rel_keys, add_out = 'Series')
                          rel=rel, rel_keys=[(x,sg) for x in rel_keys], 
                          add_out = 'Series')
            # name='_'.join([group_main,av,sg])
            name='_'.join([av,sg])
            df_out[name]=tmp
    if Transpose:
        df_out=df_out.T
    return df_out

def Multi_conc(df, group_main='Donor', anat='VA', 
               met='Kruskal', alpha=0.05,
               stdict={'WC_vol':['A','B','L'],'WC_vol_rDA':['B','C','L'],
                       'lu_F_mean':['B'],'DEFlutoB':['C','G','L'],
                       'Hyst_APn':['B'],'DHAPntoB':['C','G','L']},
               rel=False, rel_keys=[], kws={}):
    out=pd.DataFrame([],dtype='O')
    for i in stdict.keys():
        if anat=='VA':
            out2=group_ANOVA_MComp_multi(df=df, 
                                 group_main=group_main, 
                                 group_sub=stdict[i],
                                 ano_Var=[i],
                                 mpop=met, alpha=alpha, **kws)
        elif anat=='HT':
            out2=Hypo_test_multi(df=df, 
                                 group_main=group_main, 
                                 group_sub=stdict[i],
                                 ano_Var=[i], mcomp=met, alpha=alpha,
                                 rel=rel, rel_keys=rel_keys, **kws)
        elif anat=='CD':
            out2=CD_test_multi(df=df, 
                                 group_main=group_main, 
                                 group_sub=stdict[i],
                                 ano_Var=[i], det_met=met, **kws)
        out=pd.concat([out,out2])
    out.index=pd.MultiIndex.from_product([[group_main],out.index])
    return out

import lmfit
def func_exp(x, a, b, c):
    """Return values from a general exponential function."""
    return a + b * np.exp(c * x)

def func_exp_str(xl, yl, 
                 a, b, c, 
                 t_form='{a:.3e},{b:.3e},{c:.3e}'):
    """Return string from a general exponential function."""
    txtc = t_form.format(**{'a':a,'b':b,'c':c},).split(',')
    # txt = r'{y} = {0} + {1} $e$^{{2}{x}}'.format(*txtc,**{'y':yl,'x':xl},)
    txtd = '{0}\,{1}'.format(txtc[2],xl,)
    # txt = r'{y} = {0} + {1} $e^{2}{3}{4}$'.format(txtc[0],txtc[1],
    #                                             '{',txtd,'}',**{'y':yl},)
    if a==0:
        txt = r'${y}\,=\,{0}\,e^{1}{2}{3}$'.format(txtc[1],
                                                    '{',txtd,'}',**{'y':yl},)
    elif a==(-b):
        txt = r'${y}\,=\,{0}\,(e^{1}{2}{3}-1)$'.format(txtc[1],
                                                '{',txtd,'}',**{'y':yl},)
    else:
        txt = r'${y}\,=\,{0}\,+\,{1}\,e^{2}{3}{4}$'.format(txtc[0],txtc[1],
                                                    '{',txtd,'}',**{'y':yl},)
    # txt = r'{y} = {0} + {1} $e$^{2}{3}{4}'.format(txtc[0],txtc[1],
    #                                             '(',txtd,')',**{'y':yl},)
    return txt


def func_lin(x, a, b):
    """Return values from a general linear function."""
    return a + b * x

def func_lin_str(xl, yl, 
                 a, b,
                 t_form='{a:.3e},{b:.3e}'):
    """Return string from a general linear function."""
    txtc = t_form.format(**{'a':a,'b':b},).split(',')
    if a==0:
        txt = r'${y}\,=\,{0}\,{x}$'.format(txtc[1],
                                           **{'y':yl,'x':xl},)
    else:
        txt = r'${y}\,=\,{0}\,+\,{1}\,{x}$'.format(txtc[0],txtc[1],
                                                   **{'y':yl,'x':xl},)
    return txt

def plt_Regc(pdo, x, y, xl,yl, 
             fit,
             title, path, plt_Fig_dict):
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax1.set_title(title)
    sns.scatterplot(x=x, y=y, data=pdo, ax=ax1, label='data')
    xtmp = np.linspace(pdo[x].min(), pdo[x].max(), 101)
    ytmp = fit.model.eval(x=xtmp, **fit.result.params.valuesdict())
    sns.lineplot(x=xtmp, y=ytmp, ax=ax1, label='fit')
    ax1.set_xlabel(xl+' / -')
    ax1.set_ylabel(yl+' / -')
    ax1.legend()
    fig.suptitle('')
    Evac.plt_handle_suffix(fig,path=path,**plt_Fig_dict)
    
def plt_ax_Reg(pdo, x, y,  
               fit={}, ax=None, label_d='Data', label_f=False,
               xlabel=None, ylabel=None, title=None,
               skws={}, lkws={}):
    if ax is None: ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.scatterplot(x=x, y=y, data=pdo, ax=ax, label=label_d, **skws)
    xtmp = np.linspace(pdo[x].min(), pdo[x].max(), 101)
    ytmp = fit['Model'].eval(x=xtmp, **fit['PD'])
    if label_f:
        ltxt=fit['Exp_txt']+' ($R²$={:.3f})'.format(fit['Rquad'])
    else:
        ltxt='Fit-%s'%fit['Name']
    sns.lineplot(x=xtmp, y=ytmp, ax=ax, label=ltxt, **lkws)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax

def plt_ax_Reg_fo(pdo, x,   
                  fit={}, ax=None, label_f=False, lkws={}):
    if ax is None: ax = plt.gca()
    xtmp = np.linspace(pdo[x].min(), pdo[x].max(), 101)
    ytmp = fit['Model'].eval(x=xtmp, **fit['PD'])
    if label_f:
        ltxt=fit['Exp_txt']+' ($R²$={:.3f})'.format(fit['Rquad'])
    else:
        ltxt='Fit-%s'%fit['Name']
    sns.lineplot(x=xtmp, y=ytmp, ax=ax, label=ltxt, **lkws)
    return ax

def Regfitret(pdo, x,y, name='exponential', guess = dict(a=-0.01, b=0.1, c=-1),
              xl=r'$X$', yl=r'$Y$', t_form='{a:.3e},{b:.3e},{c:.3e}', max_nfev=1000):
    if name in ['linear','linear_nc']:
        func=func_lin
        func_str=func_lin_str
    elif name in ['exponential','exponential_nc','exponential_x0']:
        func=func_exp
        func_str=func_exp_str
    else:
        raise NotImplementedError('Type %s not implemented!'%name)
    model=lmfit.Model(func)
    params=model.make_params()
    if name in ['linear_nc','exponential_nc']:
        params.add('a', value=0, vary=False)
    elif name in ['linear_x0']:
        params.add('a', value=guess['a'], vary=False)
    elif name in ['exponential_x0']:
        y0=guess.pop('a')
        params.add('a', value=None, expr="{:.8e}-b".format(y0), vary=False)
    # fit = model.fit(pdo[y], x=pdo[x], nan_policy='omit', **guess)
    fit = model.fit(pdo[y], x=pdo[x], params=params, nan_policy='omit', **guess)
    Rquad  = 1 - fit.residual.var() / np.nanvar(pdo[y])      
    fit.rsqur=Rquad
    rtxt=lmfit.fit_report(fit.result)
    rtxt=rtxt.replace("\n[[Variables]]",
                      "\n    R-square           = %1.8f\n[[Variables]]"%Rquad)
    etxt=func_str(xl=xl, yl=yl, t_form=t_form, **fit.result.params.valuesdict())
    out = {'Name': name, 'Model': model, 'Result': fit, 
           'PD': fit.result.params.valuesdict(),
           'Exp_txt': etxt, 'Rep_txt': rtxt, 'Rquad': Rquad}
    return out

# =============================================================================
#%% Einlesen und auswählen
# data="S1"
# data="S2"
data="Complete"

no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
               'B01.1','B01.2','B01.3', 'B02.3',
               'C01.1','C01.2','C01.3', 'C02.3',
               'D01.1','D01.2','D01.3', 'D02.3',
               'F01.1','F01.2','F01.3', 'F02.3',
               'G01.1','G01.2','G01.3', 'G02.3']
var_suffix = [""] #Suffix of variants of measurements (p.E. diffferent moistures)
VIPar_plt_renamer = {'fy':'$f_{y}$','fu':'$f_{u}$','fb':'$f_{b}$',
                     'ey_con':r'$\epsilon_{y,con}$','Wy_con':'$W_{y,con}$',
                     'eu_con':r'$\epsilon_{u,con}$','Wu_con':'$W_{u,con}$',
                     'eb_con':r'$\epsilon_{b,con}$','Wb_con':'$W_{b,con}$',
                     'ey_opt':r'$\epsilon_{y,opt}$','Wy_opt':'$W_{y,opt}$',
                     'eu_opt':r'$\epsilon_{u,opt}$','Wu_opt':'$W_{u,opt}$',
                     'eb_opt':r'$\epsilon_{b,opt}$','Wb_opt':'$W_{b,opt}$',
                     # 'E_con': '$E_{con}$','E_opt': '$E_{opt}$',
                     'Density_app': r'$\rho_{app}$',
                     'Length_test': r'$l_{test}$',
                     # 'thickness_mean': r'$\overline{t}$',
                     # 'width_mean': r'$\overline{w}$',
                     'thickness_mean': r'$\overline{h}$',
                     'thickness_2': r'$h_{mid}$','geo_dthick': r'$\Delta h/h_{mid}$',
                     'width_mean': r'$\overline{b}$',
                     'width_2': r'$b_{mid}$','geo_dwidth': r'$\Delta b/b_{mid}$',
                     'Area_CS': r'$\overline{A}_{CS}$', 'Volume': r'$\overline{V}$',
                     'MoI_mid': r'$I_{mid}$','geo_MoI_mean':r'$\overline{I}$','geo_dMoI': r'$\Delta I/I_{mid}$',
                     'geo_curve_max': r'$\kappa_{max}$','geo_dcurve': r'$\Delta \kappa/\kappa_{mid}$',
                     'geo_curve_mid_circ': r'$\kappa_{mid,circle}$',
                     'ind_R_max': r'$ind_{el,max}$','ind_R_mean': r'$\overline{ind}_{el}$',
                     'ind_U_max': r'$ind_{u,max}$','ind_U_mean': r'$\overline{ind}_{u}$',
                     'DMWtoA': r'$D_{m,org}$','DMWtoG': r'$D_{m,dry}$',
                     'RHenv': r'$\Phi_{env}$', 'Humidity_store':r'$\Phi_{sto}$',
                     'WC_vol': r'$\Phi_{vol}$', 'WC_gra': r'$\Phi_{gra}$',
                     'WC_vol_toA':r'$\Delta_{\Phi_{vol},org}$','WC_vol_rDA':r'$D_{\Phi_{vol},org}$',
                     'WC_gra_toA':r'$\Delta_{\Phi_{gra},org}$','WC_gra_rDA':r'$D_{\Phi_{gra},org}$',
                     'lu_F_mean': r'$\overline{E}_{lu}$','lu_F_ratio': r'${E}_{u}/{E}_{l}$',
                     'DEFlutoB': r'$D_{E,sat}$','DEFlutoG': r'$D_{E,dry}$',
                     'Hyst_An': r'$H_{n}$','DHAntoB': r'$D_{H_{n},sat}$','DHAntoG': r'$D_{H_{n},dry}$',
                     'Hyst_APn': r'$H_{n}$','DHAPntoB': r'$D_{H_{n},sat}$','DHAPntoG': r'$D_{H_{n},dry}$'}

Variants_env_relH={'B':1.20, 'C':1.00, 'D':0.90, 'E':0.75, 'F':0.60,
                   'G':0.00, 'H':0.60, 'I':0.75, 'J':0.90, 'K':1.00, 'L':1.20}

protpaths = pd.DataFrame([],dtype='string')
combpaths = pd.DataFrame([],dtype='string')
protpaths.loc['S1','name_prot'] = "FU_Kort_S1-DBV_Protocol_220321.xlsx"
protpaths.loc['S1','path_eva2'] = "Serie_1/"
protpaths.loc['S1','fname_in']  = "S1_CBMM_TBT-Summary"
protpaths.loc['S1','fname_out'] = "S1_CBMM_TBT-Conclusion"
protpaths.loc['S1','name'] = "1st series"

protpaths.loc['S2','name_prot'] = "FU_Kort_S2-DBV_Protocol_220713.xlsx"
protpaths.loc['S2','path_eva2'] = "Serie_2/"
protpaths.loc['S2','fname_in']  = "S2_CBMM_TBT-Summary"
protpaths.loc['S2','fname_out'] = "S2_CBMM_TBT-Conclusion"
protpaths.loc['S2','name'] = "2nd series"

protpaths.loc['Complete','name_prot'] = ""
protpaths.loc['Complete','path_eva2'] = "Complete/"
# protpaths.loc['Complete','fname_in']  = "CBMM_TBT-Summary"
protpaths.loc['Complete','fname_in']  = "CBMM_TBT-Summary_all"
protpaths.loc['Complete','fname_out'] = "CBMM_TBT-Conclusion"
protpaths.loc['Complete','name'] = "two series"

protpaths.loc[:,'path_main']    = "F:/Mess_FU_Kort/"
protpaths.loc[:,'path_eva1']    = "Auswertung/"

combpaths['prot'] = protpaths['path_main']+protpaths['path_eva1']+protpaths['name_prot']
combpaths['in']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']+protpaths['fname_in']
combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']+protpaths['fname_out']
path_doda=os.path.abspath(os.path.join(protpaths.loc['Complete','path_main'],
                                       protpaths.loc['Complete','path_eva1'],
                                       'MM-CB_Donordata_full.xlsx'))

# name_Head = "Moisture Manipulation Compact Bone (%s, %s)\n"%(protpaths.loc[data,'name'],
#                                                            date.today().strftime("%d.%m.%Y"))
name_Head = ""

# out_full= os.path.abspath(path+name_out)
out_full= combpaths.loc[data,'out']
h5_conc = 'Summary'
# h5_data = 'Test_Data'
h5_data = '/Add_/Measurement'



YM_con=['inc','R','A0Al','meanwoso']
YM_opt=['inc','R','D2Mgwt','meanwoso']
YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
YM_opt_str='E_{}_{}_{}_{}'.format(*YM_opt)



log_mg=open(out_full+'.log','w')
Evac.MG_strlog(protpaths.loc[data,'fname_out'], log_mg, printopt=False)
Evac.MG_strlog("\n   Paths:", log_mg, printopt=False)
Evac.MG_strlog("\n   - in:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(combpaths.loc[data,'in']+'.h5'), log_mg, printopt=False)
Evac.MG_strlog("\n   - out:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(combpaths.loc[data,'out']+'.h5'), log_mg, printopt=False)


data_read = pd.HDFStore(combpaths.loc[data,'in']+'.h5','r')
dfa=data_read.select(h5_conc)
dft=data_read.select(h5_data)
data_read.close()

del dfa['Number']

dfa['Side_LR']=dfa.Origin.str.split(' ', expand=True)[3]
dfa['Side_pd']=dfa.Origin.str.split(' ', expand=True)[4]
dfa['Series']=dfa.Designation.str[3]
dfa['Numsch']=dfa.Designation.apply(lambda x: '{0}'.format(x[-3:]))
dfa.Failure_code  = Evac.list_cell_compiler(dfa.Failure_code)
dfa['statistics'] = Evac.list_interpreter(dfa.Failure_code, no_stats_fc)



#Spender:
doda = pd.read_excel(path_doda, skiprows=range(1,2), index_col=0)
# d_eva_c = Evac.pd_agg(cs.groupby('Donor')[cs_num_cols],agg_funcs,True)
# d_eva_c = pd.concat([doda,d_eva_c],axis=1)

Methods_excl_names=["F4Agfu","F4Sgfu","F4Mgfu",
                    "F4gfu"]
Methods_excl_str='|'.join(Methods_excl_names)

#%% QaD
def FC_join(l,r, opt='sort'):
    if l[0]=='nan':
        l=[]
    if r[0]=='nan':
        r=[]
    out=l+r
    if opt == 'sort':
        out = out.sort()
    return out

if data == "Complete":
    protvt={}
    for i in combpaths.loc[~(combpaths.index == 'Complete')].index:
            protvt[i]=pd.read_excel(combpaths.loc[i,'prot'],
                        sheet_name='Protocol_Variants',
                        header=[10,11], skiprows=range(12,13),
                        index_col=0)
    protv = pd.concat(protvt.values(), axis=0)
else:
    # protv=pd.read_excel("F:/Mess_FU_Kort/Auswertung/FU_Kort_S1-DBV_Protocol_220321.xlsx",
    protv=pd.read_excel(combpaths.loc[data,'prot'],
                        sheet_name='Protocol_Variants',
                        header=[10,11], skiprows=range(12,13),
                        index_col=0)
del protv['General']
protv=protv.T.unstack(0).T
protv.index.names = ['Key','Variant']
protv.columns.name=None
# protv.columns = protv.columns.to_series().apply(lambda x: '{0}{1}'.format(*x)).values
# protv = protv.T
protv.Failure_code = Evac.list_cell_compiler(protv.Failure_code)
# protv.index=protv.index.to_series().apply(lambda x: x.split('_')[-1])
# protv[['Temperature_test','Humidity_test','Mass']]=protv[['Temperature_test','Humidity_test','Mass']].astype('float')
protv.loc(axis=1)[idx['Temperature_test':'Mass']]=protv.loc(axis=1)[idx['Temperature_test':'Mass']].astype('float')

i1 = dfa.index.to_series().apply(lambda x: '{0}'.format(x[:-1]))
i2 = dfa.index.to_series().apply(lambda x: '{0}'.format(x[-1]))
dfa.index = pd.MultiIndex.from_arrays([i1,i2], names=['Key','Variant'])

dft=pd.concat([dfa,protv],axis=1)
t = dft.Failure_code.iloc(axis=1)[0]+dft.Failure_code.iloc(axis=1)[1]
dft=dft.loc[:, ~dft.columns.duplicated()]
t2=t.apply(lambda x: x.remove('nan') if ((len(x)>1)&('nan' in x)) else x)
dft.Failure_code = t
dft['statistics'] = Evac.list_interpreter(dft.Failure_code, no_stats_fc)
dft['RHenv'] = dft.index.droplevel(0) # SettingWithCopyWarning
dft['RHenv'] = dft['RHenv'] .map(Variants_env_relH)
dft['RHstore'] =dft['Humidity_store']/100

#%% Prepare
# Wassergehalt
tmp=dft.Mass.loc(axis=0)[:,'G'].droplevel(1)
dft['DMWorg']=(dft.Mass_org-tmp)/tmp
tmp=dft.Mass.loc(axis=0)[:,'G'].droplevel(1)
dft['DMWtoG']=(dft.Mass-tmp)/tmp
tmp=dft.Mass_org.loc(axis=0)[:,'G'].droplevel(1)
dft['DMWtoA']=(dft.Mass-tmp)/tmp

## Wassergehaltsänderung org als df
# # dft_A=dft.loc(axis=0)[:,'G'][['Mass_org']]
# # dft_A.columns=['Mass']
# # dft_A=dft.loc(axis=0)[:,'G'][['Series','Donor','Side_LR','Side_pd','Mass_org']].copy(deep=True)
# dft_A=dft.loc(axis=0)[:,'G'][['Designation','Type','Series','Donor','Origin',
#                               'Side_LR','Side_pd','Mass_org',
#                               'thickness_1','thickness_2','thickness_3',
#                               'width_1','width_2','width_3','Length']].copy(deep=True)
# dft_A.columns=dft_A.columns.str.replace('Mass_org','Mass')
dft_A=dft.loc(axis=0)[:,'G'].loc(axis=1)['Designation':'geo_MoI_mid'].copy(deep=True)
tmp=dft.loc(axis=0)[:,'G'].loc(axis=1)[idx['Side_LR','Side_pd','Series','Numsch']].copy(deep=True)
dft_A=pd.concat([dft_A,tmp],axis=1)
dft_A['Mass']=dft_A['Mass_org']
dft_A.index.set_levels(dft_A.index.get_level_values(1).str.replace('G','A'), 
                       level=1, inplace=True, verify_integrity=False)
tmp=dft.Mass.loc(axis=0)[:,'G'].droplevel(1)
dft_A['DMWtoG']=(dft_A.Mass-tmp)/tmp
dft_A['statistics']=False

# Wassergehalt (gravimetrisch und volumetrisch)
tmp=pd.concat([dft[['Mass']],dft_A[['Mass']]]).sort_index()
tmp2=tmp['Mass'].loc(axis=0)[:,'G'].droplevel(1)
tmp['MassW']=tmp['Mass']-tmp2
tmp['VolW']=tmp['MassW']/(0.997/1000) # Dichte Wasser in g/cm³
tmp['WC_gra']=tmp['MassW']/tmp['Mass']
tmp['VolDry']=tmp['Mass']*0 + tmp2/(1.362/1000) # Reindichte Gewebe in g/mm³ aus Vortest He-Pyk !Ersetzen durch Ergebnisse!
# tmp['WC_vol']=tmp['VolW']/(tmp['VolDry']+tmp['VolW'])
tmp2=(dft[['width_1','width_2','width_3']].mean(axis=1)*dft[['thickness_1','thickness_2','thickness_3']].mean(axis=1)*dft['Length']).loc(axis=0)[:,'G'].droplevel(1)
tmp['VolTot']=tmp['Mass']*0 + tmp2 # Volumen aus Geometrie
tmp['WC_vol']=tmp['VolW']/tmp['VolTot']
tmp['Density_app']=tmp['Mass']/tmp['VolTot']

# dft=pd.concat([dft,tmp[['WC_gra','WC_vol']].query("Variant != 'A'")],axis=1)
# dft_A=pd.concat([dft_A,tmp[['WC_gra','WC_vol']].query("Variant == 'A'")],axis=1)
dft  =pd.concat([dft,  tmp[['VolTot','Density_app','MassW','WC_gra','WC_vol']].query("Variant != 'A'")],axis=1)
dft_A=pd.concat([dft_A,tmp[['VolTot','Density_app','MassW','WC_gra','WC_vol']].query("Variant == 'A'")],axis=1)
dft[['WC_gra_toA','WC_vol_toA']]=dft[['WC_gra','WC_vol']]-dft_A[['WC_gra','WC_vol']].droplevel(1)
dft[['WC_gra_rDA','WC_vol_rDA']]=relDev(dft[['WC_gra','WC_vol']],dft_A[['WC_gra','WC_vol']].droplevel(1))

# E-Modul
t=dft['E_inc_F_D2Mgwt_meanwoso'].loc[:,'B']
dft['DEFtoB']=(dft['E_inc_F_D2Mgwt_meanwoso']-t)/t
t=dft['E_inc_R_D2Mgwt_meanwoso'].loc[:,'B']
dft['DERtoB']=(dft['E_inc_R_D2Mgwt_meanwoso']-t)/t

t=dft['E_inc_F_D2Mgwt_meanwoso'].loc[:,'G']
dft['DEFtoG']=(dft['E_inc_F_D2Mgwt_meanwoso']-t)/t
t=dft['E_inc_R_D2Mgwt_meanwoso'].loc[:,'G']
dft['DERtoG']=(dft['E_inc_R_D2Mgwt_meanwoso']-t)/t

a=Evac.list_interpreter(dft.Failure_code, no_stats_fc, option='include')
b=dft[a].loc(axis=1)['Failure_code','Note']
Evac.MG_strlog("\n\n   - Statistical exclusion: (%d)"%len(dft[a].index),
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("\n"+b.to_string(),5),
               log_mg, printopt=False)

cs = dft.loc[dft.statistics]
# cs['RHenv'] = cs.index.droplevel(0) # SettingWithCopyWarning
# cs['RHenv'] = cs['RHenv'] .map(Variants_env_relH)
cs_num_cols = cs.select_dtypes(include=['int','float']).columns


# t=cs.loc(axis=1)[['Designation','DMWtoA','DMWtoG',
#                   'DEFtoB','DEFtoG','DERtoB','DERtoG']].reset_index()
# t['Numsch']=t.Designation.apply(lambda x: '{0}'.format(x[-3:]))
t=cs.loc(axis=1)[['Designation','Numsch',
                  'DMWtoA','DMWtoG','WC_gra','WC_vol',
                  'DEFtoB','DEFtoG','DERtoB','DERtoG']].reset_index()
t=t.sort_values(['Designation','Variant'])
t_des=t[t.Variant<='G']
t_abs=t[t.Variant>='G']

t_ona=t.dropna().sort_values(['Designation','Variant'])
t_ona_abs=t_abs.dropna().sort_values(['Designation','Variant'])
t_ona_des=t_des.dropna().sort_values(['Designation','Variant'])


#%% Prepare load and unload (09.12.22) (should be done in original evaluation)
def VIP_searcher(Vdf, col=0, sel='l', ser='F', rstart='A', rend='B'):
    sen=sel+'_'+ser
    seV=ser+sel
    if Vdf.index.str.contains('{0}{1}|{0}{2}'.format(*[seV,rstart,rend])).sum()==2:
        val=Vdf.loc[[seV+rstart,seV+rend],col].values
    else:
        val=[np.nan,np.nan]
    out={sen:tuple(val)}
    return out
def VIP_rebuild(ser):
    tmp=ser[~(ser=='')]
    tmp2=pd.Series(tmp.index, tmp.values, name=ser.name)
    tmp3=pd.Series([], dtype='int64', name=ser.name)
    for i in tmp2.index:
        for j in i.split(','):
            tmp3[j]=tmp2.loc[i]
    return tmp3

def interp_twocols(pdo, indcol, valcol, searchind=0.0, 
                   iS=None, iE=None, intmeth='spline', order=2):
    pdt=Evac.pd_limit(pdo, iS=iS, iE=iE).copy(deep=True)
    pdt=pd.Series(pdt[valcol].values, index=pdt[indcol].values)
    pdt.loc[searchind]=np.nan
    pdt=pdt.sort_index().interpolate(intmeth, order=order, limit_direction='both')
    return pdt.loc[searchind]

# data_read = pd.HDFStore(combpaths.loc[data,'in']+'_all.h5','r')
data_read = pd.HDFStore(combpaths.loc[data,'in']+'.h5','r')
dfEt=data_read['Add_/E_inc_df']
# dfVIP=data_read['Add_/VIP'] # Hdf defekt
dfMt=data_read['Add_/Measurement']
data_read.close()

cEt=pd.DataFrame([])
cHt=pd.DataFrame([])
for i in dfEt.index:
    # tmpV=dfVIP.loc[i]
    tmpV=pd.concat([VIP_rebuild(dfMt.loc[i].VIP_m),
                    VIP_rebuild(dfMt.loc[i].VIP_d)], axis=1)
    tmpE=dfEt.loc[i]
    Evar=pd.DataFrame([],columns=['l_F','l_R','u_F','u_R'])
    Evar.loc['con']=pd.Series({**VIP_searcher(tmpV,'VIP_m','l','F'),
                               **VIP_searcher(tmpV,'VIP_m','l','R'),
                               **VIP_searcher(tmpV,'VIP_m','u','F'),
                               **VIP_searcher(tmpV,'VIP_m','u','R')}) # nach Hdf defekt von 0-VIP_m
    Evar.loc['opt']=pd.Series({**VIP_searcher(tmpV,'VIP_d','l','F'),
                               **VIP_searcher(tmpV,'VIP_d','l','R'),
                               **VIP_searcher(tmpV,'VIP_d','u','F'),
                               **VIP_searcher(tmpV,'VIP_d','u','R')}) # nach Hdf defekt von 1-VIP_d
    tmpEr=pd.Series([],name=i, dtype='float64')
    cols_con=tmpE.columns.str.contains('0')
    for j in Evar.columns:
        tmpErcont=tmpE.loc(axis=1)[cols_con].indlim(*Evar.loc['con',j]).agg(Evac.meanwoso)
        tmpEroptt=tmpE.loc(axis=1)[~cols_con].indlim(*Evar.loc['opt',j]).agg(Evac.meanwoso)
        tmpErcont.index=j+'_'+tmpErcont.index
        tmpEroptt.index=j+'_'+tmpEroptt.index
        tmpEr=tmpEr.append([tmpErcont,tmpEroptt])
        tmpEr.name=i
        del tmpErcont,tmpEroptt
    cEt=cEt.append(tmpEr)
    del tmpE, Evar, cols_con, tmpEr
    #Hysteresis    
    tmp=pd.Series([], name=i, dtype='float64')
    tmpM=dfMt.loc[i]
    # tmpV=dfVIP.loc[i]
    try:
       indU=tmpV.loc['U','VIP_d']
    except:
       indU=tmpV.loc['H','VIP_d']        
    try:
        indPl=tmpV.loc['Pl','VIP_d']
    except:
        indPl=tmpV.loc['S','VIP_d']
    try:
        indPu=Evac.Find_closest(tmpM['Strain_opt_d_M'], 
                                tmpM.loc[indPl,'Strain_opt_d_M'], indU)
    except:
        indPu=tmpM.index[-1]
    ncols=tmpM.select_dtypes(include=['int','float']).columns
    tmp['Wdu_l_opt']    = Evac.pd_trapz(tmpM.loc[:indU],
                                        y='Stress',x='Strain_opt_d_M')
    tmp['Wdu_u_opt']    = Evac.pd_trapz(tmpM.loc[indU::].iloc[::-1],
                                        y='Stress',x='Strain_opt_d_M')
    tmp['Wdu_l_opt_P']  = Evac.pd_trapz(tmpM.loc[indPl:indU],
                                        y='Stress',x='Strain_opt_d_M')
    tmp['Wdu_u_opt_P']  = Evac.pd_trapz(tmpM.loc[indU:indPu].iloc[::-1],
                                        y='Stress',x='Strain_opt_d_M')
    tmp['epl_opt'] = interp_twocols(pdo=tmpM, indcol='Stress', valcol='Strain_opt_d_M',
                               iS=indU, iE=None, intmeth='spline', order=2)
    cHt=cHt.append(tmp)

cEt.columns = cEt.columns.str.split('_', expand=True)
cEt.columns.names=['Load','Range','Method']
cEt=cEt.reorder_levels(['Method','Load','Range'],axis=1).sort_index(axis=1,level=0)
# cEt.to_csv(out_full+'-E_lu_FR.csv',sep=';')
cEE=cEt.loc(axis=1)['D2Mgwt',:,:].droplevel([0],axis=1)
i1 = cEE.index.to_series().apply(lambda x: '{0}'.format(x[:-1]))
i2 = cEE.index.to_series().apply(lambda x: '{0}'.format(x[-1]))
cEE.index = pd.MultiIndex.from_arrays([i1,i2], names=['Key','Variant'])

i1 = cHt.index.to_series().apply(lambda x: '{0}'.format(x[:-1]))
i2 = cHt.index.to_series().apply(lambda x: '{0}'.format(x[-1]))
cHt.index = pd.MultiIndex.from_arrays([i1,i2], names=['Key','Variant'])
cHt.sort_index(inplace=True)
# 0000-0001-8378-3108_LEIULANA_35-21_LuPeCo_MMS212G: eu_opt=nan
cHt['Hyst_A']=cHt['Wdu_l_opt']-cHt['Wdu_u_opt']
cHt['Hyst_AP']=cHt['Wdu_l_opt_P']-cHt['Wdu_u_opt_P']
cHt['Hyst_An']=cHt['Hyst_A']/dft['fu']/dft['eu_opt']
cHt['Hyst_APn']=cHt['Hyst_AP']/dft['fu']/dft['eu_opt']
cHt['epl_opt_n']=cHt['epl_opt']/dft['eu_opt']
stafc_dest=dft['statistics'].copy(deep=True)
stafc_dest.loc[:,'G']=stafc_dest.loc[:,'H'].values
# cH_eva=cHt.loc[dft['statistics']].sort_index()
cH_eva=cHt.loc[stafc_dest].sort_index() #exclude destructive Variant G
tmp=cH_eva[['Hyst_A','Hyst_An','Hyst_AP','Hyst_APn']]
tmp2=relDev(tmp,tmp.loc(axis=0)[:,'B'].droplevel([1],axis=0))
tmp2.rename({'Hyst_A': 'DHAtoB', 'Hyst_An': 'DHAntoB',
             'Hyst_AP':'DHAPtoB','Hyst_APn':'DHAPntoB'},axis=1, inplace=True)
cH_eva=pd.concat([cH_eva,tmp2],axis=1)
tmp2=relDev(tmp,tmp.loc(axis=0)[:,'G'].droplevel([1],axis=0))
tmp2.rename({'Hyst_A': 'DHAtoG', 'Hyst_An': 'DHAntoG',
             'Hyst_AP':'DHAPtoG','Hyst_APn':'DHAPntoG'},axis=1, inplace=True)
cH_eva=pd.concat([cH_eva,tmp2],axis=1)
cH_eva_allVs=cH_eva.unstack().dropna(axis=0).stack()



cEE_eva=cEE.loc[dft['statistics']].sort_index()
cEEm_eva=cEE_eva.groupby(axis=1,level=1).mean()
cEEm_eva.rename({'F':'lu_F_mean','R':'lu_R_mean'},axis=1, inplace=True)
tmp=cEE_eva['u']/cEE_eva['l']
tmp.rename({'F':'lu_F_ratio','R':'lu_R_ratio'},axis=1, inplace=True)
cEEm_eva=pd.concat([cEEm_eva,tmp],axis=1)
tmp=cEEm_eva[['lu_F_mean','lu_R_mean']]
tmp=relDev(tmp,tmp.loc(axis=0)[:,'B'].droplevel([1],axis=0))
tmp.rename({'lu_F_mean':'DEFlutoB','lu_R_mean':'DERlutoB'},axis=1, inplace=True)
cEEm_eva=pd.concat([cEEm_eva,tmp],axis=1)
tmp=cEEm_eva[['lu_F_mean','lu_R_mean']]
tmp=relDev(tmp,tmp.loc(axis=0)[:,'G'].droplevel([1],axis=0))
tmp.rename({'lu_F_mean':'DEFlutoG','lu_R_mean':'DERlutoG'},axis=1, inplace=True)
cEEm_eva=pd.concat([cEEm_eva,tmp],axis=1)
cEEm_eva_allVs=cEEm_eva.unstack().dropna(axis=0).stack()

t2=pd.concat([cs.loc(axis=1)[['Designation','Numsch',
                             'DMWtoA','DMWtoG','WC_gra','WC_vol',
                              'DEFtoB','DEFtoG','DERtoB','DERtoG']],
             cEEm_eva,cH_eva],axis=1)
t2=t2.reset_index()
# t2['Numsch']=t2.Designation.apply(lambda x: '{0}'.format(x[-3:]))
t2=t2.sort_values(['Designation','Variant'])


dft_comb=pd.concat([dft,dft_A],axis=0).sort_index()
dft_comb=pd.concat([dft_comb,cEEm_eva,cH_eva],axis=1)

dft_doda_comb=pd.merge(left=dft_comb,right=doda,
                       left_on='Donor',right_index=True)
dft_doda_comb['Storage_Time']=(dft_doda_comb['Date_SpPrep']-dft_doda_comb['Date_Death']).dt.days

#%% Data-Export
rel_col_com=['Designation','Series','Donor','Origin','Side_LR','Side_pd']
rel_col_geo=['thickness_1','thickness_2','thickness_3',
             'width_1','width_2','width_3','Length']
rel_col_add=['VolTot','RHstore','Mass','Density_app','MassW','WC_gra','WC_vol',
             'WC_gra_rDA','WC_vol_rDA']
rel_col_mec=['statistics','fu','eu_opt',
             'lu_F_mean','DEFlutoB','DEFlutoG','lu_F_ratio',
             'Hyst_AP','Hyst_APn','DHAPntoB','DHAPntoG']
dft_comb_rel = dft_comb[rel_col_com+rel_col_geo+rel_col_add+rel_col_mec]
dft_comb_rel.to_excel(out_full+'.xlsx', sheet_name='Conclusion')

#%%Logging - descriptive

Evac.MG_strlog("\n\n   - org-/dry-Mass and org water content:",
               log_mg, printopt=False)
tmp=dft[['Mass_org','Mass','DMWorg']].loc(axis=0)[:,'G']
Evac.MG_strlog(Evac.str_indent(tmp.to_string(),5),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Max-/Minimum Deviation:",
               log_mg, printopt=False)
tmp=cs.loc(axis=1)[['DMWtoG','DEFtoB','DEFtoG','DERtoB','DERtoG']]
Evac.MG_strlog(Evac.str_indent(tmp.agg(['max','idxmax']).T.to_string(),5),
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(tmp.agg(['min','idxmin']).T.to_string(),5),
               log_mg, printopt=False)


agg_funcs=['mean','std','min','max']
Evac.MG_strlog("\n\n   - Donor data:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(doda.T.to_string(),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n\n     - descriptive:",
               log_mg, printopt=False)
Evac.MG_strlog("\n      - Donors: %d"%doda.count().iloc[0],
               log_mg, printopt=False)
Evac.MG_strlog("\n      - Sex: w={w} - m={m}".format(**doda['Sex'].value_counts()),
               log_mg, printopt=False)
Evac.MG_strlog("\n      - Age:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(doda['Age'].agg(agg_funcs).to_string(),
                               8), log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Test conditions:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(dft[['Temperature_test','Humidity_test']].agg(agg_funcs).to_string()
                               ,5),
               log_mg, printopt=False)
Evac.MG_strlog("\n\n   - Storage conditions:",
               log_mg, printopt=False)
Evac.MG_strlog("\n    -Envisaged to measured (grouped by series)",
               log_mg, printopt=False)
tmp2=dft.groupby(['Series','Variant'])[['RHstore','RHenv']].max()
tmp2=tmp2.unstack(level=0).loc['C':'K']
tmp=tmp2['RHstore']-tmp2['RHenv']
tmp.columns=pd.MultiIndex.from_product([['Difference'],tmp.columns])
tmp2=pd.concat([tmp2,tmp],axis=1)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Difference actual-envisaged",
               log_mg, printopt=False)
# tmp=dft[['Humidity_store','RHenv']]
# tmp=tmp.eval("Humidity_store-RHenv*100").groupby('Variant').agg(agg_funcs)
tmp=dft[['RHstore','RHenv']]
tmp2=tmp.eval("RHstore-RHenv").groupby('Variant').agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Relative deviation actual-envisaged",
               log_mg, printopt=False)
# tmp2=relDev(tmp['Humidity_store'],tmp['RHenv']*100).groupby('Variant').agg(agg_funcs)
tmp2=relDev(tmp['RHstore'],tmp['RHenv']).groupby('Variant').agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)


Evac.MG_strlog("\n\n   - Water content (relative deviation to dry mass):",
               log_mg, printopt=False)
tmp=pd.concat([dft['DMWtoG'],dft_A['DMWtoG']]).unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Water content (gravimetric):",
               log_mg, printopt=False)
tmp=pd.concat([dft['WC_gra'],dft_A['WC_gra']]).unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Water content (volumetric):",
               log_mg, printopt=False)
tmp=pd.concat([dft['WC_vol'],dft_A['WC_vol']]).unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Difference of water content to original (volumetric):",
               log_mg, printopt=False)
# tmp=pd.concat([dft['WC_vol'],dft_A['WC_vol']]).unstack()
# # tmp=tmp.sub(tmp['A'],axis=0).div(tmp['A'],axis=0)
# tmp=tmp.sub(tmp['A'],axis=0)
tmp=dft['WC_vol_toA'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Water content relative deviation to original (volumetric):",
               log_mg, printopt=False)
tmp=dft['WC_vol_rDA'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Youngs Modulus (fixed range-mean load and unload):",
               log_mg, printopt=False)
tmp=cEEm_eva['lu_F_mean'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Youngs Modulus ratio unload to load (fixed range):",
               log_mg, printopt=False)
tmp=cEEm_eva['lu_F_ratio'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Youngs Modulus relative deviation to saturated (fixed range-mean load and unload):",
               log_mg, printopt=False)
tmp=cEEm_eva['DEFlutoB'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
                log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
                log_mg, printopt=False)

Evac.MG_strlog("\n\n   - Youngs Modulus relative deviation to dry (fixed range-mean load and unload):",
               log_mg, printopt=False)
tmp=cEEm_eva['DEFlutoG'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
                log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
                log_mg, printopt=False)


Evac.MG_strlog("\n\n   - Normed hysteresis area (from preload to equivalent):",
               log_mg, printopt=False)
tmp=cH_eva['Hyst_APn'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)


Evac.MG_strlog("\n\n   - Normed hysteresis area relative deviation to saturated (from preload to equivalent):",
               log_mg, printopt=False)
tmp=cH_eva['DHAPntoB'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)


Evac.MG_strlog("\n\n   - Normed hysteresis area relative deviation to dry (from preload to equivalent):",
               log_mg, printopt=False)
tmp=cH_eva['DHAPntoG'].unstack()
Evac.MG_strlog("\n    - to variants:",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(Evac.agg_add_ci(tmp, agg_funcs).to_string(),6),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Ratio of ad- to desorption:",
               log_mg, printopt=False)
tmp2=tmp.eval('''
              DeAd_WST = L / B
              DeAd_100 = K / C
              DeAd_090 = J / D
              DeAd_075 = I / E
              DeAd_060 = H / F
              ''')
tmp2=Evac.agg_add_ci(tmp2.loc(axis=1)[tmp2.columns.str.contains("DeAd_")],agg_funcs)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string(),6),
               log_mg, printopt=False)

#%% 1st Eva
#%%% Statistic
alpha=0.05
# alpha=0.10
# Testtype
stat_ttype_parametric=False

if stat_ttype_parametric:
    mpop="ANOVA"
    # Tukey HSD test:
    MComp_kws={'do_mcomp_a':2, 'mcomp':'TukeyHSD', 'mpadj':'', 
                'Ffwalpha':1, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    # # Independendent T-test:
    # MComp_kws={'do_mcomp_a':2, 'mcomp':'ttest_ind', 'mpadj':'holm', 
    #            'Ffwalpha':2, 'mkws':{'equal_var': False, 'nan_policy': 'omit'},
    #            'add_T_ind':3, 'add_out':True}
    mcorr="pearson"
else:
    mpop="Kruskal-Wallis H-test"
    # Mann-Whitney U test: (two independent)
    MComp_kws={'do_mcomp_a':2, 'mcomp':'mannwhitneyu', 'mpadj':'holm', 
                'Ffwalpha':1, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    # # Wilcoxon W test: (two related-or difference)
    # MComp_kws={'do_mcomp_a':2, 'mcomp':'wilcoxon', 'mpadj':'holm', 
    #             'Ffwalpha':2, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    mcorr="spearman"
    MCompdf_kws={'do_mcomp_a':1, 'mcomp':'mannwhitneyu', 'mpadj':'holm', 
                 'Ffwalpha':1, 'mkws':{}}
    
#%%% Normalverteilung
# fig, ax1 = plt.subplots()
# tmp=pd.concat([dft['DMWtoG'],dft_A['DMWtoG']]).unstack()
# tmp=tmp/tmp.max(axis=0)
# sns.histplot(tmp,
#              stat='count', bins=20,
#              ax=ax1, kde=True, color=sns.color_palette()[0],
#              edgecolor=None, legend = True, alpha=0.25)
# ax1.set_title('%sHistogram of water content to variant'%name_Head)
# ax1.set_xlabel('Normed parameter / -')
# ax1.set_ylabel('Count / -')
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=out_full+'-Hist-DMWtoG',**plt_Fig_dict)

def stat_agg(pdo, sfunc=stats.normaltest, mkws={}):
    t = pdo.agg(sfunc, **mkws)
    out = pd.Series({'F':t[0], 'p':t[1]})
    return out
tmp=pd.concat([dft,dft_A])
tmp.groupby('Variant')['DMWtoG'].apply(stat_agg,**{'sfunc':stats.shapiro})


#%%% Gruppenanalyse
#%%%% Varianzanalyse Varianten
Evac.MG_strlog("\n "+"="*100, log_mg, 1, printopt=False)
Evac.MG_strlog("\n Variants: ", log_mg, 1, printopt=False)
Evac.MG_strlog("\n  %s water content based on dry mass vs. variant:"%mpop,
               log_mg, 1, printopt=False)
tmp=pd.concat([dft,dft_A])
txt,_,T = Evac.group_ANOVA_MComp(df=pd.concat([dft,dft_A]).reset_index(),
                                 groupby='Variant', 
                                 ano_Var='DMWtoG',
                                 group_str='Variant', ano_str='DMWtoG',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Evac.MG_strlog("\n   -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
               log_mg, 1, printopt=False)
Evac.MG_strlog(Evac.str_indent(d,6), log_mg,1, printopt=False)
tmp2=stats.wilcoxon(tmp.loc[idx[:,['A','B']],'DMWtoG'].unstack().eval("B-A"),
               alternative='greater')
Evac.MG_strlog(Evac.str_indent("DMWtoG_B greater A: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)

Evac.MG_strlog("\n\n  %s volumetric water content vs. variant:"%mpop,
               log_mg, 1, printopt=False)
tmp=pd.concat([dft,dft_A])
txt,_,T = Evac.group_ANOVA_MComp(df=pd.concat([dft,dft_A]).reset_index(),
                                 groupby='Variant', 
                                 ano_Var='WC_vol',
                                 group_str='Variant', ano_str='WC_vol',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Evac.MG_strlog("\n   -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
               log_mg, 1, printopt=False)
Evac.MG_strlog(Evac.str_indent(d,6), log_mg,1, printopt=False)

tmp2=stats.wilcoxon(tmp.loc[idx[:,['A','B']],'WC_vol'].unstack().eval("B-A"),
               alternative='greater')
Evac.MG_strlog(Evac.str_indent("WC_vol B greater A: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)
tmp2=stats.wilcoxon(tmp.loc[idx[:,['A','C']],'WC_vol'].unstack().eval("C-A"))
Evac.MG_strlog(Evac.str_indent("WC_vol C equal A: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)



Evac.MG_strlog("\n\n  %s YM deviation (to saturated) vs. variant:"%mpop,
               log_mg, 1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=t2, groupby='Variant', 
                                 ano_Var='DEFlutoB',
                                 group_str='Variant', ano_str='Dev. YM to B',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Evac.MG_strlog("\n  -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
                log_mg,1,printopt=False)
Evac.MG_strlog(Evac.str_indent(d,6), log_mg,1,printopt=False)

Evac.MG_strlog("\n\n  %s YM deviation (to saturated) vs. variant (only desorption)):"%mpop,
               log_mg, 1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=t2[t2.Variant<='G'], groupby='Variant', 
                                 ano_Var='DEFlutoB',
                                 group_str='Variant (desorption)', 
                                 ano_str='Dev. YM to B',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Evac.MG_strlog("\n   -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
                log_mg,1,printopt=False)
Evac.MG_strlog(Evac.str_indent(d,6), log_mg,1,printopt=False)

tmp2=stats.wilcoxon(cEEm_eva.loc[idx[:,'C'],'DEFlutoB'])
Evac.MG_strlog(Evac.str_indent("DEtoB C equal B: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)
tmp2=stats.wilcoxon(cEEm_eva.loc[idx[:,'D'],'DEFlutoB'])
Evac.MG_strlog(Evac.str_indent("DEtoB D equal B: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)
tmp2=stats.wilcoxon(cEEm_eva.loc[idx[:,'G'],'DEFlutoB'])
Evac.MG_strlog(Evac.str_indent("DEtoB G equal B: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)

Evac.MG_strlog("\n\n  %s YM deviation (to dry) vs. variant:"%mpop,
               log_mg, 1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=t2, groupby='Variant', 
                                 ano_Var='DEFlutoG',
                                 group_str='Variant', ano_str='Dev. YM to B',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Evac.MG_strlog("\n  -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
                log_mg,1,printopt=False)
Evac.MG_strlog(Evac.str_indent(d,6), log_mg,1,printopt=False)
tmp2=stats.wilcoxon(cEEm_eva.loc[idx[:,'L'],'DEFlutoG'])
Evac.MG_strlog(Evac.str_indent("DEtoG L equal G: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)



Evac.MG_strlog("\n\n  %s mean YM vs. variant:"%mpop,
               log_mg, 1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=t2, groupby='Variant', 
                                 ano_Var='lu_F_mean',
                                 group_str='Variant', ano_str='E_lu_F_mean',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Evac.MG_strlog("\n   -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
               log_mg,1,printopt=False)
Evac.MG_strlog(Evac.str_indent(d,6), log_mg,1,printopt=False)
tmp2=stats.wilcoxon((cEEm_eva.loc[idx[:,['C','B']],'lu_F_mean']).unstack().diff(axis=1).iloc[:,1])
Evac.MG_strlog(Evac.str_indent("E C equal B: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)
tmp2=stats.wilcoxon((cEEm_eva.loc[idx[:,['G','B']],'lu_F_mean']).unstack().diff(axis=1).iloc[:,1])
Evac.MG_strlog(Evac.str_indent("E G equal B: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)
tmp2=stats.wilcoxon((cEEm_eva.loc[idx[:,['L','B']],'lu_F_mean']).unstack().diff(axis=1).iloc[:,1].dropna())
Evac.MG_strlog(Evac.str_indent("E L equal B: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)

Evac.MG_strlog("\n\n  %s normed hysteris area deviation (to saturated) vs. variant (nan's  dropped):"%mpop,
               log_mg, 1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=t2[['Variant','DHAPntoB']].dropna(axis=0), 
                                 groupby='Variant', 
                                 ano_Var='DHAPntoB',
                                 group_str='Variant', ano_str='Dev. HAn to B',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Evac.MG_strlog("\n  -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
                log_mg,1,printopt=False)
Evac.MG_strlog(Evac.str_indent(d,6), log_mg,1,printopt=False)
tmp2=stats.wilcoxon(cH_eva.loc[idx[:,'C'],'DHAPntoB'])
Evac.MG_strlog(Evac.str_indent("Hystereseisarea C equal B: %s"%str(tmp2),6), 
               log_mg,1, printopt=False)

Evac.MG_strlog("\n\n  %s Conclusion:"%mpop,
               log_mg, 1, printopt=False)
tmp3=group_ANOVA_MComp_multi(df=dft_comb,
                group_main='Variant',group_sub=[],
               ano_Var=['WC_vol','WC_vol_rDA',
                        'lu_F_mean','DEFlutoB',
                        'Hyst_APn','DHAPntoB'], 
               mpop=mpop, alpha=alpha, **MCompdf_kws)
Evac.MG_strlog(Evac.str_indent(tmp3.loc(axis=1)['DF1':'H0'].to_string()),
               log_mg,1, printopt=False)


#%%%% Varianzanalysen Spender


Evac.MG_strlog("\n\n "+"="*100, log_mg, 1, printopt=False)
Evac.MG_strlog("\n Donordata: ", log_mg, 1, printopt=False)
# Evac.MG_strlog("\n  %s water content (org) vs. donor:"%mpop,
#                log_mg, 1, printopt=False)
# txt,_,T = Evac.group_ANOVA_MComp(df=csdoda[csdoda.Variant=='B'], groupby='Donor', 
#                                  ano_Var='DMWorg',
#                                  group_str='Donor', ano_str='DMWorg',
#                                  mpop=mpop, alpha=alpha,  
#                                  group_ren=doda.Naming.to_dict(), **MComp_kws)
# Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)

# Evac.MG_strlog("\n  %s water loss vs. donor:"%mpop,
#                log_mg, 1, printopt=False)
# txt,_,T = Evac.group_ANOVA_MComp(df=csdoda[csdoda.Variant=='B'], groupby='Donor', 
#                                  ano_Var='DMWtoG',
#                                  group_str='Donor', ano_str='DMWBtoG',
#                                  mpop=mpop, alpha=alpha,  
#                                  group_ren=doda.Naming.to_dict(), **MComp_kws)
# Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
Evac.MG_strlog("\n  %s water content vs. donor:"%mpop,
               log_mg, 1, printopt=False)
# txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'A'], groupby='Donor', 
#                                  ano_Var='WC_vol',
#                                  group_str='Donor', ano_str='A: WC',
#                                  mpop=mpop, alpha=alpha,  
#                                  group_ren=doda.Naming.to_dict(), **MComp_kws)
# Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
# txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'B'], groupby='Donor', 
#                                  ano_Var='WC_vol',
#                                  group_str='Donor', ano_str='B: WC',
#                                  mpop=mpop, alpha=alpha,  
#                                  group_ren=doda.Naming.to_dict(), **MComp_kws)
# Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
# txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'L'], groupby='Donor', 
#                                  ano_Var='WC_vol',
#                                  group_str='Donor', ano_str='L: WC',
#                                  mpop=mpop, alpha=alpha,  
#                                  group_ren=doda.Naming.to_dict(), **MComp_kws)
# Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
tmp2=group_ANOVA_MComp_multi(df=dft_comb.unstack(), 
                           group_main='Donor', 
                           group_sub=['A','B','L'],
                           ano_Var=['WC_vol'],
                           mpop=mpop, alpha=alpha,  
                           group_ren={}, **MCompdf_kws)
Evac.MG_strlog(Evac.str_indent(tmp2.to_string()),log_mg,1, printopt=False)

Evac.MG_strlog("\n  %s deviation water content vs. donor:"%mpop,
               log_mg, 1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'B'], groupby='Donor', 
                                 ano_Var='WC_vol_rDA',
                                 group_str='Donor', ano_str='B: WC to fresh',
                                 mpop=mpop, alpha=alpha,  
                                 group_ren=doda.Naming.to_dict(), **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'C'], groupby='Donor', 
                                 ano_Var='WC_vol_rDA',
                                 group_str='Donor', ano_str='C: WC to fresh',
                                 mpop=mpop, alpha=alpha,  
                                 group_ren=doda.Naming.to_dict(), **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'L'], groupby='Donor', 
                                 ano_Var='WC_vol_rDA',
                                 group_str='Donor', ano_str='L: WC to fresh',
                                 mpop=mpop, alpha=alpha,  
                                 group_ren=doda.Naming.to_dict(), **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)

Evac.MG_strlog("\n  %s deviation Youngs Modulus vs. donor:"%mpop,
               log_mg, 1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'G'], groupby='Donor', 
                                 ano_Var='DEFlutoB',
                                 group_str='Donor', ano_str='G: YM to saturated',
                                 mpop=mpop, alpha=alpha,  
                                 group_ren=doda.Naming.to_dict(), **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'L'][['Donor','DEFlutoG']].dropna(axis=0), 
                                 groupby='Donor', 
                                 ano_Var='DEFlutoG',
                                 group_str='Donor', ano_str='L: YM to dry (NaN dropped',
                                 mpop=mpop, alpha=alpha,  
                                 group_ren=doda.Naming.to_dict(), **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=dft_comb.loc(axis=0)[:,'C'], groupby='Donor', 
                                 ano_Var='DEFlutoB',
                                 group_str='Donor', ano_str='C: YM to saturated',
                                 mpop=mpop, alpha=alpha,  
                                 group_ren=doda.Naming.to_dict(), **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1, printopt=False)


Evac.MG_strlog("\n\n  Hypothesistests to donordata (H-test):",
               log_mg, 1, printopt=False)
tmp3=Multi_conc(df=dft_comb.unstack(),group_main='Donor', anat='VA',
               stdict={'WC_vol':['A','B','L'],'WC_vol_rDA':['B','C','L'],
                       'lu_F_mean':['B'],'DEFlutoB':['C','G','L'],
                       'Hyst_APn':['B'],'DHAPntoB':['C','G','L']}, 
               met=mpop, alpha=alpha, kws=MCompdf_kws)
Evac.MG_strlog(Evac.str_indent(tmp3.loc(axis=1)['DF1':'H0'].to_string()),
               log_mg,1, printopt=False)
Evac.MG_strlog("\n  -> Multicomparision (Mann-Whitney-U)):",
               log_mg, 1, printopt=False)
for i in tmp3.loc[tmp3.H0 == False].index:
    txt="{}:\n{}".format(i,tmp3.loc[i,'MCP'],)
    Evac.MG_strlog(Evac.str_indent(txt,5),
                   log_mg,1, printopt=False)
    
Evac.MG_strlog("\n\n  Hypothesistests to location (proximal/distal) (Mann-Whitney-U/Wilcoxon):",
               log_mg, 1, printopt=False)
tmp2=Multi_conc(df=dft_comb.unstack(),group_main='Side_pd', anat='HT',
               stdict={'WC_vol':['A','B','L'],'WC_vol_rDA':['B','C','L'],
                       'lu_F_mean':['B'],'DEFlutoB':['C','G','L'],
                       'Hyst_APn':['B'],'DHAPntoB':['C','G','L']}, 
               met='mannwhitneyu', alpha=alpha, 
               rel=False, rel_keys=['Donor','Side_LR'], kws={})
tmp3=Multi_conc(df=dft_comb.unstack(),group_main='Side_pd', anat='HT',
               stdict={'WC_vol':['A','B','L'],'WC_vol_rDA':['B','C','L'],
                       'lu_F_mean':['B'],'DEFlutoB':['C','G','L'],
                       'Hyst_APn':['B'],'DHAPntoB':['C','G','L']}, 
               met='wilcoxon', alpha=alpha, 
               rel=True, rel_keys=['Donor','Side_LR'], kws={})
tmp=pd.concat([tmp2, tmp3], axis=1, keys=['mannwhitneyu', 'wilcoxon'])
Evac.MG_strlog(Evac.str_indent(tmp.to_string()),
               log_mg,1, printopt=False)

Evac.MG_strlog("\n\n  Hypothesistests to side (left/right) (Mann-Whitney-U/Wilcoxon):",
               log_mg, 1, printopt=False)
tmp2=Multi_conc(df=dft_comb.unstack(),group_main='Side_LR', anat='HT',
               stdict={'WC_vol':['A','B','L'],'WC_vol_rDA':['B','C','L'],
                       'lu_F_mean':['B'],'DEFlutoB':['C','G','L'],
                       'Hyst_APn':['B'],'DHAPntoB':['C','G','L']}, 
               met='mannwhitneyu', alpha=alpha, 
               rel=False, rel_keys=['Donor','Side_pd'], kws={})
tmp3=Multi_conc(df=dft_comb.unstack(),group_main='Side_LR', anat='HT',
               stdict={'WC_vol':['A','B','L'],'WC_vol_rDA':['B','C','L'],
                       'lu_F_mean':['B'],'DEFlutoB':['C','G','L'],
                       'Hyst_APn':['B'],'DHAPntoB':['C','G','L']}, 
               met='wilcoxon', alpha=alpha, 
               rel=True, rel_keys=['Donor','Side_pd'], kws={})
tmp=pd.concat([tmp2, tmp3], axis=1, keys=['mannwhitneyu', 'wilcoxon'])
Evac.MG_strlog(Evac.str_indent(tmp.to_string()),
               log_mg,1, printopt=False)

#%%%% Varianzanalysen Serie
# tmp=Hypo_test(df=dft.loc(axis=0)[:,'B'], groupby='Series', ano_Var='WC_vol', 
#             mcomp='wilcoxon', rel=True, rel_keys=['Donor','Side_pd'],add_out = True)
# tmp=Hypo_test(df=dft.loc(axis=0)[:,'B'].droplevel(1), groupby='Series', 
#               ano_Var='WC_vol', 
#               mcomp='mannwhitneyu', rel=False, rel_keys=[],add_out = True)
    
# tmp=Hypo_test(df=csdoda[csdoda.Variant=='B'], groupby='Side_LR', ano_Var='DMWtoG', 
#           mcomp='ttest_rel', rel=True, rel_keys=['Donor','Side_pd'],add_out = True)
# tmp=Hypo_test(df=csdoda[csdoda.Variant=='B'], groupby='Side_LR', ano_Var='DMWtoG', 
#           mcomp='ttest_ind', rel=False, rel_keys=['Donor','Side_pd'],add_out = True)
# tmp=Hypo_test(df=csdoda[csdoda.Variant=='B'], groupby='Side_LR', ano_Var='DMWtoG', 
#               mkws={'equal_var':False},
#           mcomp='ttest_ind', rel=False, rel_keys=['Donor','Side_pd'],add_out = True)
# tmp=pd.concat([dft,dft_A])['DMWtoG'].reset_index().query("Variant=='A' or Variant=='C'")
# tmp=Hypo_test(df=tmp.query("Variant=='A'or Variant=='C'"), 
#               groupby='Variant', ano_Var='DMWtoG', 
#               mcomp='ttest_rel', rel=True, rel_keys=['Key'],add_out = True)

# dft_comb=pd.concat([dft,dft_A],axis=0).sort_index()
# tmp=Hypo_test_multi(df=dft_comb.unstack(), group_main='Series', group_sub=['A','B'],
#                     ano_Var=['WC_vol','WC_vol_rDA'],
#                     mcomp='mannwhitneyu', mkws={},
#                     rel=False, rel_keys=[])

Evac.MG_strlog("\n\n "+"="*100, log_mg, 1, printopt=False)
Evac.MG_strlog("\n Repeatability (Series dependence): ", log_mg, 1, printopt=False)
tmp=Hypo_test_multi(df=dft_comb.unstack(), group_main='Series', 
                    group_sub=['A'],
                    ano_Var=['WC_vol'],
                    mcomp='mannwhitneyu', mkws={},
                    rel=False, rel_keys=[])
tmp2=Hypo_test_multi(df=dft_comb.unstack(), group_main='Series', 
                    # group_sub=['B','C','L'],
                     group_sub=['B','C','D','E','F','H','I','J','K','L'],
                    ano_Var=['WC_vol','WC_vol_rDA'],
                    mcomp='mannwhitneyu', mkws={},
                    rel=False, rel_keys=[])
tmp=pd.concat([tmp,tmp2],axis=0)
tmp2=Hypo_test_multi(df=dft_comb.unstack(), group_main='Series', 
                     # group_sub=['C','G','L'],
                     group_sub=['C','D','E','F','G','H','I','J','K','L'],
                    ano_Var=['DEFlutoB','DHAPntoB'],
                    mcomp='mannwhitneyu', mkws={},
                    rel=False, rel_keys=[])
tmp=pd.concat([tmp,tmp2],axis=0)
Evac.MG_strlog(Evac.str_indent(tmp.sort_index().to_string(),3),
               log_mg,1, printopt=False)

Evac.MG_strlog("\n   - DEFlutoB - Variant C and D: ", log_mg, 1, printopt=False)
tmp=dft_comb.query("Series=='1'")['DEFlutoB']
tmp2=stats.mannwhitneyu(tmp.loc(axis=0)[:,'C'],tmp.loc(axis=0)[:,'D'])
Evac.MG_strlog(Evac.str_indent('- Series 1: '+str(tmp2),6),
               log_mg,1, printopt=False)
tmp=dft_comb.query("Series=='2'")['DEFlutoB']
tmp2=stats.mannwhitneyu(tmp.loc(axis=0)[:,'C'],tmp.loc(axis=0)[:,'D'])
Evac.MG_strlog(Evac.str_indent('- Series 2: '+str(tmp2),6),
               log_mg,1, printopt=False)

# det_met_CD_rep='SM-RRT'
det_met_CD_rep='SD'
Evac.MG_strlog("\n\n  - Critical Differences (Method:%s):"%det_met_CD_rep,
               log_mg,1, printopt=False)
# Evac.MG_strlog("\n\n    - Water content:",
#                log_mg,1, printopt=False)
# for i in ['A','B','C','D','E','F','H','I','J','K','L']:
#     tmp='- %s: '%i+CD_rep(dft_comb.loc(axis=0)[:,i], groupby='Series', 
#                         var='WC_vol', det_met=det_met_CD_rep, outtype='txt')
#     Evac.MG_strlog(Evac.str_indent(tmp,6),
#                    log_mg,1, printopt=False)
# Evac.MG_strlog("\n\n    - Relative deviation of water content to fresh:",
#                log_mg,1, printopt=False)
# for i in ['B','C','D','E','F','H','I','J','K','L']:
#     tmp='- %s: '%i+CD_rep(dft_comb.loc(axis=0)[:,i], groupby='Series', 
#                         var='WC_vol_rDA', det_met=det_met_CD_rep,outtype='txt')
#     Evac.MG_strlog(Evac.str_indent(tmp,6),
#                    log_mg,1, printopt=False)
# Evac.MG_strlog("\n\n    - Relative deviation of YM to saturated:",
#                log_mg,1, printopt=False)
# # tmp='- G: '+CD_rep(dft_comb.loc(axis=0)[:,'G'], groupby='Series', 
# #                     var='DEFlutoB', det_met=det_met_CD_rep, outtype='txt')
# for i in ['C','D','E','F','H','I','J','K','L']:
#     tmp='- %s: '%i+CD_rep(dft_comb.loc(axis=0)[:,i], groupby='Series', 
#                         var='DEFlutoB', det_met=det_met_CD_rep,outtype='txt')
#     Evac.MG_strlog(Evac.str_indent(tmp,6),
#                    log_mg,1, printopt=False)
# Evac.MG_strlog("\n\n    - Relative deviation of normed hysteresis area to saturated:",
#                log_mg,1, printopt=False)
# for i in ['C','D','E','F','H','I','J','K','L']:
#     tmp='- %s: '%i+CD_rep(dft_comb.loc(axis=0)[:,i], groupby='Series', 
#                         var='DHAPntoB', det_met=det_met_CD_rep,outtype='txt')
#     Evac.MG_strlog(Evac.str_indent(tmp,6),
#                    log_mg,1, printopt=False)

tmp3=Multi_conc(df=dft_comb.unstack(),group_main='Series', anat='CD',
                stdict={'WC_vol':['A','B','C','D','E','F','H','I','J','K','L'],
                        'WC_vol_rDA':['B','C','D','E','F','H','I','J','K','L'],
                        'lu_F_mean':['B','C','D','E','F','G','H','I','J','K','L'],
                        'DEFlutoB':['C','D','E','F','H','I','J','K','L'],
                        'Hyst_APn':['B','C','D','E','F','G','H','I','J','K','L'],
                        'DHAPntoB':['C','D','E','F','H','I','J','K','L']},
                met=det_met_CD_rep)
Evac.MG_strlog(Evac.str_indent(tmp3.to_string(),6),
               log_mg,1, printopt=False)

Evac.MG_strlog("\n\n  - Descriptive (Mean-values):",
               log_mg,1, printopt=False)
tmp=dft_comb.reset_index().groupby(['Variant','Series'])[['RHstore','WC_vol','WC_vol_rDA','DEFlutoB']].mean().unstack()
Evac.MG_strlog(Evac.str_indent(tmp.to_string(),4),log_mg,1, printopt=False)

Evac.MG_strlog("\n\n  - Regression (exponential):",
               log_mg,1, printopt=False)
Evac.MG_strlog("\n\n    - Parameters:",
               log_mg,1, printopt=False)
tmp=pd.concat([cs[['Series','WC_vol_rDA']],cEEm_eva['DEFlutoB']],axis=1)
tmp2={'Desorption_Complete':tmp.query("Variant<='G'"),
      'Desorption_Series1':tmp.query("Variant<='G'").query("Series=='1'"),
      'Desorption_Series2':tmp.query("Variant<='G'").query("Series=='2'"),
      'Adsorption_Complete':tmp.query("Variant>='G'"),
      'Adsorption_Series1':tmp.query("Variant>='G'").query("Series=='1'"),
      'Adsorption_Series2':tmp.query("Variant>='G'").query("Series=='2'")}
tmp3=pd.DataFrame([], columns=['a','b','c'], dtype='float64')
for i in tmp2.keys():
    if 'Desorption' in i:
        tmp3.loc[i]=pd.Series(Regfitret(pdo=tmp2[i],
                    x='WC_vol_rDA', y='DEFlutoB',
                    name='exponential_x0', guess = dict(a=0, b=0.01, c=-1),
                    xl=r'D_{\Phi_{vol},fresh}',yl=r'D_{E,1,des}', 
                    t_form='{a:.3e},{b:.3e},{c:.3f}'))['PD']
    if 'Adsorption' in i:
        tmp3.loc[i]=pd.Series(Regfitret(pdo=tmp2[i],
                    x='WC_vol_rDA', y='DEFlutoB',
                    name='exponential', guess = dict(a=0.01, b=0.01, c=1),
                    xl=r'D_{\Phi_{vol},fresh}',yl=r'D_{E,1,des}', 
                    t_form='{a:.3e},{b:.3e},{c:.3f}'))['PD']
tmp3.index=tmp3.index.str.split('_',expand=True)
tmp3.index.names=['DesAds','Selection']
Evac.MG_strlog(Evac.str_indent(tmp3.to_string(),6),log_mg,1, printopt=False)
tmp2=relDev(tmp3.loc(axis=0)[:,['Series1','Series2']],
            tmp3.loc(axis=0)[:,'Complete'].droplevel(1,axis=0))
Evac.MG_strlog("\n\n    - relative deviation of parameters:",
               log_mg,1, printopt=False)
Evac.MG_strlog(Evac.str_indent(tmp3.to_string(),6),log_mg,1, printopt=False)


#%%% Plots
fig, ax1 = plt.subplots()
tmp=dft.loc(axis=1)[dft.columns.str.contains(r'^E_inc_F_.*_meanwoso$')].copy(deep=True)
tmp.columns=tmp.columns.to_series().apply(lambda x: x.split('_')[-2])
tmp.columns.name='Methods'
tmp2=pd.melt(tmp.loc(axis=0)[:,['B','G']].reset_index(), 
           id_vars='Variant', value_vars=tmp.columns)
ax = sns.barplot(data=tmp2, 
                 x='Methods',y='value', hue='Variant', ax=ax1, 
                 seed=0, errwidth=1, capsize=0.2)
ax1.set_title('%sYoungs moduli by determination variant (only load and variants B and G)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.set_ylabel(r'$E$ / MPa')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylim([0,4500])
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-YM-allMethods',**plt_Fig_dict)

# cs.boxplot(column='DMWtoG',by='Variant')
fig, ax1 = plt.subplots()
ax = sns.boxplot(data=dft.DMWtoG.unstack(),ax=ax1)
ax1.set_title('%sWater content based on dry mass of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{mass,Water}$ / -')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-DMWtoG',**plt_Fig_dict)

fig, ax1 = plt.subplots()
tmp=pd.concat([dft['DMWtoG'],dft_A['DMWtoG']]).unstack()
# tmp=pd.concat([dft.query("Variant!='G'")['DMWtoG'],dft_A['DMWtoG']]).unstack()
# ax = sns.barplot(data=tmp,ax=ax1, seed=0,
#                  errwidth=1, capsize=0.4)
ax = sns.boxplot(data=tmp,ax=ax1,
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(data=tmp, ax=ax1, 
                   dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.axvline(0.5,0,1, color='grey',ls='--')
ax1.text(5,1.1,"Desorption",ha='center',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax1.axvline(6.0,0,1, color='grey',ls='--')
ax1.text(7,1.1,"Adsorption",ha='center',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax1.set_title('%sWater content based on dry mass of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{mass,Water}$ / -')
ax1.set_yscale("log")
ax1.set_yticks([0.1,1.0])
ax1.grid(True, which='both', axis='y')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-DMWtoG-ext',**plt_Fig_dict)

fig, ax1 = plt.subplots()
tmp=pd.concat([dft['WC_vol'],dft_A['WC_vol']]).unstack()
# tmp=pd.concat([dft.query("Variant!='G'")['DMWtoG'],dft_A['DMWtoG']]).unstack()
# ax = sns.barplot(data=tmp,ax=ax1, seed=0,
#                  errwidth=1, capsize=0.4)
ax = sns.boxplot(data=tmp,ax=ax1,
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(data=tmp, ax=ax1, 
                   dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.axvline(0.5,0,1, color='grey',ls='--')
ax1.text(5,0.5,"Desorption",ha='center',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax1.axvline(6.0,0,1, color='grey',ls='--')
ax1.text(7,0.5,"Adsorption",ha='center',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax1.set_title('%sVolumetric water content of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$\Phi_{vol,Water}=V_{Water}/V_{Total}$ / -')
ax1.set_yscale("log")
ax1.grid(True, which='both', axis='y')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-WC_Vol-ext',**plt_Fig_dict)

fig, ax1 = plt.subplots()
# tmp=pd.concat([dft['WC_vol'],dft_A['WC_vol']]).unstack()
# tmp=tmp.sub(tmp['A'],axis=0).loc(axis=1)[idx['B':'L']]
tmp=dft['WC_vol_toA'].unstack()
# tmp=pd.concat([dft.query("Variant!='G'")['DMWtoG'],dft_A['DMWtoG']]).unstack()
# ax = sns.barplot(data=tmp,ax=ax1, seed=0,
#                  errwidth=1, capsize=0.4)
ax = sns.boxplot(data=tmp,ax=ax1,
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(data=tmp, ax=ax1, 
                   dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%sDifference of volumetric water content to original'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$\Delta_{\Phi_{vol,Water}}$ / -')
ax1.grid(True, which='both', axis='y')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-WC_Vol-dtoA-ext',**plt_Fig_dict)



# fig, ax1 = plt.subplots()
# ax = sns.boxplot(data=dft.DMWtoA.unstack(),ax=ax1)
# ax1.set_title('%sWater content based on original mass of the different manipulation variants'%name_Head)
# ax1.set_xlabel('Variant / -')
# ax1.set_ylabel(r'$D_{mass,Water}$ / -')
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=out_full+'-Box-DMWtoA',**plt_Fig_dict)

fig, ax1 = plt.subplots()
ax = sns.boxplot(data=cEEm_eva['DEFlutoB'].unstack(),ax=ax1)
ax1.set_title('%sDeviation of Youngs Modulus (fixed range) based on water storaged\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-DEFlutoB',**plt_Fig_dict)

fig, ax1 = plt.subplots()
tmp=cEEm_eva['DEFlutoB'].unstack()
# tmp=cEEm_eva_allVs['DEFlutoB'].unstack()
# ax = sns.barplot(data=tmp,ax=ax1, seed=0,
#                  errwidth=1, capsize=0.4)
ax = sns.boxplot(data=tmp,ax=ax1,
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(data=tmp, ax=ax1, 
                   dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.text(3,2.0,"Desorption",ha='center',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax1.axvline(5.0,0,1, color='grey',ls='--')
ax1.text(7,2.0,"Adsorption",ha='center',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax1.set_title('%sDeviation of Youngs Modulus (fixed range) based on water storaged\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-DEFlutoB-ext',**plt_Fig_dict)

fig, ax1 = plt.subplots()
ax = sns.boxplot(data=cEEm_eva['DEFlutoG'].unstack(),ax=ax1)
ax1.set_title('%sDeviation of Youngs Modulus (fixed range) based on dry mass\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-DEFlutoG',**plt_Fig_dict)


fig, ax1 = plt.subplots()
ax = sns.boxplot(data=cEEm_eva['lu_F_ratio'].unstack(),ax=ax1)
ax1.set_title('%sYoungs Modulus (fixed range) ratio load to unload\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$E_{u}/E_{l}$ / -')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-RatioEutol',**plt_Fig_dict)

fig, ax1 = plt.subplots()
ax = sns.boxplot(data=cH_eva['Hyst_APn'].unstack(),ax=ax1)
ax1.set_title('%sNormed hysteresis area of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$A_{Hyst,norm}$ / -')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-Hyst_An',**plt_Fig_dict)

fig, ax1 = plt.subplots()
tmp = cH_eva['DHAPntoB'].unstack()
# tmp = cH_eva['DHAPntoB'].unstack().dropna(axis=0)
# ax = sns.boxplot(data=cH_eva['DHAntoB'].unstack(),ax=ax1)
ax = sns.boxplot(data=tmp,ax=ax1,
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(data=tmp, ax=ax1, 
                   dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.text(3,1.7,"Desorption",ha='center',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax1.axvline(5.0,0,1, color='grey',ls='--')
ax1.text(7,1.7,"Adsorption",ha='center',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax1.set_title('%sDeviation of normed hysteresis area based on saturated\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{H_{n},sat}$ / -')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-DHAntoB',**plt_Fig_dict)

fig, ax1 = plt.subplots()
# ax = sns.barplot(data=cH_eva['DHAntoB'].unstack(),ax=ax1, seed=0,
#                  errwidth=1, capsize=0.4)
ax = sns.barplot(data=cH_eva['DHAPntoB'].unstack(),ax=ax1, seed=0,
                  errwidth=1, capsize=0.4)
ax1.set_title('%sDeviation of normed hysteresis area based on water storaged\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{Hyst,norm}$ / -')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Bar-DHAntoB',**plt_Fig_dict)


tmp=dft.reset_index().sort_values(['Numsch','Variant'])
fig, ax1 = plt.subplots()
ax = sns.scatterplot(y='DMWtoA',x='Variant',hue='Numsch',
                     data=tmp,ax=ax1)
ax = sns.lineplot(y='DMWtoA',x='Variant',hue='Numsch',
                  data=tmp,ax=ax1, legend=False)
ax1.set_title('%sRelative deviation based on original mass of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{mass,Water}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SL-DMWtoA-Var',**plt_Fig_dict)

fig, ax1 = plt.subplots()
ax = sns.scatterplot(y='DMWtoG',x='Variant',hue='Numsch',
                     data=tmp,ax=ax1)
ax = sns.lineplot(y='DMWtoG',x='Variant',hue='Numsch',
                  data=tmp,ax=ax1, legend=False)
ax1.set_title('%sWater content based on dry mass of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{mass,Water}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SL-DMWtoG-Var',**plt_Fig_dict)

tmp=dft_comb.reset_index().sort_values(['Numsch','Variant'])
fig, ax1 = plt.subplots()
ax = sns.scatterplot(y='WC_vol',x='Variant',hue='Numsch',
                     data=tmp,ax=ax1)
ax = sns.lineplot(y='WC_vol',x='Variant',hue='Numsch',
                  data=tmp,ax=ax1, legend=False)
ax1.set_title('%sWater content of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$\Phi_{vol}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SL-WC_vol-Var',**plt_Fig_dict)

tmp=dft_comb.reset_index().sort_values(['Numsch','Variant'])
fig, ax1 = plt.subplots()
ax = sns.scatterplot(y='WC_vol_rDA',x='Variant',hue='Numsch',
                     data=tmp,ax=ax1)
ax = sns.lineplot(y='WC_vol_rDA',x='Variant',hue='Numsch',
                  data=tmp,ax=ax1, legend=False)
ax1.set_title('%sRelative deviation of water content based on original one to manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{\Phi_{vol,org}}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SL-WC_vol_rDA-Var',**plt_Fig_dict)


fig, ax1 = plt.subplots()
ax = sns.scatterplot(y='DEFlutoB',x='Variant',hue='Numsch',data=t2,ax=ax1)
ax = sns.lineplot(y='DEFlutoB',x='Variant',hue='Numsch',data=t2,ax=ax1, legend=False)
ax1.set_title('%sDeviation of Youngs Modulus (fixed range) based on water storaged\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SL-DEFlutoB-Var',**plt_Fig_dict)

fig, ax1 = plt.subplots()
ax = sns.scatterplot(y='DEFlutoG',x='Variant',hue='Numsch',data=t2,ax=ax1)
ax = sns.lineplot(y='DEFlutoG',x='Variant',hue='Numsch',data=t2,ax=ax1, legend=False)
ax1.set_title('%sDeviation of Youngs Modulus (fixed range) based on dry mass\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SL-DEFlutoG-Var',**plt_Fig_dict)




fig, ax1 = plt.subplots()
ax = sns.scatterplot(x='WC_vol',y='DEFlutoB',hue='Numsch',data=t2,ax=ax1)
ax1.set_title('%sYoungs Modulus (fixed range) deviation versus water content'%name_Head)
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SL-WC_vol-DEFlutoB',**plt_Fig_dict)

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(x='WC_vol',y='DEFlutoG',hue='Numsch',data=t2,ax=ax1)
ax1.set_title('%sYoungs Modulus (fixed range) deviation versus water content'%name_Head)
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SL-WC_vol-DEFlutoG',**plt_Fig_dict)


#%%% Plots neu
# tmp=pd.concat([dft['DMWtoA'],cEEm_eva['DEFlutoB']], axis=1)
# df=pd.melt(tmp.reset_index(), id_vars=['Variant'], value_vars=['DMWtoA','DEFlutoB'])
# #df.sort_values(['Variant','variable'],inplace=True)
# fig, ax1 = plt.subplots()
# ax = sns.boxplot(x="Variant", y="value", hue="variable", data=df, ax=ax1, 
#                   showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="Variant", y="value", hue="variable",
#                     data=df, ax=ax1, dodge=True, edgecolor="black",
#                     linewidth=.5, alpha=.5, size=2)
# handles, _ = ax1.get_legend_handles_labels()
# ax1.legend(handles[0:2], ['DMWtoA','DEFlutoB'], loc="best")
# Evac.tick_legend_renamer(ax=ax1,
#                          renamer=VIPar_plt_renamer,
#                          title=None)
# ax1.set_xlabel('Variant / -')
# ax1.set_ylabel('Deviation / -')
# Evac.plt_handle_suffix(fig,path=out_full+'-Box-comb_DMWtoA-DEFlutoB',**plt_Fig_dict)

# tmp=pd.concat([dft[['DMWtoA','DMWtoG']],cEEm_eva[['DEFlutoB','DEFlutoG']]], axis=1)
# df=pd.melt(tmp.reset_index(), id_vars=['Variant'], value_vars=['DMWtoA','DMWtoG','DEFlutoB','DEFlutoG'])
# #df.sort_values(['Variant','variable'],inplace=True)
# fig, ax1 = plt.subplots()
# ax = sns.boxplot(x="Variant", y="value", hue="variable", data=df, ax=ax1, 
#                   showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="Variant", y="value", hue="variable",
#                     data=df, ax=ax1, dodge=True, edgecolor="black",
#                     linewidth=.5, alpha=.5, size=2)
# handles, _ = ax1.get_legend_handles_labels()
# ax1.legend(handles[0:4], ['DMWtoA','DMWtoG','DEFlutoB','DEFlutoG'], loc="best")
# Evac.tick_legend_renamer(ax=ax1,
#                          renamer=VIPar_plt_renamer,
#                          title=None)
# ax1.set_xlabel('Variant / -')
# ax1.set_ylabel('Deviation / -')
# Evac.plt_handle_suffix(fig,path=out_full+'-Box-comb_DMW-DEFlu',**plt_Fig_dict)


fig, ax1 = plt.subplots()
ax1.set_title('Exponential regression of water content vs. relative storage humidity')
ax1.set_xlabel(r'$\Phi_{env}$ / -')
ax1.set_ylabel(r'$\Phi_{vol}$ / -')
sns.scatterplot(data=dft,
                y='WC_vol',x='RHstore',
                style='Series',hue='Variant',ax=ax1)
tmp=ax1.legend(ncol=5,bbox_to_anchor=(0.01, 0.77),
           loc='lower left')
tmp._legend_box.align = "left"
ax2=ax1.twinx()
tmp=Regfitret(pdo=dft.query("Variant<='G'"), x='RHstore', y='WC_vol',
          name='exponential_x0', guess = dict(a=0, b=0.1, c=10),
          xl=r'\Phi_{env}',yl=r'\Phi_{vol,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg_fo(pdo=dft.query("Variant<='G'"), x='RHstore',
            fit=tmp, ax=ax2, label_f=True,
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=dft.query("Variant>='G'"), x='RHstore', y='WC_vol',
          name='exponential_x0', guess = dict(a=0, b=0.1, c=2),
          xl=r'\Phi_{env}',yl=r'\Phi_{vol,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg_fo(pdo=dft.query("Variant>='G'"), x='RHstore',
              fit=tmp, ax=ax2, label_f=True,
              lkws=dict(color=sns.color_palette("tab10")[0]))
ax2.set_ylim(ax1.get_ylim())
ax2.tick_params(right = False , labelright = False)
tmp=ax2.legend(title='         Exponential fit', bbox_to_anchor=(0.01, 0.52),
           loc='lower left')
tmp._legend_box.align = "left"
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

#%%% SupMat-Plots
figsize_sup=(16.0/2.54, 22.0/2.54)

fig, ax = plt.subplots(ncols=1,nrows=3,
                        figsize=figsize_sup, constrained_layout=True)
ax[0] = sns.boxplot(data=dft_comb.query("Variant!='A'").reset_index(), 
                 x='Variant', y='WC_vol_rDA', hue='Series', ax=ax[0],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"10","alpha":0.75})
ax[0].set_title('Relative deviation of volumetric water content based on fresh')
ax[0].set_xlabel('Variant / -')
ax[0].set_ylabel(r'$D_{\Phi_{vol},fresh}$ / -')
ax[1] = sns.boxplot(data=dft_comb.query("Variant!='A'").reset_index(), 
                 x='Variant', y='DEFlutoB', hue='Series', ax=ax[1],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"10","alpha":0.75})
ax[1].set_title('Relative deviation of elastic modulus based on water storaged')
ax[1].set_xlabel('Variant / -')
ax[1].set_ylabel(r'$D_{E,saturated}$ / -')

tmp2=pd.concat([cs[['Series','WC_vol_rDA']],cEEm_eva['DEFlutoB']],axis=1)
tmp21=tmp2.query("Series=='1'")
tmp22=tmp2.query("Series=='2'")
sns.scatterplot(data=tmp21.query("Variant<='G'"),
                x='WC_vol_rDA', y='DEFlutoB',
                ax=ax[2], label='Series 1 - Desorption', 
                **dict(color=sns.color_palette("tab10")[0],
                       marker="o"))
sns.scatterplot(data=tmp21.query("Variant>='G'"),
                x='WC_vol_rDA', y='DEFlutoB',
                ax=ax[2], label='Series 1 - Adsorption',
                **dict(color=sns.color_palette("tab10")[0],
                       marker="P"))
sns.scatterplot(data=tmp22.query("Variant<='G'"),
                x='WC_vol_rDA', y='DEFlutoB',
                ax=ax[2], label='Series 2 - Desorption', 
                **dict(color=sns.color_palette("tab10")[1],
                       marker="o"))
sns.scatterplot(data=tmp22.query("Variant>='G'"),
                x='WC_vol_rDA', y='DEFlutoB',
                ax=ax[2], label='Series 2 - Adsorption',
                **dict(color=sns.color_palette("tab10")[1],
                       marker="P"))
tmp=Regfitret(pdo=tmp21.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
              name='exponential_x0', guess = dict(a=0, b=0.01, c=-1),
              xl=r'D_{\Phi_{vol},fresh}',yl=r'D_{E,1,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg_fo(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol_rDA', fit=tmp, ax=ax[2], label_f=True,
              lkws=dict(color=sns.color_palette("tab10")[0],
                        linestyle='--'))
tmp=Regfitret(pdo=tmp21.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
              name='exponential', guess = dict(a=0.01, b=0.01, c=1),
              xl=r'D_{\Phi_{vol},fresh}',yl=r'D_{E,1,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg_fo(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol_rDA', fit=tmp, ax=ax[2], label_f=True,
              lkws=dict(color=sns.color_palette("tab10")[0],
                        linestyle=':'))
tmp=Regfitret(pdo=tmp22.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
              name='exponential_x0', guess = dict(a=0, b=0.01, c=-1),
              xl=r'D_{\Phi_{vol},fresh}',yl=r'D_{E,2,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg_fo(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol_rDA', fit=tmp, ax=ax[2], label_f=True,
              lkws=dict(color=sns.color_palette("tab10")[1],
                        linestyle='--'))
tmp=Regfitret(pdo=tmp22.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
              name='exponential', guess = dict(a=0.01, b=0.01, c=1),
              xl=r'D_{\Phi_{vol},fresh}',yl=r'D_{E,2,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg_fo(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol_rDA', fit=tmp, ax=ax[2], label_f=True,
              lkws=dict(color=sns.color_palette("tab10")[1],
                        linestyle=':'))
tmp1,tmp2=ax[2].get_legend_handles_labels()
tmp3=[4,5,6,7,0,1,2,3]
ax[2].legend([tmp1[i] for i in tmp3], [tmp2[i] for i in tmp3],
             ncol=2, fontsize=6)
ax[2].set_xlabel(r'$D_{\Phi_{vol},fresh}$ / -')
ax[2].set_ylabel(r'$D_{E,saturated}$ / -')
ax[2].set_title('Relation between the relative deviations of elastic modulus and water content')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SM-Series_dep',**plt_Fig_dict)

fig, ax = plt.subplots(ncols=1,nrows=3,
                        figsize=figsize_sup, constrained_layout=True)
ax[0] = sns.boxplot(data=dft_comb.query("Variant!='G'").reset_index(), 
                 x='Variant', y='WC_vol', hue='Donor', ax=ax[0],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"4","alpha":0.75})
ax[0].set_title('Volumetric water content')
ax[0].set_xlabel('Variant / -')
ax[0].set_ylabel(r'$\Phi_{vol}$ / -')
ax[0].set_yscale("log")
ax[0].set_yticks([0.1,1.0])
ax[0].grid(True, which='both', axis='y')
ax[1] = sns.boxplot(data=dft_comb.query("Variant!='A'").reset_index(), 
                 x='Variant', y='WC_vol_rDA', hue='Donor', ax=ax[1],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"4","alpha":0.75})
ax[1].set_title('Relative deviation of volumetric water content based on fresh')
ax[1].set_xlabel('Variant / -')
ax[1].set_ylabel(r'$D_{\Phi_{vol},fresh}$ / -')
ax[2] = sns.boxplot(data=dft_comb.query("Variant!='A'").reset_index(), 
                 x='Variant', y='DEFlutoB', hue='Donor', ax=ax[2],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"4","alpha":0.75})
ax[2].set_title('Relative deviation of elastic modulus based on water storaged')
ax[2].set_xlabel('Variant / -')
ax[2].set_ylabel(r'$D_{E,saturated}$ / -')
for i in range(ax.size):
    ax[i].legend(loc="best",ncol=2)
    Evac.tick_legend_renamer(ax=ax[i],
                             renamer=doda.Naming.to_dict(),
                             title='Donor')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SM-Donor_dep',**plt_Fig_dict)

fig, ax = plt.subplots(ncols=1,nrows=3,
                        figsize=figsize_sup, constrained_layout=True)
ax[0] = sns.boxplot(data=dft_comb.query("Variant!='G'").reset_index(), 
                 x='Variant', y='WC_vol', hue='Side_LR', ax=ax[0],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"10","alpha":0.75})
ax[0].set_title('Volumetric water content')
ax[0].set_xlabel('Variant / -')
ax[0].set_ylabel(r'$\Phi_{vol}$ / -')
ax[0].set_yscale("log")
ax[0].set_yticks([0.1,1.0])
ax[0].grid(True, which='both', axis='y')
ax[1] = sns.boxplot(data=dft_comb.query("Variant!='A'").reset_index(), 
                 x='Variant', y='WC_vol_rDA', hue='Side_LR', ax=ax[1],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"10","alpha":0.75})
ax[1].set_title('Relative deviation of volumetric water content based on fresh')
ax[1].set_xlabel('Variant / -')
ax[1].set_ylabel(r'$D_{\Phi_{vol},fresh}$ / -')
ax[2] = sns.boxplot(data=dft_comb.query("Variant!='A'").reset_index(), 
                 x='Variant', y='DEFlutoB', hue='Side_LR', ax=ax[2],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"10","alpha":0.75})
ax[2].set_title('Relative deviation of elastic modulus based on water storaged')
ax[2].set_xlabel('Variant / -')
ax[2].set_ylabel(r'$D_{E,saturated}$ / -')
for i in range(ax.size):
    Evac.tick_legend_renamer(ax=ax[i],
                             renamer={'L':'Left','R':'Right'},
                             title='Body side')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SM-Side_LR_dep',**plt_Fig_dict)

fig, ax = plt.subplots(ncols=1,nrows=3,
                        figsize=figsize_sup, constrained_layout=True)
ax[0] = sns.boxplot(data=dft_comb.query("Variant!='G'").reset_index(), 
                 x='Variant', y='WC_vol', hue='Side_pd', ax=ax[0],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"10","alpha":0.75})
ax[0].set_title('Volumetric water content')
ax[0].set_xlabel('Variant / -')
ax[0].set_ylabel(r'$\Phi_{vol}$ / -')
ax[0].set_yscale("log")
ax[0].set_yticks([0.1,1.0])
ax[0].grid(True, which='both', axis='y')
ax[1] = sns.boxplot(data=dft_comb.query("Variant!='A'").reset_index(), 
                 x='Variant', y='WC_vol_rDA', hue='Side_pd', ax=ax[1],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"10","alpha":0.75})
ax[1].set_title('Relative deviation of volumetric water content based on fresh')
ax[1].set_xlabel('Variant / -')
ax[1].set_ylabel(r'$D_{\Phi_{vol},fresh}$ / -')
ax[2] = sns.boxplot(data=dft_comb.query("Variant!='A'").reset_index(), 
                 x='Variant', y='DEFlutoB', hue='Side_pd', ax=ax[2],
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                            "markeredgecolor":"black", "markersize":"10","alpha":0.75})
ax[2].set_title('Relative deviation of elastic modulus based on water storaged')
ax[2].set_xlabel('Variant / -')
ax[2].set_ylabel(r'$D_{E,saturated}$ / -')
for i in range(ax.size):
    Evac.tick_legend_renamer(ax=ax[i],
                             renamer={'proximal':'proximal','distal':'distal'},
                             title='Harvesting location')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-SM-Side_pd_dep',**plt_Fig_dict)

#%%% Influence
tmp = dft[['RHenv','Humidity_store','DMWtoA','DMWtoG',
           'WC_vol','WC_vol_toA','WC_vol_rDA']]
fig, ax = plt.subplots(nrows=3, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1],
                                    'height_ratios': [1,1,1]},
                       figsize = (6.3,3*3.54))
fig.suptitle('%s\nInfluence water manipulation'%name_Head,
             fontweight="bold")
ax[0,0].set_title("Complete")
tmp_corr = tmp.corr(method=mcorr)
g=sns.heatmap(tmp_corr.round(2), 
              center=0, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0,0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[0,0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[0,1].tick_params(axis='y', labelsize=8)
ax[1,0].set_title("Desorption")
tmp_corr =  tmp.loc(axis=0)[:,['B','C','D','E','F','G']].corr(method=mcorr)
g=sns.heatmap(tmp_corr.round(2), 
              center=0, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[1,0],cbar_ax=ax[1,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[1,0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[1,0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[1,1].tick_params(axis='y', labelsize=8)
ax[2,0].set_title("Adsorption")
tmp_corr =  tmp.loc(axis=0)[:,['G','H','I','J','K','L']].corr(method=mcorr)
g=sns.heatmap(tmp_corr.round(2), 
              center=0, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[2,0],cbar_ax=ax[2,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[2,0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[2,0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[2,1].tick_params(axis='y', labelsize=8)
Evac.plt_handle_suffix(fig,path=out_full+'-Corr-WMan',**plt_Fig_dict)


tmp = pd.concat([cs.loc(axis=1)[['DMWtoA','DMWtoG','WC_vol','WC_vol_rDA']],
                 cEEm_eva.loc(axis=1)[['lu_F_mean','lu_F_ratio',
                                       'DEFlutoB','DEFlutoG']],
                 cH_eva.loc(axis=1)[['Hyst_APn','DHAPntoB','DHAPntoG']]],
                axis=1)
tmp_corr =  tmp.corr(method=mcorr)
fig, ax = plt.subplots(nrows=1, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1]},
                       constrained_layout=True)
fig.suptitle('%s\nInfluence of material data - Complete'%name_Head,
             fontweight="bold")
g=sns.heatmap(tmp_corr.round(2), 
              center=0, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[0],cbar_ax=ax[1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[1].tick_params(axis='y', labelsize=8)
ax[0].set_xlabel('Influences')
ax[0].set_ylabel('Influences')
plt.show()

tmp_corr =  tmp.loc(axis=0)[:,['B','C','D','E','F','G']].corr(method=mcorr)
fig, ax = plt.subplots(nrows=1, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1]},
                       constrained_layout=True)
fig.suptitle('%s\nInfluence of material data - Desorption'%name_Head,
             fontweight="bold")
g=sns.heatmap(tmp_corr.round(2), 
              center=0, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[0],cbar_ax=ax[1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[1].tick_params(axis='y', labelsize=8)
ax[0].set_xlabel('Influences')
ax[0].set_ylabel('Influences')
plt.show()

tmp_corr =  tmp.loc(axis=0)[:,['G','H','I','J','K','L']].corr(method=mcorr)
fig, ax = plt.subplots(nrows=1, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1]},
                       constrained_layout=True)
fig.suptitle('%s\nInfluence of material data - Absorption'%name_Head,
             fontweight="bold")
g=sns.heatmap(tmp_corr.round(2), 
              center=0, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[0],cbar_ax=ax[1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[1].tick_params(axis='y', labelsize=8)
ax[0].set_xlabel('Influences')
ax[0].set_ylabel('Influences')
plt.show()


tmp=dft_comb.loc(axis=0)[:,'A'][['thickness_1','thickness_2','thickness_3']].mean(axis=1).droplevel(1)
tmp.name='t_mean'
tmp2=dft_comb.loc(axis=0)[:,'A']['VolTot'].droplevel(1)
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['A','G']][['Density_app']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,'A'][['WC_vol']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['B','C','L']][['WC_vol_rDA']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,'B'][['lu_F_mean']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['C','G','L']][['DEFlutoB']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,'B'][['Hyst_APn']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['C','G','L']][['DHAPntoB']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp_ren={'t_mean':r'$\overline{t}$','VolTot':r'$V_{total}$',
         ('Density_app','A'): r'$\rho_{app,fresh}$',
         ('Density_app','G'): r'$\rho_{app,dry}$',
         ('WC_vol','A'): r'$\Phi_{vol,fresh}$',
         ('WC_vol_rDA','B'):r'$D_{\Phi_{vol},B-fresh}$',
         ('WC_vol_rDA','C'):r'$D_{\Phi_{vol},C-fresh}$',
         ('WC_vol_rDA','L'):r'$D_{\Phi_{vol},L-fresh}$',
         ('lu_F_mean','B'):r'$E_{sat}$',
         ('DEFlutoB','C'):r'$D_{E,C-sat}$',
         ('DEFlutoB','G'):r'$D_{E,G-sat}$',
         ('DEFlutoB','L'):r'$D_{E,L-sat}$',
         ('Hyst_APn','B'):r'$H_{n,sat}$',
         ('DHAPntoB','C'):r'$D_{H_{n},C-sat}$',
         ('DHAPntoB','G'):r'$D_{H_{n},G-sat}$',
         ('DHAPntoB','L'):r'$D_{H_{n},L-sat}$'}
fig, ax = plt.subplots(nrows=2, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1],
                                    'height_ratios': [1,1.4]},
                       figsize = (6.3,3.54))
tmp3=tmp.corr(method=mcorr).loc[['t_mean','VolTot',('Density_app','A'),('Density_app','G'),
                                 ('WC_vol','A')],
                                [('WC_vol','A'),('WC_vol_rDA','B'),('WC_vol_rDA','C'),
                                 ('WC_vol_rDA','L')]]
g=sns.heatmap(tmp3.rename(index=tmp_ren,columns=tmp_ren).round(2), 
              center=0, vmin=-0.7,vmax=0.7, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[0,0],cbar_ax=ax[0,1])
ax[0,0].set_title("Influences on water manipulation")
ax[0,0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[0,0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[0,1].tick_params(axis='y', labelsize=8)
ax[0,0].set_xlabel('Selected parameters')
ax[0,0].set_ylabel('Selected influences')
tmp3=tmp.corr(method=mcorr).loc[['t_mean','VolTot',
                                 ('Density_app','A'),('Density_app','G'),
                                 ('WC_vol','A'),('WC_vol_rDA','B'),('WC_vol_rDA','L')],
                                [('lu_F_mean','B'),('DEFlutoB','C'),('DEFlutoB','G'),('DEFlutoB','L'),
                                 ('Hyst_APn','B'),('DHAPntoB','C'),('DHAPntoB','G'),('DHAPntoB','L')]]
g=sns.heatmap(tmp3.rename(index=tmp_ren,columns=tmp_ren).round(2), 
              center=0, vmin=-0.7,vmax=0.7, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[1,0],cbar_ax=ax[1,1])
ax[1,0].set_title("Influences on mechanical parameters")
ax[1,0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[1,0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[1,1].tick_params(axis='y', labelsize=8)
ax[1,0].set_xlabel('Selected parameters')
ax[1,0].set_ylabel('Selected influences')
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=out_full+'-Corr-Sel',**plt_Fig_dict)

# Correlation to Donordata
tmp=dft_comb.loc(axis=0)[:,'A'][['thickness_1','thickness_2','thickness_3']].mean(axis=1).droplevel(1)
tmp.name='t_mean'
tmp2=dft_comb.loc(axis=0)[:,['A','G']][['Mass','VolTot']]
tmp2=(tmp2['Mass']/tmp2['VolTot']).unstack()
tmp2.columns=tmp2.columns.to_series().replace({'A':'Density_fresh','G':'Density_dry'})
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,'A'][['WC_vol']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['B','C','L']][['WC_vol_rDA']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['B']][['lu_F_mean']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['C','G','L']][['DEFlutoB']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['B']][['Hyst_APn']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp2=dft_comb.loc(axis=0)[:,['C','G','L']][['DHAPntoB']].unstack()
tmp=pd.concat([tmp,tmp2],axis=1)
tmp3=dft_doda_comb.loc[idx[:,'G'],['Postmortem_LT','Storage_Time','Age','BMI','Sex']].droplevel(1).copy(deep=True)
tmp3['Sex'].replace({'w':1,'m':-1}, inplace=True)
tmp_corr=pd.concat([tmp,tmp3],axis=1).corr(method=mcorr)
tmp_ren={'t_mean':r'$\overline{h}$','VolTot':r'$V_{total}$',
         'Density_fresh': r'$\rho_{app,fresh}$','Density_dry': r'$\rho_{app,dry}$',
         ('WC_vol','A'): r'$\Phi_{vol,fresh}$',
         ('WC_vol_rDA','B'):r'$D_{\Phi_{vol},B-fresh}$',
         ('WC_vol_rDA','C'):r'$D_{\Phi_{vol},C-fresh}$',
         ('WC_vol_rDA','L'):r'$D_{\Phi_{vol},L-fresh}$',
         ('lu_F_mean','B'):r'$E_{sat}$',
         ('DEFlutoB','C'):r'$D_{E,C-sat}$',
         ('DEFlutoB','G'):r'$D_{E,G-sat}$',
         ('DEFlutoB','L'):r'$D_{E,L-sat}$',
         ('Hyst_APn','B'):r'$H_{n,sat}$',
         ('DHAPntoB','C'):r'$D_{H_{n},C-sat}$',
         ('DHAPntoB','G'):r'$D_{H_{n},G-sat}$',
         ('DHAPntoB','L'):r'$D_{H_{n},L-sat}$',
         'Postmortem_LT':r'$t_{PML}$',
         'Storage_Time':r'$t_{store}$',
         'Age':r'$Age$','BMI':r'$BMI$','Sex':r'$Sex$'}
fig, ax = plt.subplots(nrows=1, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1]},
                       constrained_layout=True)
#ax[0].set_title('%s\nInfluence of donor data'%name_Head)
g=sns.heatmap(tmp_corr.loc[tmp.columns,tmp3.columns].rename(index=tmp_ren,columns=tmp_ren).round(2), 
              center=0, annot=True, annot_kws={"size":8, 'rotation':0},
              xticklabels=1, ax=ax[0],cbar_ax=ax[1])
ax[0].tick_params(axis='x', labelrotation=0, labelsize=8)
ax[0].tick_params(axis='y', labelrotation=0, labelsize=8)
ax[1].tick_params(axis='y', labelsize=8)
ax[0].set_xlabel('Selected donor related influences')
ax[0].set_ylabel('Selected parameters')
fig.suptitle(None)
Evac.plt_handle_suffix(fig,path=out_full+'-Corr-Doda',**plt_Fig_dict)

#%%% Regression
#%%%% Berechnung
txt="\n "+"="*100
txt+=("\n Linear regression:")
Lin_reg_df =pd.DataFrame([],dtype='O')

def MG_linRegStats(Y,X,Y_txt,X_txt):
    des_txt = ("\n  - Linear relation between %s and %s:\n   "%(Y_txt,X_txt))
    tmp  = Evac.YM_sigeps_lin(Y, X)
    smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
    out=pd.Series([*tmp,smst,des_txt],
                  index=['s','c','Rquad','fit','smstat','Description'])
    return out

Lin_reg_df['Com-DMWtoG-DEFtoG'] = MG_linRegStats(t_ona['DEFtoG'], t_ona['DMWtoG'],
                                                "Complete Deviation YM (fixed) to dry", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DEFtoG'] = MG_linRegStats(t_ona_des['DEFtoG'], t_ona_des['DMWtoG'],
                                                "Desorption Deviation YM (fixed) to dry", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoG-DEFtoG'] = MG_linRegStats(t_ona_abs['DEFtoG'], t_ona_abs['DMWtoG'],
                                                "Absorption Deviation YM (fixed) to dry", 
                                                "Absorption water content")

Lin_reg_df['Com-DMWtoG-DEFtoB'] = MG_linRegStats(t_ona['DEFtoB'], t_ona['DMWtoG'],
                                                "Complete Deviation YM (fixed) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DEFtoB'] = MG_linRegStats(t_ona_des['DEFtoB'], t_ona_des['DMWtoG'],
                                                "Desorption Deviation YM (fixed) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoG-DEFtoB'] = MG_linRegStats(t_ona_abs['DEFtoB'], t_ona_abs['DMWtoG'],
                                                "Absorption Deviation YM (fixed) to saturated", 
                                                "Absorption water content")

Lin_reg_df['Com-DMWtoA-DEFtoB'] = MG_linRegStats(t_ona['DEFtoB'], t_ona['DMWtoA'],
                                                "Complete Deviation YM (fixed) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoA-DEFtoB'] = MG_linRegStats(t_ona_des['DEFtoB'], t_ona_des['DMWtoA'],
                                                "Desorption Deviation YM (fixed) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoA-DEFtoB'] = MG_linRegStats(t_ona_abs['DEFtoB'], t_ona_abs['DMWtoA'],
                                                "Absorption Deviation YM (fixed) to saturated", 
                                                "Absorption water content")


Lin_reg_df['Com-DMWtoG-DERtoG'] = MG_linRegStats(t_ona['DERtoG'], t_ona['DMWtoG'],
                                                "Complete Deviation YM (refined) to dry", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DERtoG'] = MG_linRegStats(t_ona_des['DERtoG'], t_ona_des['DMWtoG'],
                                                "Desorption Deviation YM (refined) to dry", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoG-DERtoG'] = MG_linRegStats(t_ona_abs['DERtoG'], t_ona_abs['DMWtoG'],
                                                "Absorption Deviation YM (refined) to dry", 
                                                "Absorption water content")

Lin_reg_df['Com-DMWtoG-DERtoB'] = MG_linRegStats(t_ona['DERtoB'], t_ona['DMWtoG'],
                                                "Complete Deviation YM (refined) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DERtoB'] = MG_linRegStats(t_ona_des['DERtoB'], t_ona_des['DMWtoG'],
                                                "Desorption Deviation YM (refined) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoG-DERtoB'] = MG_linRegStats(t_ona_abs['DERtoB'], t_ona_abs['DMWtoG'],
                                                "Absorption Deviation YM (refined) to saturated", 
                                                "Absorption water content")

Lin_reg_df['Com-DMWtoA-DERtoB'] = MG_linRegStats(t_ona['DERtoB'], t_ona['DMWtoA'],
                                                "Complete Deviation YM (refined) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoA-DERtoB'] = MG_linRegStats(t_ona_des['DERtoB'], t_ona_des['DMWtoA'],
                                                "Desorption Deviation YM (refined) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoA-DERtoB'] = MG_linRegStats(t_ona_abs['DERtoB'], t_ona_abs['DMWtoA'],
                                                "Absorption Deviation YM (refined) to saturated", 
                                                "Absorption water content")

# load and unload:
Lin_reg_df['Com-DMWtoG-DEFlutoG'] = MG_linRegStats(t2['DEFlutoG'], 
                                                   t2['DMWtoG'],
                                                "Complete Deviation YM (lu-fixed) to dry", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DEFlutoG'] = MG_linRegStats(t2[t2.Variant<='G']['DEFlutoG'], 
                                                   t2[t2.Variant<='G']['DMWtoG'],
                                                "Desorption Deviation YM (lu-fixed) to dry", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoG-DEFlutoG'] = MG_linRegStats(t2[t2.Variant>='G']['DEFlutoG'], 
                                                   t2[t2.Variant>='G']['DMWtoG'],
                                                "Absorption Deviation YM (lu-fixed) to dry", 
                                                "Absorption water content")

Lin_reg_df['Com-DMWtoG-DEFlutoB'] = MG_linRegStats(t2['DEFlutoB'], 
                                                   t2['DMWtoG'],
                                                "Complete Deviation YM (lu-fixed) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DEFlutoB'] = MG_linRegStats(t2[t2.Variant<='G']['DEFlutoB'], 
                                                   t2[t2.Variant<='G']['DMWtoG'],
                                                "Desorption Deviation YM (lu-fixed) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoG-DEFlutoB'] = MG_linRegStats(t2[t2.Variant>='G']['DEFlutoB'], 
                                                   t2[t2.Variant>='G']['DMWtoG'],
                                                "Absorption Deviation YM (lu-fixed) to saturated", 
                                                "Absorption water content")

Lin_reg_df['Com-DMWtoA-DEFlutoB'] = MG_linRegStats(t2['DEFlutoB'], 
                                                   t2['DMWtoA'],
                                                "Complete Deviation YM (lu-fixed) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoA-DEFlutoB'] = MG_linRegStats(t2[t2.Variant<='G']['DEFlutoB'], 
                                                   t2[t2.Variant<='G']['DMWtoA'],
                                                "Desorption Deviation YM (lu-fixed) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Abs-DMWtoA-DEFlutoB'] = MG_linRegStats(t2[t2.Variant>='G']['DEFlutoB'], 
                                                   t2[t2.Variant>='G']['DMWtoA'],
                                                "Absorption Deviation YM (lu-fixed) to saturated", 
                                                "Absorption water content")

for i in Lin_reg_df.columns:
    txt += Lin_reg_df.loc['Description',i]
    txt += Evac.str_indent(Evac.fit_report_adder(*Lin_reg_df.loc[['fit','Rquad'],i]))
    txt += Evac.str_indent(Lin_reg_df.loc['smstat',i])

Evac.MG_strlog(txt,log_mg,1,printopt=False)





#%%%% Plots
formula_in_dia = True

def plt_RegA(pdo, x, y, xl,yl, title, path, plt_Fig_dict):
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax1.set_title(title)
    sns.regplot(x=x, y=y, data=pdo,                   ax=ax1, label='Complete')
    sns.regplot(x=x, y=y, data=pdo[pdo.Variant<='G'], ax=ax1, label='Desorption')
    sns.regplot(x=x, y=y, data=pdo[pdo.Variant>='G'], ax=ax1, label='Absorption')
    ax1.set_xlabel(xl+' / -')
    ax1.set_ylabel(yl+' / -')
    ax1.legend()
    fig.suptitle('')
    Evac.plt_handle_suffix(fig,path=path,**plt_Fig_dict)
    
def plt_RegD(pdo, x, y, xl,yl,
             formula_in_dia, Lin_reg_df,
             title, path, plt_Fig_dict):
    
    Lr_xy='-'+x+'-'+y
    # t_form='{0:5.3e},{1:5.3e},{2:.3f}'
    t_form='{0:.3f},{1:.5f},{2:.3f}'
    if formula_in_dia:
        txtc = t_form.format(*Lin_reg_df['Com'+Lr_xy].iloc[:3]).split(',')
        txtc = r'{y} = {0} {x} + {1} ($R²$ = {2})'.format(*txtc,**{'y':yl,'x':xl},)
        txtd = t_form.format(*Lin_reg_df['Des'+Lr_xy].iloc[:3]).split(',')
        txtd = r'{y} = {0} {x} + {1} ($R²$ = {2})'.format(*txtd,**{'y':yl,'x':xl},)
        txta = t_form.format(*Lin_reg_df['Abs'+Lr_xy].iloc[:3]).split(',')
        txta = r'{y} = {0} {x} + {1} ($R²$ = {2})'.format(*txta,**{'y':yl,'x':xl},)
    else:
        txtc = r'$D_{E}$'
    fig, axa = plt.subplots(nrows=3, ncols=1, 
                            sharex=True, sharey=False, figsize = (6.3,3*3.54))
    fig.suptitle(title)
    axa[0].grid(True)
    axa[0].set_title('Desorption and Absorption (all Variants)')
    sns.regplot(x=x, y=y, data=pdo,ax=axa[0],label=txtc)
    axa[0].set_ylabel(yl+' / -')
    axa[0].set_xlabel(None)
    axa[0].legend()
    axa[1].grid(True)
    axa[1].set_title('Desorption')
    sns.regplot(x=x, y=y, data=pdo[pdo.Variant<='G'],ax=axa[1],label=txtd)
    axa[1].set_ylabel(yl+' / -')
    axa[1].set_xlabel(None)
    axa[1].legend()
    axa[2].grid(True)
    axa[2].set_title('Absorption')
    sns.regplot(x=x, y=y, data=pdo[pdo.Variant>='G'],ax=axa[2],label=txta)
    axa[2].set_xlabel(xl+' / -')
    axa[2].set_ylabel(yl+' / -')
    axa[2].legend()
    Evac.plt_handle_suffix(fig,path=path,**plt_Fig_dict)
    
    

# if formula_in_dia:
#     txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoG-DEFtoG'].iloc[:3]).split(',')
#     txtc = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtc,)
#     txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoG-DEFtoG'].iloc[:3]).split(',')
#     txtd = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtd,)
#     txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Abs-DMWtoG-DEFtoG'].iloc[:3]).split(',')
#     txta = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txta,)
# else:
#     txtc = r'$D_{E}$'
    
# fig, axa = plt.subplots(nrows=3, ncols=1, 
#                               sharex=True, sharey=False, figsize = (6.3,3*3.54))
# fig.suptitle('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
# axa[0].grid(True)
# axa[0].set_title('Desorption and Absorption (all Variants)')
# ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t,ax=axa[0],label=txtc)
# axa[0].set_ylabel(r'$D_{E,dry}$ / -')
# axa[0].set_xlabel(None)
# axa[0].legend()
# axa[1].grid(True)
# axa[1].set_title('Desorption')
# ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
# axa[1].set_ylabel(r'$D_{E,dry}$ / -')
# axa[1].set_xlabel(None)
# axa[1].legend()
# axa[2].grid(True)
# axa[2].set_title('Absorption')
# ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t[t.Variant>='G'],ax=axa[2],label=txta)
# axa[2].set_xlabel(r'$D_{mass,Water}$ / -')
# axa[2].set_ylabel(r'$D_{E,dry}$ / -')
# axa[2].legend()
# Evac.plt_handle_suffix(fig,path=out_full+'-Regd-DMWtoG-DEFtoG',**plt_Fig_dict)

# fig, ax1 = plt.subplots()
# ax1.grid(True)
# ax1.set_title('%sRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
# ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t,ax=ax1, label='Complete')
# ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
# ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t[t.Variant>='G'],ax=ax1, label='Absorption')
# ax1.set_xlabel(r'$D_{mass,Water}$ / -')
# ax1.set_ylabel(r'$D_{E,dry}$ / -')
# ax1.legend()
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=out_full+'-Rega-DMWtoG-DEFtoG',**plt_Fig_dict)

# #------------
# # if formula_in_dia:
# #     txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoG-DERtoG'].iloc[:3]).split(',')
# #     txtc = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtc,)
# #     txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoG-DERtoG'].iloc[:3]).split(',')
# #     txtd = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtd,)
# #     txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Abs-DMWtoG-DERtoG'].iloc[:3]).split(',')
# #     txta = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txta,)
# # else:
# #     txtc = r'$D_{E}$'
    
# # fig, axa = plt.subplots(nrows=3, ncols=1, 
# #                               sharex=True, sharey=False, figsize = (6.3,3*3.54))
# # fig.suptitle('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
# # axa[0].grid(True)
# # axa[0].set_title('Desorption and Absorption (all Variants)')
# # ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t,ax=axa[0],label=txtc)
# # axa[0].set_ylabel(r'$D_{E,dry}$ / -')
# # axa[0].set_xlabel(None)
# # axa[0].legend()
# # axa[1].grid(True)
# # axa[1].set_title('Desorption')
# # ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
# # axa[1].set_ylabel(r'$D_{E,dry}$ / -')
# # axa[1].set_xlabel(None)
# # axa[1].legend()
# # axa[2].grid(True)
# # axa[2].set_title('Absorption')
# # ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t[t.Variant>='G'],ax=axa[2],label=txta)
# # axa[2].set_xlabel(r'$D_{mass,Water}$ / -')
# # axa[2].set_ylabel(r'$D_{E,dry}$ / -')
# # axa[2].legend()
# # Evac.plt_handle_suffix(fig,path=out_full+'-Regd-DMWtoG-DERtoG',**plt_Fig_dict)

# # fig, ax1 = plt.subplots()
# # ax1.grid(True)
# # ax1.set_title('%sRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
# # ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t,ax=ax1, label='Complete')
# # ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
# # ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t[t.Variant>='G'],ax=ax1, label='Absorption')
# # ax1.set_xlabel(r'$D_{mass,Water}$ / -')
# # ax1.set_ylabel(r'$D_{E,dry}$ / -')
# # ax1.legend()
# # fig.suptitle('')
# # Evac.plt_handle_suffix(fig,path=out_full+'-Rega-DMWtoG-DERtoG',**plt_Fig_dict)

# #==================
# if formula_in_dia:
#     txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoG-DEFtoB'].iloc[:3]).split(',')
#     txtc = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtc,)
#     txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoG-DEFtoB'].iloc[:3]).split(',')
#     txtd = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtd,)
#     txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Abs-DMWtoG-DEFtoB'].iloc[:3]).split(',')
#     txta = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txta,)
# else:
#     txtc = r'$D_{E}$'
    
# fig, axa = plt.subplots(nrows=3, ncols=1, 
#                               sharex=True, sharey=False, figsize = (6.3,3*3.54))
# fig.suptitle('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
# axa[0].grid(True)
# axa[0].set_title('Desorption and Absorption (all Variants)')
# ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t,ax=axa[0],label=txtc)
# axa[0].set_ylabel(r'$D_{E,saturated}$ / -')
# axa[0].set_xlabel(None)
# axa[0].legend()
# axa[1].grid(True)
# axa[1].set_title('Desorption')
# ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
# axa[1].set_ylabel(r'$D_{E,saturated}$ / -')
# axa[1].set_xlabel(None)
# axa[1].legend()
# axa[2].grid(True)
# axa[2].set_title('Absorption')
# ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t[t.Variant>='G'],ax=axa[2],label=txta)
# axa[2].set_xlabel(r'$D_{mass,Water}$ / -')
# axa[2].set_ylabel(r'$D_{E,saturated}$ / -')
# axa[2].legend()
# Evac.plt_handle_suffix(fig,path=out_full+'-Regd-DMWtoG-DEFtoB',**plt_Fig_dict)

# fig, ax1 = plt.subplots()
# ax1.grid(True)
# ax1.set_title('%sRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
# ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t,ax=ax1, label='Complete')
# ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
# ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t[t.Variant>='G'],ax=ax1, label='Absorption')
# ax1.set_xlabel(r'$D_{mass,Water}$ / -')
# ax1.set_ylabel(r'$D_{E,saturated}$ / -')
# ax1.legend()
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=out_full+'-Rega-DMWtoG-DEFtoB',**plt_Fig_dict)

# #------------
# # if formula_in_dia:
# #     txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoG-DERtoB'].iloc[:3]).split(',')
# #     txtc = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtc,)
# #     txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoG-DERtoB'].iloc[:3]).split(',')
# #     txtd = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtd,)
# #     txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Abs-DMWtoG-DERtoB'].iloc[:3]).split(',')
# #     txta = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txta,)
# # else:
# #     txtc = r'$D_{E}$'
    
# # fig, axa = plt.subplots(nrows=3, ncols=1, 
# #                               sharex=True, sharey=False, figsize = (6.3,3*3.54))
# # fig.suptitle('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
# # axa[0].grid(True)
# # axa[0].set_title('Desorption and Absorption (all Variants)')
# # ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t,ax=axa[0],label=txtc)
# # axa[0].set_ylabel(r'$D_{E,saturated}$ / -')
# # axa[0].set_xlabel(None)
# # axa[0].legend()
# # axa[1].grid(True)
# # axa[1].set_title('Desorption')
# # ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
# # axa[1].set_ylabel(r'$D_{E,saturated}$ / -')
# # axa[1].set_xlabel(None)
# # axa[1].legend()
# # axa[2].grid(True)
# # axa[2].set_title('Absorption')
# # ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t[t.Variant>='G'],ax=axa[2],label=txta)
# # axa[2].set_xlabel(r'$D_{mass,Water}$ / -')
# # axa[2].set_ylabel(r'$D_{E,saturated}$ / -')
# # axa[2].legend()
# # Evac.plt_handle_suffix(fig,path=out_full+'-Regd-DMWtoG-DERtoB',**plt_Fig_dict)

# # fig, ax1 = plt.subplots()
# # ax1.grid(True)
# # ax1.set_title('%sRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
# # ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t,ax=ax1, label='Complete')
# # ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
# # ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t[t.Variant>='G'],ax=ax1, label='Absorption')
# # ax1.set_xlabel(r'$D_{mass,Water}$ / -')
# # ax1.set_ylabel(r'$D_{E,saturated}$ / -')
# # ax1.legend()
# # fig.suptitle('')
# # Evac.plt_handle_suffix(fig,path=out_full+'-Rega-DMWtoG-DERtoB',**plt_Fig_dict)

# #==================
# if formula_in_dia:
#     txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoA-DEFtoB'].iloc[:3]).split(',')
#     txtc = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txtc,)
#     txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoA-DEFtoB'].iloc[:3]).split(',')
#     txtd = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txtd,)
#     txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Abs-DMWtoA-DEFtoB'].iloc[:3]).split(',')
#     txta = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txta,)
# else:
#     txtc = r'$D_{E}$'
    
# fig, axa = plt.subplots(nrows=3, ncols=1, 
#                               sharex=True, sharey=False, figsize = (6.3,3*3.54))
# fig.suptitle('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
# axa[0].grid(True)
# axa[0].set_title('Desorption and Absorption (all Variants)')
# ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t,ax=axa[0],label=txtc)
# axa[0].set_ylabel(r'$D_{E,saturated}$ / -')
# axa[0].set_xlabel(None)
# axa[0].legend()
# axa[1].grid(True)
# axa[1].set_title('Desorption')
# ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
# axa[1].set_ylabel(r'$D_{E,saturated}$ / -')
# axa[1].set_xlabel(None)
# axa[1].legend()
# axa[2].grid(True)
# axa[2].set_title('Absorption')
# ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t[t.Variant>='G'],ax=axa[2],label=txta)
# axa[2].set_xlabel(r'$D_{mass,org}$ / -')
# axa[2].set_ylabel(r'$D_{E,saturated}$ / -')
# axa[2].legend()
# Evac.plt_handle_suffix(fig,path=out_full+'-Regd-DMWtoA-DEFtoB',**plt_Fig_dict)

# fig, ax1 = plt.subplots()
# ax1.grid(True)
# ax1.set_title('%sRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
# ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t,ax=ax1, label='Complete')
# ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
# ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t[t.Variant>='G'],ax=ax1, label='Absorption')
# ax1.set_xlabel(r'$D_{mass,Water}$ / -')
# ax1.set_ylabel(r'$D_{E,saturated}$ / -')
# ax1.legend()
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=out_full+'-Rega-DMWtoA-DEFtoB',**plt_Fig_dict)

# #------------
# # if formula_in_dia:
# #     txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoA-DERtoB'].iloc[:3]).split(',')
# #     txtc = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txtc,)
# #     txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoA-DERtoB'].iloc[:3]).split(',')
# #     txtd = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txtd,)
# #     txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Abs-DMWtoA-DERtoB'].iloc[:3]).split(',')
# #     txta = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txta,)
# # else:
# #     txtc = r'$D_{E}$'
    
# # fig, axa = plt.subplots(nrows=3, ncols=1, 
# #                               sharex=True, sharey=False, figsize = (6.3,3*3.54))
# # fig.suptitle('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
# # axa[0].grid(True)
# # axa[0].set_title('Desorption and Absorption (all Variants)')
# # ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t,ax=axa[0],label=txtc)
# # axa[0].set_ylabel(r'$D_{E,saturated}$ / -')
# # axa[0].set_xlabel(None)
# # axa[0].legend()
# # axa[1].grid(True)
# # axa[1].set_title('Desorption')
# # ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
# # axa[1].set_ylabel(r'$D_{E,saturated}$ / -')
# # axa[1].set_xlabel(None)
# # axa[1].legend()
# # axa[2].grid(True)
# # axa[2].set_title('Absorption')
# # ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t[t.Variant>='G'],ax=axa[2],label=txta)
# # axa[2].set_xlabel(r'$D_{mass,org}$ / -')
# # axa[2].set_ylabel(r'$D_{E,saturated}$ / -')
# # axa[2].legend()
# # Evac.plt_handle_suffix(fig,path=out_full+'-Regd-DMWtoA-DERtoB',**plt_Fig_dict)

# # fig, ax1 = plt.subplots()
# # ax1.grid(True)
# # ax1.set_title('%sRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
# # ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t,ax=ax1, label='Complete')
# # ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
# # ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t[t.Variant>='G'],ax=ax1, label='Absorption')
# # ax1.set_xlabel(r'$D_{mass,org}$ / -')
# # ax1.set_ylabel(r'$D_{E,saturated}$ / -')
# # ax1.legend()
# # fig.suptitle('')
# # Evac.plt_handle_suffix(fig,path=out_full+'-Rega-DMWtoA-DERtoB',**plt_Fig_dict)


plt_RegA(pdo=t, x='DMWtoA', y='DEFtoB', 
         xl=r'$D_{m,org}$',yl=r'$D_{E,sat}$', 
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus mass deviation to original'%name_Head,
         path=out_full+'-Rega-DMWtoA-DEFtoB', plt_Fig_dict=plt_Fig_dict)
plt_RegD(pdo=t, x='DMWtoA', y='DEFtoB', 
         xl=r'$D_{m,org}$',yl=r'$D_{E,sat}$', 
         formula_in_dia=formula_in_dia, Lin_reg_df=Lin_reg_df,
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus mass deviation to original'%name_Head,
         path=out_full+'-Regd-DMWtoA-DEFtoB', plt_Fig_dict=plt_Fig_dict)

plt_RegA(pdo=t, x='DMWtoG', y='DEFtoB', 
         xl=r'$D_{m,dry}$',yl=r'$D_{E,sat}$', 
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head,
         path=out_full+'-Rega-DMWtoG-DEFtoB', plt_Fig_dict=plt_Fig_dict)
plt_RegD(pdo=t, x='DMWtoG', y='DEFtoB', 
         xl=r'$D_{m,dry}$',yl=r'$D_{E,sat}$', 
         formula_in_dia=formula_in_dia, Lin_reg_df=Lin_reg_df,
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head,
         path=out_full+'-Regd-DMWtoG-DEFtoB', plt_Fig_dict=plt_Fig_dict)

plt_RegA(pdo=t, x='DMWtoG', y='DEFtoG', 
         xl=r'$D_{m,dry}$',yl=r'$D_{E,dry}$', 
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head,
         path=out_full+'-Rega-DMWtoG-DEFtoG', plt_Fig_dict=plt_Fig_dict)
plt_RegD(pdo=t, x='DMWtoG', y='DEFtoG', 
         xl=r'$D_{m,dry}$',yl=r'$D_{E,dry}$', 
         formula_in_dia=formula_in_dia, Lin_reg_df=Lin_reg_df,
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head,
         path=out_full+'-Regd-DMWtoG-DEFtoG', plt_Fig_dict=plt_Fig_dict)


plt_RegA(pdo=t2, x='DMWtoA', y='DEFlutoB', 
         xl=r'$D_{m,org}$',yl=r'$D_{E,sat}$', 
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus mass deviation to original'%name_Head,
         path=out_full+'-Rega-DMWtoA-DEFtoB', plt_Fig_dict=plt_Fig_dict)
plt_RegD(pdo=t2, x='DMWtoA', y='DEFlutoB', 
         xl=r'$D_{m,org}$',yl=r'$D_{E,sat}$', 
         formula_in_dia=formula_in_dia, Lin_reg_df=Lin_reg_df,
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus mass deviation to original'%name_Head,
         path=out_full+'-Regd-DMWtoA-DEFlutoB', plt_Fig_dict=plt_Fig_dict)

plt_RegA(pdo=t2, x='DMWtoG', y='DEFlutoB', 
         xl=r'$D_{m,dry}$',yl=r'$D_{E,sat}$', 
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head,
         path=out_full+'-Rega-DMWtoG-DEFlutoB', plt_Fig_dict=plt_Fig_dict)
plt_RegD(pdo=t2, x='DMWtoG', y='DEFlutoB', 
         xl=r'$D_{m,dry}$',yl=r'$D_{E,sat}$', 
         formula_in_dia=formula_in_dia, Lin_reg_df=Lin_reg_df,
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head,
         path=out_full+'-Regd-DMWtoG-DEFlutoB', plt_Fig_dict=plt_Fig_dict)

plt_RegA(pdo=t2, x='DMWtoG', y='DEFlutoG', 
         xl=r'$D_{m,dry}$',yl=r'$D_{E,dry}$', 
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head,
         path=out_full+'-Rega-DMWtoG-DEFlutoG', plt_Fig_dict=plt_Fig_dict)
plt_RegD(pdo=t2, x='DMWtoG', y='DEFlutoG', 
         xl=r'$D_{m,dry}$',yl=r'$D_{E,dry}$', 
         formula_in_dia=formula_in_dia, Lin_reg_df=Lin_reg_df,
         title='%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head,
         path=out_full+'-Regd-DMWtoG-DEFlutoG', plt_Fig_dict=plt_Fig_dict)

#%%% Exponential Regression

tmp=Regfitret(pdo=t2[t2.Variant<='G'], x='DMWtoA', y='DEFlutoB',
          # name='exponential', guess = dict(a=-0.01, b=0.1, c=-1),
          name='exponential_x0', guess = dict(a=0, b=0.1, c=-1),
          xl=r'D_{m,org}',yl=r'D_{E,sat}', t_form='{a:.3e},{b:.3e},{c:.3f}')
fig, ax1 = plt.subplots()
ax1.grid(True)
plt_ax_Reg(pdo=t2[t2.Variant<='G'], x='DMWtoA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True,
            xlabel=r'$D_{m,org}$', ylabel=r'$D_{E,sat}$', title='bla')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

tmp=Regfitret(pdo=t2[t2.Variant<='G'], x='DMWtoA', y='DEFlutoB',
          name='linear', guess = dict(a=-0.01, b=0.1),
          xl=r'D_{m,org}',yl=r'D_{E,sat}', t_form='{a:.3e},{b:.3e}')
fig, ax1 = plt.subplots()
ax1.grid(True)
plt_ax_Reg(pdo=t2[t2.Variant<='G'], x='DMWtoA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True,
            xlabel=r'$D_{m,org}$', ylabel=r'$D_{E,sat}$', title='bla')
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

# plt_Regc(t2[t2.Variant<='G'],y='DEFlutoB', x='DMWtoA',
#              fit=fit,
#              xl=r'$D_{m,org}$',yl=r'$D_{E,sat}$', 
# title='%s\nRegression of Youngs Modulus (fixed range) deviation versus mass deviation to original'%name_Head,
# path=out_full+'-Rege-DMWtoA-DEFlutoB', plt_Fig_dict=plt_Fig_dict)

# guess = dict(a=-0.01, b=0.1, c=-1)
# fit = model.fit(t2[t2.Variant>='G']['DEFlutoB'], 
#                 x=t2[t2.Variant>='G']['DMWtoA'],
#                 nan_policy='omit',**guess)
# Rquad    = 1 - fit.residual.var() / np.nanvar(t2[t2.Variant>='G']['DEFlutoB'])
# print(fit.fit_report())
# print('R²:',Rquad)

# plt_Regc(t2[t2.Variant>='G'],y='DEFlutoB', x='DMWtoA',
#              fit=fit,
#              xl=r'$D_{m,org}$',yl=r'$D_{E,sat}$', 
# title='%s\nRegression of Youngs Modulus (fixed range) deviation versus mass deviation to original'%name_Head,
# path=out_full+'-Rege-DMWtoA-DEFlutoB', plt_Fig_dict=plt_Fig_dict)

# tmp=Regfitret(pdo=cs.query("Variant<='G'"), x='RHenv', y='DMWtoG',
#           name='exponential', guess = dict(a=-0.01, b=0.1, c=-1),
#           xl=r'\Phi_{env}',yl=r'D_{m,dry}', t_form='{a:.3e},{b:.3e},{c:.3f}')
# fig, ax1 = plt.subplots()
# plt_ax_Reg(pdo=cs.query("Variant<='G'"), x='RHenv', y='DMWtoG',
#             fit=tmp, ax=ax1, label_f=True,
#             xlabel=r'$\Phi_{env}$', ylabel=r'$D_{m,dry}$', title='bla')
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

# tmp=Regfitret(pdo=cs.query("Variant<='G'"), x='RHenv', y='DMWtoG',
#           name='exponential_x0', guess = dict(a=0, b=0.1, c=-1),
#           xl=r'\Phi_{env}',yl=r'D_{m,dry}', t_form='{a:.3e},{b:.3e},{c:.3f}')
# fig, ax1 = plt.subplots()
# plt_ax_Reg(pdo=cs.query("Variant<='G'"), x='RHenv', y='DMWtoG',
#             fit=tmp, ax=ax1, label_f=True,
#             xlabel=r'$\Phi_{env}$', ylabel=r'$D_{m,dry}$', title='bla')
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

# tmp=Regfitret(pdo=dft.query("Variant<='G'"), x='Humidity_store', y='DMWtoG',
#           name='exponential_x0', guess = dict(a=0, b=0.1, c=0.03),
#           xl=r'\Phi_{env}',yl=r'D_{m,dry}', t_form='{a:.3e},{b:.3e},{c:.3f}')
# fig, ax1 = plt.subplots()
# plt_ax_Reg(pdo=dft.query("Variant<='G'"), x='Humidity_store', y='DMWtoG',
#             fit=tmp, ax=ax1, label_f=True,
#             xlabel=r'$\Phi_{env}$', ylabel=r'$D_{m,dry}$', title='bla')
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)
# tmp=Regfitret(pdo=dft.query("Variant>='G'"), x='Humidity_store', y='DMWtoG',
#           name='exponential_x0', guess = dict(a=0, b=0.001, c=0.03),
#           xl=r'\Phi_{env}',yl=r'D_{m,dry}', t_form='{a:.3e},{b:.3e},{c:.3f}')
# fig, ax1 = plt.subplots()
# plt_ax_Reg(pdo=dft.query("Variant>='G'"), x='Humidity_store', y='DMWtoG',
#             fit=tmp, ax=ax1, label_f=True,
#             xlabel=r'$\Phi_{env}$', ylabel=r'$D_{m,dry}$', title='bla')
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

# fig, ax1 = plt.subplots()
# tmp=Regfitret(pdo=dft.query("Variant<='G'"), x='Humidity_store', y='WC_vol',
#           name='exponential_x0', guess = dict(a=0, b=0.1, c=0.03),
#           xl=r'\Phi_{env}',yl=r'\Phi_{vol,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
# plt_ax_Reg(pdo=dft.query("Variant<='G'"), x='Humidity_store', y='WC_vol',
#             fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
#             xlabel=r'$\Phi_{env}$', ylabel=r'$\Phi_{vol,des}$', title='bla',
#             skws=dict(color=sns.color_palette("tab10")[3]),
#             lkws=dict(color=sns.color_palette("tab10")[1]))
# tmp=Regfitret(pdo=dft.query("Variant>='G'"), x='Humidity_store', y='WC_vol',
#           name='exponential_x0', guess = dict(a=0, b=0.1, c=0.03),
#           xl=r'\Phi_{env}',yl=r'\Phi_{vol,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
# plt_ax_Reg(pdo=dft.query("Variant>='G'"), x='Humidity_store', y='WC_vol',
#             fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
#             xlabel=r'$\Phi_{env}$ / %', ylabel=r'$\Phi_{vol}$ / -', 
#             title='Exponential regression of water content vs. relative storage humidity',
#             skws=dict(color=sns.color_palette("tab10")[2]),
#             lkws=dict(color=sns.color_palette("tab10")[0]))
fig, ax1 = plt.subplots()
tmp=Regfitret(pdo=dft.query("Variant<='G'"), x='RHstore', y='WC_vol',
          name='exponential_x0', guess = dict(a=0, b=0.1, c=10),
          xl=r'\Phi_{env}',yl=r'\Phi_{vol,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=dft.query("Variant<='G'"), x='RHstore', y='WC_vol',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$\Phi_{env}$', ylabel=r'$\Phi_{vol,des}$', title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=dft.query("Variant>='G'"), x='RHstore', y='WC_vol',
          name='exponential_x0', guess = dict(a=0, b=0.1, c=2),
          xl=r'\Phi_{env}',yl=r'\Phi_{vol,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=dft.query("Variant>='G'"), x='RHstore', y='WC_vol',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$\Phi_{env}$ / -', ylabel=r'$\Phi_{vol}$ / -', 
            title='Exponential regression of water content vs. relative storage humidity',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)


# fig, ax1 = plt.subplots()
# tmp=Regfitret(pdo=dft.query("Variant<='G' and Variant!='B'"),
#           x='Humidity_store', y='WC_vol',
#           name='exponential', guess = dict(a=0, b=0.1, c=0.03),
#           xl=r'\Phi_{env}',yl=r'\Phi_{vol,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
# plt_ax_Reg(pdo=dft.query("Variant<='G' and Variant!='B'"), 
#             x='Humidity_store', y='WC_vol',
#             fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
#             xlabel=r'$\Phi_{env}$', ylabel=r'$\Phi_{vol,des}$', title='bla',
#             skws=dict(color=sns.color_palette("tab10")[3]),
#             lkws=dict(color=sns.color_palette("tab10")[1]))
# tmp=Regfitret(pdo=dft.query("Variant>='G' and Variant!='L'"),
#               x='Humidity_store', y='WC_vol',
#           name='exponential', guess = dict(a=0, b=0.1, c=0.03),
#           xl=r'\Phi_{env}',yl=r'\Phi_{vol,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
# plt_ax_Reg(pdo=dft.query("Variant>='G' and Variant!='L'"), 
#             x='Humidity_store', y='WC_vol',
#             fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
#             xlabel=r'$\Phi_{env}$ / %', ylabel=r'$\Phi_{vol}$ / -', 
#             title='Exponential regression of water content vs. relative storage humidity',
#             skws=dict(color=sns.color_palette("tab10")[2]),
#             lkws=dict(color=sns.color_palette("tab10")[0]))
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)


# angestrebte Luftfeuchte
fig, ax1 = plt.subplots()
tmp=Regfitret(pdo=dft.query("Variant<='G' and Variant!='B'"),
          x='RHenv', y='WC_vol',
          name='exponential', guess = dict(a=0, b=0.1, c=0.03),
          xl=r'\Phi_{env}',yl=r'\Phi_{vol,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=dft.query("Variant<='G' and Variant!='B'"), 
            x='RHenv', y='WC_vol',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$\Phi_{env}$', ylabel=r'$\Phi_{vol,des}$', title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=dft.query("Variant>='G' and Variant!='L'"),
              x='RHenv', y='WC_vol',
          name='exponential', guess = dict(a=0, b=0.1, c=0.03),
          xl=r'\Phi_{env}',yl=r'\Phi_{vol,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=dft.query("Variant>='G' and Variant!='L'"), 
            x='RHenv', y='WC_vol',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$\Phi_{env}$ / %', ylabel=r'$\Phi_{vol}$ / -', 
            title='Exponential regression of water content vs. envisaged relative storage humidity',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

fig, ax1 = plt.subplots()
tmp=Regfitret(pdo=dft.query("Variant<='G'and Variant!='B'"),
              x='RHenv', y='WC_vol',
          name='exponential_x0', guess = dict(a=0, b=0.01, c=5),
          xl=r'\Phi_{env}',yl=r'\Phi_{vol,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=dft.query("Variant<='G'and Variant!='B'"), 
           x='RHenv', y='WC_vol',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$\Phi_{env}$', ylabel=r'$\Phi_{vol,des}$', title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=dft.query("Variant>='G'and Variant!='L'"), 
              x='RHenv', y='WC_vol',
          name='exponential_x0', guess = dict(a=0, b=0.01, c=5),
          xl=r'\Phi_{env}',yl=r'\Phi_{vol,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=dft.query("Variant>='G'and Variant!='L'"), 
           x='RHenv', y='WC_vol',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$\Phi_{env}$ / -', ylabel=r'$\Phi_{vol}$ / -', 
            title='Exponential regression of water content vs. envisaged relative storage humidity',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)



fig, ax1 = plt.subplots()
tmp2=pd.concat([cs['WC_vol_rDA'],cEEm_eva['DEFlutoB']],axis=1)
tmp=Regfitret(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
          name='exponential_x0', guess = dict(a=0, b=0.01, c=-1),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
          name='exponential', guess = dict(a=0.01, b=0.01, c=1),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$', 
            title='Relation between relative deviation of Youngs Modulus to saturated\nand relative deviation of water content to fresh',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

fig, ax1 = plt.subplots()
tmp2=pd.concat([cs['WC_vol'],cEEm_eva['DEFlutoB']],axis=1)
tmp=Regfitret(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol', y='DEFlutoB',
          name='exponential', guess = dict(a=0.01, b=0.01, c=-1),
          xl=r'\Phi_{vol}',yl=r'D_{E,sat,des}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$\Phi_{vol}$', ylabel=r'$D_{E,sat}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol', y='DEFlutoB',
          name='exponential', guess = dict(a=0.01, b=0.01, c=1),
          xl=r'\Phi_{vol}',yl=r'D_{E,sat,ads}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$\Phi_{vol}$', ylabel=r'$D_{E,sat}$', 
            title='Relation between relative deviation of Youngs Modulus to saturated and water content',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

fig, ax1 = plt.subplots()
tmp2=pd.concat([cs['RHenv'],cEEm_eva['DEFlutoB']],axis=1)
tmp=Regfitret(pdo=tmp2.query("Variant<='G' and Variant!='B'"),
              x='RHenv', y='DEFlutoB',
          name='linear', guess = dict(a=0.01, b=0.01),
          xl=r'\Phi_{env}',yl=r'D_{E,sat,des}', t_form='{a:.3e},{b:.3e}')
plt_ax_Reg(pdo=tmp2.query("Variant<='G' and Variant!='B'"),
              x='RHenv', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$\Phi_{env}$', ylabel=r'$D_{E,sat}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp2.query("Variant>='G' and Variant!='L'"),
              x='RHenv', y='DEFlutoB',
          name='linear', guess = dict(a=0.01, b=0.01),
          xl=r'\Phi_{env}',yl=r'D_{E,sat,ads}', t_form='{a:.3e},{b:.3e}')
plt_ax_Reg(pdo=tmp2.query("Variant>='G'and Variant!='L'"),
              x='RHenv', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$\Phi_{env}$', ylabel=r'$D_{E,sat}$', 
            title='Relation between relative deviation of Youngs Modulus to saturated\nand envisaged relative storage humidity',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)


fig, ax1 = plt.subplots()
tmp2=pd.concat([cs['WC_vol_rDA'],cEEm_eva['DEFlutoG']],axis=1)
tmp=Regfitret(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoG',
          name='linear', guess = dict(a=0.001, b=0.01),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,dry,des}', t_form='{a:.3e},{b:.3e}')
plt_ax_Reg(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoG',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,dry}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoG',
          name='linear', guess = dict(a=0.01, b=0.01),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,dry,ads}', t_form='{a:.3e},{b:.3e}')
plt_ax_Reg(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoG',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,dry}$', 
            title='Relation between relative deviation of Youngs Modulus to dry\nand relative deviation of water content to fresh',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)


fig, ax1 = plt.subplots()
tmp2=pd.concat([cs['WC_vol'],cEEm_eva['DEFlutoG']],axis=1)
tmp=Regfitret(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol', y='DEFlutoG',
          name='linear', guess = dict(a=0.001, b=0.01),
          xl=r'\Phi_{vol}',yl=r'D_{E,dry,des}', t_form='{a:.3f},{b:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol', y='DEFlutoG',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$\Phi_{vol}$', ylabel=r'$D_{E,dry}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol', y='DEFlutoG',
          name='linear', guess = dict(a=0.01, b=0.01),
          xl=r'\Phi_{vol}',yl=r'D_{E,dry,ads}', t_form='{a:.3f},{b:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol', y='DEFlutoG',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$\Phi_{vol}$', ylabel=r'$D_{E,dry}$', 
            title='Relation between relative deviation of Youngs Modulus to dry and water content',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('') 
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)
fig, ax1 = plt.subplots()
tmp2=pd.concat([cs['WC_vol'],cEEm_eva['DEFlutoG']],axis=1)
tmp=Regfitret(pdo=tmp2.query("Variant<'G'"),
              x='WC_vol', y='DEFlutoG',
          name='exponential_x0', guess = dict(a=0, b=0.1, c=-10),
          xl=r'\Phi_{vol}',yl=r'D_{E,dry,des}', t_form='{a:.3f},{b:.3f},{c:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant<'G'"),
              x='WC_vol', y='DEFlutoG',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$\Phi_{vol}$', ylabel=r'$D_{E,dry}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp2.query("Variant>'G'"),
              x='WC_vol', y='DEFlutoG',
            name='exponential_x0', guess = dict(a=0, b=0.1, c=-10),
          xl=r'\Phi_{vol}',yl=r'D_{E,dry,ads}', t_form='{a:.3f},{b:.3f},{c:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant>'G'"),
              x='WC_vol', y='DEFlutoG',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$\Phi_{vol}$', ylabel=r'$D_{E,dry}$', 
            title='Relation between relative deviation of Youngs Modulus to dry and water content',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)



fig, ax1 = plt.subplots()
tmp2=pd.concat([cs['WC_vol'],cH_eva['DHAPntoG']],axis=1)
tmp=Regfitret(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol', y='DHAPntoG',
          name='linear', guess = dict(a=0.001, b=0.01),
          xl=r'\Phi_{vol}',yl=r'D_{HAN,dry,des}', t_form='{a:.3f},{b:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol', y='DHAPntoG',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Desorption',
            xlabel=r'$\Phi_{vol}$', ylabel=r'$D_{HAN,dry}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol', y='DHAPntoG',
          name='linear', guess = dict(a=0.01, b=0.01),
          xl=r'\Phi_{vol}',yl=r'D_{HAN,dry,ads}', t_form='{a:.3f},{b:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol', y='DHAPntoG',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Adsorption',
            xlabel=r'$\Phi_{vol}$', ylabel=r'$D_{HAN,dry}$', 
            title='Relation between relative deviation of normed hystereis area to dry and water content',
            skws=dict(color=sns.color_palette("tab10")[2]),
            lkws=dict(color=sns.color_palette("tab10")[0]))
fig.suptitle('') 
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)

#%%% Hypothest Serie
# tmp=pd.concat([cs['WC_vol_rDA'],cEEm_eva['DEFlutoB']],axis=1)
# tmp2=tmp.query("Variant<='G'").groupby(axis=0,level='Key').apply(lambda x: Regfitret(pdo=x, 
#           x='WC_vol_rDA', y='DEFlutoB',
#           name='exponential_x0', guess = dict(a=0, b=0.01, c=-1),
#           xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat,des}', 
#           t_form='{a:.3e},{b:.3e},{c:.3f}'))
# tmp2=tmp2.apply(pd.Series)

# fig, ax1 = plt.subplots()
# for i in tmp2.index:
#     plt_ax_Reg(pdo=tmp.query("Variant<='G'"), x='WC_vol_rDA', y='DEFlutoB',
#                 fit=tmp2.loc[i], ax=ax1, label_f=False,
#                 xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$', title=None)
# ax1.legend([None])
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)
fig, ax1 = plt.subplots()
tmp2=pd.concat([cs[['Series','WC_vol_rDA']],cEEm_eva['DEFlutoB']],axis=1)
tmp21=tmp2.query("Series=='1'")
tmp22=tmp2.query("Series=='2'")
tmp=Regfitret(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
          name='exponential_x0', guess = dict(a=0, b=0.01, c=-1),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data complete',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp21.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
          name='exponential_x0', guess = dict(a=0, b=0.01, c=-1),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat,1}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp21.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Series 1',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[0]),
            lkws=dict(color=sns.color_palette("tab10")[2]))
tmp=Regfitret(pdo=tmp22.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
          name='exponential_x0', guess = dict(a=0, b=0.01, c=-1),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat,2}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp22.query("Variant<='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Series 2',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$', 
            title='Relation between relative deviation of Youngs Modulus to saturated\nand relative deviation of water content to fresh (Desorption, separated by series)',
            skws=dict(color=sns.color_palette("tab10")[4]),
            lkws=dict(color=sns.color_palette("tab10")[5]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)
fig, ax1 = plt.subplots()
tmp2=pd.concat([cs[['Series','WC_vol_rDA']],cEEm_eva['DEFlutoB']],axis=1)
tmp21=tmp2.query("Series=='1'")
tmp22=tmp2.query("Series=='2'")
tmp=Regfitret(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
          name='exponential', guess = dict(a=0.01, b=0.01, c=1),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp2.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data complete',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$  ', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[3]),
            lkws=dict(color=sns.color_palette("tab10")[1]))
tmp=Regfitret(pdo=tmp21.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
          name='exponential', guess = dict(a=0.01, b=0.01, c=1),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat,1}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp21.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Series 1',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$', 
            title='bla',
            skws=dict(color=sns.color_palette("tab10")[0]),
            lkws=dict(color=sns.color_palette("tab10")[2]))
tmp=Regfitret(pdo=tmp22.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
          name='exponential', guess = dict(a=0.01, b=0.01, c=1),
          xl=r'D_{\Phi_{vol},org}',yl=r'D_{E,sat,2}', t_form='{a:.3e},{b:.3e},{c:.3f}')
plt_ax_Reg(pdo=tmp22.query("Variant>='G'"),
              x='WC_vol_rDA', y='DEFlutoB',
            fit=tmp, ax=ax1, label_f=True, label_d='Data Series 2',
            xlabel=r'$D_{\Phi_{vol},org}$', ylabel=r'$D_{E,sat}$', 
            title='Relation between relative deviation of Youngs Modulus to saturated\nand relative deviation of water content to fresh (Adsorption, separated by series)',
            skws=dict(color=sns.color_palette("tab10")[4]),
            lkws=dict(color=sns.color_palette("tab10")[5]))
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)
#%% Close log

log_mg.close()