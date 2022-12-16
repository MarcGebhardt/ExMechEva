# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:01:15 2021

@author: mgebhard
"""


import os
import copy
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import Eva_common as Evac

plt.rcParams['figure.figsize'] = [6.3,3.54]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size']= 8.0
#pd.set_option('display.expand_frame_repr', False)
plt.rcParams['lines.linewidth']= 1.0
plt.rcParams['lines.markersize']= 4.0
plt.rcParams['markers.fillstyle']= 'none'
plt.rcParams['axes.grid']= True
# sns.set_theme(context="paper",style="whitegrid",
#               font="DejaVu Sans",palette="tab10",
#               rc={'figure.dpi': 300.0,'figure.figsize': [16/2.54, 8/2.54],
#                   'font.size':8, 'ytick.alignment': 'center',
#                   'axes.titlesize':10, 'axes.titleweight': 'bold',
#                   'axes.labelsize':8,
#                   'xtick.labelsize': 8,'ytick.labelsize': 8,
#                   'legend.title_fontsize': 8,'legend.fontsize': 8,
#                   'lines.linewidth': 1.0,'lines.markersize': 4.0,
#                   'markers.fillstyle': 'none'})

# =============================================================================
#%% Einlesen und auswählen
ptype="TBT"
# ptype="ACT"
# ptype="ATT"
# no_stats_fc = ['1.11','1.12','1.21','1.22','1.31','2.21','3.11','3.21']
no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
               'B01.1','B01.2','B01.3', 'B02.3',
               'C01.1','C01.2','C01.3', 'C02.3',
               'D01.1','D01.2','D01.3', 'D02.3',
               'F01.1','F01.2','F01.3', 'F02.3',
               'G01.1','G01.2','G01.3', 'G02.3']
# var_suffix = ["A","B","C","D"] #Suffix of variants of measurements (p.E. diffferent moistures)
var_suffix = [""] #Suffix of variants of measurements (p.E. diffferent moistures)

VIPar_plt_renamer = {'fy':'$f_{y}$','fu':'$f_{u}$','fb':'$f_{b}$',
                     'ey_con':r'$\epsilon_{y,con}$','Wy_con':'$W_{y,con}$',
                     'eu_con':r'$\epsilon_{u,con}$','Wu_con':'$W_{u,con}$',
                     'eb_con':r'$\epsilon_{b,con}$','Wb_con':'$W_{b,con}$',
                     'ey_opt':r'$\epsilon_{y,opt}$','Wy_opt':'$W_{y,opt}$',
                     'eu_opt':r'$\epsilon_{u,opt}$','Wu_opt':'$W_{u,opt}$',
                     'eb_opt':r'$\epsilon_{b,opt}$','Wb_opt':'$W_{b,opt}$',
                     'E_con': '$E_{con}$','E_opt': '$E_{opt}$',
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
                     'ind_U_max': r'$ind_{u,max}$','ind_U_mean': r'$\overline{ind}_{u}$'}
Donor_dict={"LEIULANA_52-17":"PT2",
            "LEIULANA_67-17":"PT3",
            "LEIULANA_57-17":"PT4",
            "LEIULANA_48-17":"PT5",
            "LEIULANA_22-17":"PT6",
            "LEIULANA_60-17":"PT7"}

# path = "D:/Gebhardt/Veröffentlichungen/2022-X-X_Three_point_bending/ADD/Methodenvgl/211201/"
# path = "D:/Gebhardt/Spezial/DBV/Methodenvgl/220512/"
# path = "D:/Gebhardt/Spezial/DBV/Methodenvgl/220905/" #Kopie Daten 220512 mit neuer Auswertung (SUP)
path = "D:/Gebhardt/Spezial/DBV/Methodenvgl/221017/" #Kopie Daten 220512 mit neuer Auswertung (SUP)
name_paper="Paper_"
name_supmat="SUP_"
name_in   = "B3-B7_TBT-Summary_new2"
name_out  = "B3-B7_TBT-Conclusion"
name_Head = "Youngs Modulus determination methods - Compact bone"
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
          'Corpus vertebrae lumbales':  'CVlu'}
YM_con=['inc','R','A0Al','meanwoso']
# YM_opt=['inc','R','G1Mgwt','mean'] # Methodenumbenennung 211203
YM_opt=['inc','R','D2Mgwt','meanwoso']
YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
YM_opt_str='E_{}_{}_{}_{}'.format(*YM_opt)
VIParams_gen=["Designation","Origin","Donor"]
VIParams_geo=["thickness_mean","width_mean",
              "Area_CS","Volume","geo_MoI_mid","Density_app"]
VIParams_mat=["fy","ey_opt","Wy_opt","fu","eu_opt","Wu_opt",YM_con_str,YM_opt_str]
VIParams_don=["Sex","Age","BMI",
              "ICDCodes","Special_Anamnesis","Note_Anamnesis",
              "Fixation","Note_Fixation"]
VIParams_rename = {'geo_MoI_mid':'MoI_mid',
                    YM_con_str:'E_con',YM_opt_str:'E_opt'}


out_full= os.path.abspath(path+name_out)
out_paper= os.path.abspath(path+name_paper)
out_supmat= os.path.abspath(path+name_supmat)
path_doda = 'F:/Messung/000-PARAFEMM_Patientendaten/PARAFEMM_Donordata_full.xlsx'
h5_conc = 'Summary'
h5_data = 'Add_/Measurement'
h5_geof = 'Add_/Opt_Geo_Fit'
VIParams = copy.deepcopy(VIParams_geo)
VIParams.extend(VIParams_mat)

log_mg=open(out_full+'.log','w')
Evac.MG_strlog(name_out, log_mg, printopt=False)
Evac.MG_strlog("\n   Paths:", log_mg, printopt=False)
Evac.MG_strlog("\n   - in:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(path+name_in+'.h5'), log_mg, printopt=False)
Evac.MG_strlog("\n   - out:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(out_full), log_mg, printopt=False)
Evac.MG_strlog("\n   Donors:"+Evac.str_indent('\n{}'.format(pd.Series(Donor_dict).to_string()),5), log_mg, printopt=False)


data_read = pd.HDFStore(path+name_in+'.h5','r')
dfa=data_read.select(h5_conc)
dft=data_read.select(h5_data)
dfg=data_read.select(h5_geof)
data_read.close()
dfg=pd.DataFrame(list(dfg), index=dfg.index)

del dfa['No']
dfa.Failure_code  = Evac.list_cell_compiler(dfa.Failure_code)
dfa['statistics'] = Evac.list_interpreter(dfa.Failure_code, no_stats_fc)


h=dfa.Origin
for i in relist:
    h=h.str.replace(i,'')
h2=h.map(Locdict)
if (h2.isna()).any(): print('Locdict have missing/wrong values! \n   (Lnr: %s)'%['{:d}'.format(i) for i in h2.loc[h2.isna()].index])
dfa.insert(3,'Origin_short',h)
dfa.insert(4,'Origin_sshort',h2)
del h, h2

cs = dfa.loc[dfa.statistics]
cs_num_cols = cs.select_dtypes(include=['int','float']).columns



Methods_excl_names=["F4Agfu","F4Sgfu","F4Mgfu",
                    "F4gfu"]
Methods_excl_str='|'.join(Methods_excl_names)



#%% Statistische Auswertungen
d=cs[cs_num_cols].agg(['min','max','mean','median','std'],
         **{'skipna':True,'numeric_only':True,'ddof':1})
# d.loc(axis=1)['No':'CS_type']=np.nan
# d.loc(axis=1)['Failure_code':'statistics']=np.nan


d.loc['ouliers']      = cs[cs_num_cols].agg(Evac.stat_outliers,
                                            **{'option':'IQR', 'span':1.5,
                                               'out':'all', 'outsort' : 'ascending'})
d.loc['ouliers_high'] = cs[cs_num_cols].agg(Evac.stat_outliers,
                                            **{'option':'IQR', 'span':1.5,
                                               'out':'higher', 'outsort' : 'ascending'})
d.loc['ouliers_low']  = cs[cs_num_cols].agg(Evac.stat_outliers,
                                            **{'option':'IQR', 'span':1.5,
                                               'out':'lower', 'outsort' : 'descending'})
d.loc['ouliers_not']  = cs[cs_num_cols].agg(Evac.stat_outliers,
                                            **{'option':'IQR', 'span':1.5,
                                               'out':'inner', 'outsort' : None})


#%% Zusammenfassung nach Ort (Auswertung):
h_eva = cs.groupby('Origin_short')[cs_num_cols].agg(['size','min','max','mean','std'],
                                       **{'skipna':True,'numeric_only':True,'ddof':1})
h_eva_add = cs.groupby('Origin_short')[cs_num_cols].agg([Evac.stat_outliers],
                                                      **{'option':'IQR', 'span':1.5,
                                                          'out':'all', 'outsort' : 'ascending'})
h_eva_add = h_eva_add.rename(columns={'stat_outliers':'outliers'})
h_eva_c=h_eva.join(h_eva_add)
h_eva_c=h_eva.join(h_eva_add).stack()
h_eva_c.columns=h_eva.columns.get_level_values(0).drop_duplicates()

# h_eva=h_eva.stack()
#%% Speicherung
h_eva_c.to_csv(out_full+'-Loc.csv',sep=';')
#muss besser gehen (h_eva.loc[pd.IndexSlice[:,'Min'],:]=c.groupby(['Origin_short']).min())
dfa.append(d,sort=False).to_csv(out_full+'.csv',sep=';') 

data_store = pd.HDFStore(out_full+'.h5')
data_store['Material_Data'] = dfa   # write to HDF5
data_store['Material_Conclusion'] = d   # write to HDF5
data_store['Material_Conclusion_Location'] = h_eva_c   # write to HDF5
data_store['Test_Data'] = dft   # write to HDF5
data_store.close()


#%% HTML plot
if False:
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
    
    fig = go.Figure()
    fig.update_layout(title={'text':"<b> Stress strain curves </b> <br>(%s)" %name_Head,
                              'y':.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      xaxis_title="Strain [-]",
                      yaxis_title="Stress [MPa]",
                      legend_title='<b> Specimen </b>')
    # Add traces
    for i in dft.index:
        End_step=dft[i].loc[dft[i].VIP_m.str.contains('U').fillna(False)].index[0]
        fig.add_trace(go.Scatter(x=dft[i].Strain.loc[:End_step],
                                  y=dft[i].Stress.loc[:End_step],
                                  mode='lines',line={'dash':'dash'},
                                  name=i.split('_')[2]+' | '+i.split('_')[4],
                                  legendgroup = 'conventional'))
        fig.add_trace(go.Scatter(x=dft[i].Strain_opt_c_M.loc[:End_step],
                                  y=dft[i].Stress.loc[:End_step],
                                  mode='lines',line={'dash':'dot'},
                                  name=i.split('_')[2]+' | '+i.split('_')[4],
                                  legendgroup = 'optical'))
        
    config = {
      'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 500,
        'width': 700,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
        },
      'editable': True}
    fig.show(config=config)
    
#%% Statistik

alpha=0.05
# alpha=0.10
# Testtype
stat_ttype_parametric=False

if stat_ttype_parametric:
    mpop="ANOVA"
    # Tukey HSD test:
    MComp_kws={'do_mcomp_a':2, 'mcomp':'TukeyHSD', 'mpadj':'', 
                'Ffwalpha':2, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    # # Independendent T-test:
    # MComp_kws={'do_mcomp_a':2, 'mcomp':'ttest_ind', 'mpadj':'holm', 
    #            'Ffwalpha':2, 'mkws':{'equal_var': False, 'nan_policy': 'omit'},
    #            'add_T_ind':3, 'add_out':True}
    mcorr="pearson"
else:
    mpop="Kruskal-Wallis H-test"
    # Mann-Whitney U test:
    MComp_kws={'do_mcomp_a':2, 'mcomp':'mannwhitneyu', 'mpadj':'holm', 
                'Ffwalpha':2, 'mkws':{}, 'add_T_ind':3, 'add_out':True}
    mcorr="spearman"
    
#%%%% Prepare
# E-Methoden zu Multiindex
c_E = cs.loc(axis=1)[cs.columns.str.startswith('E_')]
c_E.columns = c_E.columns.str.split('_', expand=True)
c_E = c_E.droplevel(0,axis=1)
c_E.columns.names=['Determination','Range','Method','Parameter']
# E-Methoden relevant
c_E_lsq = c_E.loc(axis=1)['lsq']
c_E_inc = c_E.loc(axis=1)['inc']

idx = pd.IndexSlice
# c_E_lsq_m=c_E.loc(axis=1)[idx['lsq',:,:,'E']].droplevel([0],axis=1)
# c_E_inc_m=c_E.loc(axis=1)[idx['inc',:,:,'meanwoso']].droplevel([0],axis=1)
c_E_lsq_m=c_E_lsq.loc(axis=1)[idx[:,:,'E']].droplevel([2],axis=1)
c_E_inc_m=c_E_inc.loc(axis=1)[idx[:,:,'meanwoso']].droplevel([2],axis=1)

c_E_lsq_Rquad  = c_E_lsq.loc(axis=1)[idx[:,:,'Rquad']].droplevel([2],axis=1)
c_E_inc_stnorm = c_E_inc.loc(axis=1)[idx[:,:,'stdnwoso']].droplevel([2],axis=1)


# E-Methoden-vgl - Einflusseliminierung
# E_inc_m_R_MI  = c_E_inc_m['R'].droplevel(1,axis=1)
E_inc_m_R_MI  = c_E_inc_m['R']
i1 = E_inc_m_R_MI.columns.str[0:1].to_list()
i2 = E_inc_m_R_MI.columns.str[1:2].to_list()
i3 = E_inc_m_R_MI.columns.str[2:3].to_list()
i4 = E_inc_m_R_MI.columns.str[3:4].to_list()
i5 = E_inc_m_R_MI.columns.str[4: ].to_list()
itup = list(zip(*[i1,i2,i3,i4,i5]))
E_inc_m_R_MI_cols = pd.MultiIndex.from_tuples(itup, names=["Method", "Basis", "Impact", "Position", "Special"])
E_inc_m_R_MI.columns = E_inc_m_R_MI_cols
tg = E_inc_m_R_MI.groupby(level=2,axis=1)
E_inc_m_A = tg.get_group('A').droplevel([2],axis=1)
E_inc_m_S = tg.get_group('S').droplevel([2],axis=1)
E_inc_m_M = tg.get_group('M').droplevel([2],axis=1)
E_inc_m_C = tg.get_group('C').droplevel([2],axis=1)

Comp_E_inc_m_SA = ((E_inc_m_S-E_inc_m_A)/E_inc_m_A).dropna(axis=1,how='all')
Comp_E_inc_m_SA.columns = Comp_E_inc_m_SA.columns.to_series().apply(lambda x: '{0}{1}{2}{3}'.format(*x)).values
Comp_E_inc_m_MA = ((E_inc_m_M-E_inc_m_A)/E_inc_m_A).dropna(axis=1,how='all')
Comp_E_inc_m_MA.columns = Comp_E_inc_m_MA.columns.to_series().apply(lambda x: '{0}{1}{2}{3}'.format(*x)).values
Comp_E_inc_m_MS = ((E_inc_m_M-E_inc_m_S)/E_inc_m_S).dropna(axis=1,how='all')
Comp_E_inc_m_MS.columns = Comp_E_inc_m_MS.columns.to_series().apply(lambda x: '{0}{1}{2}{3}'.format(*x)).values
Comp_E_inc_m_CM = ((E_inc_m_C-E_inc_m_M)/E_inc_m_M).dropna(axis=1,how='all')
Comp_E_inc_m_CM.columns = Comp_E_inc_m_CM.columns.to_series().apply(lambda x: '{0}{1}{2}{3}'.format(*x)).values

Comp_E_inc_m_IE = pd.concat([Comp_E_inc_m_SA, Comp_E_inc_m_MS, Comp_E_inc_m_MA],
                            axis=1, keys=['SA','MS','MA'], names=['Inf','Method'])
Comp_E_inc_m_IE = Comp_E_inc_m_IE.swaplevel(axis=1)

# E-Methoden-vgl - least-square to incremental
# Comp_E_lsqinc=(c_E_lsq_m.droplevel([2],axis=1)-c_E_inc_m.droplevel([2],axis=1))/c_E_inc_m.droplevel([2],axis=1)
Comp_E_lsqinc=(c_E_lsq_m-c_E_inc_m)/c_E_inc_m
Comp_E_lsqinc=Comp_E_lsqinc.dropna(axis=1,how='all')

# E-Methoden-vgl - least-square to incremental
a=c_E_inc.loc(axis=1)['R',:,['meanwoso','stdnwoso']].droplevel(axis=1,level=[0])
Comp_E_con=pd.DataFrame([])
Comp_CV_con=pd.DataFrame([])
for i in a.droplevel(axis=1,level=[1]).columns:
    Comp_E_con[i] = (a[(i,'meanwoso')] - a[(YM_con[2],'meanwoso')])/a[(YM_con[2],'meanwoso')]
    Comp_CV_con[i] = (a[(i,'stdnwoso')] - a[(YM_con[2],'stdnwoso')])/a[(YM_con[2],'stdnwoso')]
    

# E-Check - analytische Lösung
c_C = cs.loc(axis=1)[cs.columns.str.startswith('Check_')]
c_C.columns = c_C.columns.str.split('_', expand=True)
c_C = c_C.droplevel(0,axis=1)
c_C.columns.names=['Location','Method']

c_C_MI = c_C.copy(deep=True)
i0 = c_C.columns.droplevel(1).to_list()
i1 = c_C.columns.droplevel(0).str[0:1].to_list()
i2 = c_C.columns.droplevel(0).str[1:2].to_list()
i3 = c_C.columns.droplevel(0).str[2:3].to_list()
i4 = c_C.columns.droplevel(0).str[3:4].to_list()
i5 = c_C.columns.droplevel(0).str[4: ].to_list()
itup = list(zip(*[i0,i1,i2,i3,i4,i5]))
c_C_MI.columns = pd.MultiIndex.from_tuples(itup, names=["Location","Method", "Basis", "Impact", "Position", "Special"])
c_C_g = c_C_MI.groupby(level=3,axis=1)
c_C_M = c_C_g.get_group('M')
c_C_C = c_C_g.get_group('C')
c_C_MC = pd.concat([c_C_M,c_C_C], axis=1)
i0=c_C_MC.columns.droplevel([1,2,3,4,5])
i1=c_C_MC.columns.to_series().apply(lambda x: '{1}{2}{3}{4}{5}'.format(*x)).values
c_C_MC.columns=pd.MultiIndex.from_arrays([i0,i1], names=["Location","Method"])

#%%%% Speicherung mean und std
c_E_lsq_m.agg(['min','max','mean','median','std']).to_csv(out_full+'_lsq_mean.csv',sep=';') 
c_E_inc_m.agg(['min','max','mean','median','std']).to_csv(out_full+'_inc_mean.csv',sep=';') 
c_E_inc_stnorm.agg(['min','max','mean','median','std']).to_csv(out_full+'_inc_std.csv',sep=';') 
c_C.agg(['min','max','mean','median','std']).to_csv(out_full+'_Check_DefInt.csv',sep=';') 

#%%% Logging


Evac.MG_strlog("\n "+"="*100, log_mg, printopt=False)
Evac.MG_strlog("\n Assessment code scheme exclsion:", log_mg, printopt=False)
a=Evac.list_interpreter(dfa.Failure_code, no_stats_fc, option='include')
b=dfa[a].loc(axis=1)['Failure_code','Note']
Evac.MG_strlog("\n\n  - total: (%d)"%len(dfa[a].index), log_mg, printopt=False)
# Evac.MG_strlog(Evac.str_indent("\n"+b.to_string(),5),
#                log_mg, printopt=False)

no_stats_fc_pre = ['A01.1','A01.2','A01.3', 'A02.3',
                   'B01.1','B01.2','B01.3', 'B02.3',
                   'C01.1','C01.2','C01.3', 'C02.3']
a=Evac.list_interpreter(dfa.Failure_code, no_stats_fc_pre, option='include')
b=dfa[a].loc(axis=1)['Failure_code','Note']
Evac.MG_strlog("\n\n  - before test: (%d)"%len(dfa[a].index), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("\n"+b.to_string(),5),
                log_mg, printopt=False)

no_stats_fc_eva = ['D01.1','D01.2','D01.3', 'D02.3',
                   'F01.1','F01.2','F01.3', 'F02.3',
                   'G01.1','G01.2','G01.3', 'G02.3']
a=Evac.list_interpreter(dfa.Failure_code, no_stats_fc_eva, option='include')
b=dfa[a].loc(axis=1)['Failure_code','Note']
Evac.MG_strlog("\n\n  -  during / after  test: (%d)"%len(dfa[a].index), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("\n"+b.to_string(),5),
               log_mg, printopt=False)




Evac.MG_strlog("\n "+"="*100, log_mg, printopt=False)
Evac.MG_strlog("\n Method comparison and debugging", log_mg, printopt=False)
Evac.MG_strlog("\n\n  - incremental to least-square:", log_mg, printopt=False)
a=Comp_E_lsqinc.agg(['max','idxmax','min','idxmin']).T.to_string()
Evac.MG_strlog(Evac.str_indent("\n"+a,5),
               log_mg, printopt=False)
Evac.MG_strlog("\n\n  - incremental Influence M/A:", log_mg, printopt=False)
a=Comp_E_inc_m_MA.agg(['max','idxmax','min','idxmin']).T.to_string()
Evac.MG_strlog(Evac.str_indent("\n"+a,5),
               log_mg, printopt=False)

def to_1D(series):
 return pd.Series([x for _list in series for x in _list])
def boolean_df(item_lists, unique_items):
    # Create empty dict
    bool_dict = {}
    # Loop through all the tags
    for i, item in enumerate(unique_items):
        # Apply boolean mask
        bool_dict[item] = item_lists.apply(lambda x: item in x)
    # Return the results as a dataframe
    return pd.DataFrame(bool_dict)

# t=c_E.agg(Evac.stat_outliers)
# t=c_E.loc(axis=1)[idx[:,:,:,['E','R','mean','stdnorm']]].agg(Evac.stat_outliers)
a=c_E.loc(axis=1)[idx[:,:,:,['E','R','meanwoso','stdnwoso']]].agg(Evac.stat_outliers)
a_unique = to_1D(a).drop_duplicates().sort_values().values
a_unique = np.delete(a_unique,a_unique=='nan')
a_b_df = boolean_df(a, a_unique).astype(int)

a_b_df_sum = a_b_df.sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  - stat ouliers (mean and StdNorm, first 20):", log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent('\n'+a_b_df_sum.iloc[:20].to_string(),5),
               log_mg, printopt=False)

# ----------------------------------------------------------------------------
Evac.MG_strlog("\n\n "+"="*100, log_mg, printopt=False)
Evac.MG_strlog("\n Inflence of shear deformation on total deformation (without indentation) in midspan:", log_mg, printopt=False)
Evac.MG_strlog(r"\n  $\gamma_{V}/(1+\gamma_{V}$", log_mg, printopt=False)
import Bending as Bend
a=cs.thickness_mean.apply(lambda x: Bend.gamma_V_det(poisson=0.3,t_mean=x,Length=20))
b=(a/(a+1)).agg(['min','max','mean','median','std',Evac.coefficient_of_variation])
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)

# ----------------------------------------------------------------------------
# Achtung: absolute Deviation only abs(relative Deviation), wenn alle Nenner positiv!!!

Evac.MG_strlog("\n\n "+"="*100, log_mg, printopt=False)
Evac.MG_strlog("\n Method comparison", log_mg, printopt=False)

Evac.MG_strlog("\n\n  - Comparison Youngs modulus least-squared to incremental determination:",
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Relative Deviation D_E,lsq-inc=((E_lsq -E_inc) / E_inc):",
               log_mg, printopt=False)
a=Comp_E_lsqinc.agg(['min','max','mean','median','std'])
Evac.MG_strlog(Evac.str_indent(a.T.to_string(),5), log_mg, printopt=False)
# b=a.loc['mean'].groupby(level='Range').agg(['min','max','mean','median','std'])
# Evac.MG_strlog(Evac.str_indent(b.to_string(),5), log_mg, printopt=False)
b=Comp_E_lsqinc.stack().agg(['min','max','mean','median','std'])
c = a.loc['mean'].groupby('Range').agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean of methods: \n{}".format(c),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Absolute Deviation |D_E,lsq-inc|=(|E_lsq -E_inc| / E_inc):",
               log_mg, printopt=False)
b=Comp_E_lsqinc.stack().abs().agg(['min','max','mean','median','std'])
c = a.loc['mean'].abs().groupby('Range').agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean of methods: \n{}".format(c),5),
               log_mg, printopt=False)


Evac.MG_strlog("\n\n  - Comparison Youngs modulus determination ranges ((E_F-E_R) / E_R):",
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Least-squares determination - coefficient of determination:",
               log_mg, printopt=False)
# c = c_E_lsq.loc(axis=1)[:,:,'Rquad'].groupby(axis=1,level=0).mean().agg(['mean','std'])
c = c_E_lsq.loc(axis=1)[:,:,'Rquad'].droplevel(2,axis=1).stack().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent(c.to_string(),5), log_mg, printopt=False)

# a=c_E_inc.loc(axis=1)[:,:,['meanwoso','stdnwoso']]
# # a=a['R']/a['F']
# a=(a['R']-a['F'])/a['F']
a = ((c_E_inc['F']-c_E_inc['R'])/c_E_inc['R'])
Evac.MG_strlog("\n    - Youngs modulus' mean without statistical outliers over increment:",
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Relative Deviation D_E,F-R=((E_F - E_R) / E_R):",
               log_mg, printopt=False)
b=a.loc(axis=1)[:,'meanwoso'].agg(['min','max','mean','median','std',Evac.meanwoso,Evac.stdwoso])
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'meanwoso']
c = b.droplevel(1,axis=1).stack().agg(['mean','std'])
d = b.mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Absolute Deviation |D_E,F-R|=(|E_F - E_R| / E_R):",
               log_mg, printopt=False)
c = b.droplevel(1,axis=1).stack().abs().agg(['mean','std'])
d = b.abs().mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)

Evac.MG_strlog("\n    - Youngs modulus' mean with statistical outliers over increment:",
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Relative Deviation D_E,F-R=((E_F - E_R) / E_R):",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'mean']
c = b.droplevel(1,axis=1).stack().agg(['mean','std'])
d = b.mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Absolute Deviation |D_E,F-R|=(|E_F - E_R| / E_R):",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'mean']
c = b.droplevel(1,axis=1).stack().abs().agg(['mean','std'])
d = b.abs().mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)



Evac.MG_strlog("\n    - Youngs modulus' coefficient of variation without statistical outliers over increment:",
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Relative Deviation D_CV,F-R=((CV_F - CV_R) / CV_R):",
               log_mg, printopt=False)
b=a.loc(axis=1)[:,'stdnwoso'].agg(['min','max','mean','median','std',Evac.meanwoso,Evac.stdwoso])
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'stdnwoso']
c = b.droplevel(1,axis=1).stack().agg(['mean','std'])
d = b.mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Absolute Deviation |D_CV,F-R|=(|CV_F - CV_R| / CV_R):",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'stdnwoso']
c = b.droplevel(1,axis=1).stack().abs().agg(['mean','std'])
d = b.abs().mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)

Evac.MG_strlog("\n    - Youngs modulus' coefficient of variation with statistical outliers over increment:",
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Relative Deviation D_CV,F-R=((CV_F - CV_R) / CV_R):",
               log_mg, printopt=False)
b=a.loc(axis=1)[:,'stdn'].agg(['min','max','mean','median','std',Evac.meanwoso,Evac.stdwoso])
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'stdn']
c = b.droplevel(1,axis=1).stack().agg(['mean','std'])
d = b.mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Absolute Deviation |D_CV,F-R|=(|CV_F - CV_R| / CV_R):",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'stdn']
c = b.droplevel(1,axis=1).stack().abs().agg(['mean','std'])
d = b.abs().mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)

# Evac.MG_strlog("\n    - Youngs modulus' coefficient of variation deviation with/without statistical outliers over increment:",
#                log_mg, printopt=False)
# a=c_E_inc.loc(axis=1)['F',:,'stdn'].droplevel(2,axis=1)
# b=c_E_inc.loc(axis=1)['F',:,'stdnwoso'].droplevel(2,axis=1)
# c= (a-b)/b

a = ((c_E_lsq['F']-c_E_lsq['R'])/c_E_lsq['R'])
Evac.MG_strlog("\n    - Youngs modulus' deviation (least-squares determination):",
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Relative Deviation D_E,F-R=((E_F - E_R) / E_R):",
               log_mg, printopt=False)
b=a.loc(axis=1)[:,'E'].agg(['min','max','mean','median','std',Evac.meanwoso,Evac.stdwoso])
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'E']
c = b.droplevel(1,axis=1).stack().agg(['mean','std'])
d = b.mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Absolute Deviation |D_E,F-R|=(|E_F - E_R| / E_R):",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'E']
c = b.droplevel(1,axis=1).stack().abs().agg(['mean','std'])
d = b.mean().abs().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)

Evac.MG_strlog("\n\n    - Youngs modulus' coefficient of determination deviation (least-squares determination):",
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Relative Deviation D_R,F-R=((R_F - R_R) / R_R):",
               log_mg, printopt=False)
b=a.loc(axis=1)[:,'Rquad'].agg(['min','max','mean','median','std',Evac.meanwoso,Evac.stdwoso])
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'Rquad']
c = b.droplevel(1,axis=1).stack().agg(['mean','std'])
d = b.mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n     - Relative Deviation D_R,F-R=((R_F - R_R) / R_R):",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):"%(Methods_excl_names),5),
               log_mg, printopt=False)
b = a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                  'Rquad']
c = b.droplevel(1,axis=1).abs().stack().agg(['mean','std'])
d = b.abs().mean().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)




Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)

Evac.MG_strlog("\n    - Youngs modulus' coefficient of variation (std/mean) (inc + R):",
               log_mg, printopt=False)
a=c_E_inc.loc(axis=1)['R',:,'meanwoso'].droplevel(axis=1,level=[0,2]).agg(['mean','std'])
a=(a.loc['std']/a.loc['mean']).sort_values()
Evac.MG_strlog(Evac.str_indent(a.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\n    - Youngs modulus' coefficient of variation without statistical outliers (std/mean) (inc + R):",
               log_mg, printopt=False)
b=c_E_inc.loc(axis=1)['R',:,'meanwoso'].droplevel(axis=1,level=[0,2]).agg([Evac.meanwoso,Evac.stdwoso])
b=(b.loc['stdwoso']/b.loc['meanwoso']).sort_values()
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)


Evac.MG_strlog("\n\n  - Comparison of methods to conventional:",
               log_mg, printopt=False)
a=c_E_inc.loc(axis=1)['R',:,'meanwoso'].droplevel(axis=1,level=[0,2])
Evac.MG_strlog("\n     - Relative Deviation D_E,opt-con=((E_opt - E_con) / E_opt):",
               log_mg, printopt=False)
# b=pd.DataFrame([])
# for i in a.columns:
#     b[i] = (a[i] - a[YM_con[2]])/a[YM_con[2]]
# b=b.mean().sort_values()
b=a.sub(a[YM_con[2]],axis=0).div(a[YM_con[2]],axis=0)
b=b.mean().sort_values()
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\n     - Absolute Deviation |D_E,opt-con|=(|E_opt - E_con| / E_opt):",
               log_mg, printopt=False)
b=a.sub(a[YM_con[2]],axis=0).div(a[YM_con[2]],axis=0)
b=b.abs().mean().sort_values()
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)



Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)
Evac.MG_strlog("\n\n  - Influence excamination:",
               log_mg, printopt=False)
agg_funcs=['min','max','mean','median','std',Evac.coefficient_of_variation]

Evac.MG_strlog("\n    - Relative Deviation D_E,S-A=((E_S - E_A) / E_A)",
               log_mg, printopt=False)
a=Comp_E_inc_m_SA.agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(a.T.to_string(),5), log_mg, printopt=False)
Methods_incl=['A2l','C2l','D1guw','D1gwt','D2guw','D2gwt']
Evac.MG_strlog(Evac.str_indent("Mean relevant (%s):"%(Methods_incl),5),
               log_mg, printopt=False)
c=Comp_E_inc_m_SA.loc[:,Methods_incl].stack().agg(['mean','std'])
d=a.loc['mean',Methods_incl].agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)

Evac.MG_strlog("\n    - Absolute Deviation |D_E,S-A|=(|E_S - E_A| / E_A)",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean relevant (%s):"%(Methods_incl),5),
               log_mg, printopt=False)
c=Comp_E_inc_m_SA.loc[:,Methods_incl].stack().abs().agg(['mean','std'])
d=a.loc['mean',Methods_incl].abs().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)

Evac.MG_strlog("\n    - Relative Deviation D_E,M-S=((E_M - E_S) / E_S)",
               log_mg, printopt=False)
a=Comp_E_inc_m_MS.agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(a.T.to_string(),5), log_mg, printopt=False)
# c = a.loc['mean',a.columns.difference(Methods_excl_names)].mean().agg(['mean','std'])
# Evac.MG_strlog(Evac.str_indent("Mean without exculdes (%s):\n  %2.5f  \u00B1 %2.5f"%(Methods_excl_names,*c),5),
#                log_mg, printopt=False)
Methods_incl=['B1l','B2l','C2l','D1guw','D1gwt','D2guw','D2gwt','G3g']
Evac.MG_strlog(Evac.str_indent("Mean relevant (%s):"%(Methods_incl),5),
               log_mg, printopt=False)
c=Comp_E_inc_m_MS.loc[:,Methods_incl].stack().agg(['mean','std'])
d=a.loc['mean',Methods_incl].agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
a = Comp_E_inc_m_MS.loc(axis=1)['D2gwt'].sort_values(ascending=False)
b = cs.loc[a.index[0:14],'thickness_2']
c = pd.concat([a[a.index[0:14]],b[a.index[0:14]]], axis=1)
Evac.MG_strlog("\n      - Deviation M/S and thickness in midspan",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent(c.to_string(),7), log_mg, printopt=False)
Evac.MG_strlog("\n    - Absolute Deviation |D_E,M-S|=(|E_M - E_S| / E_S)",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean relevant (%s):"%(Methods_incl),5),
               log_mg, printopt=False)
c=Comp_E_inc_m_MS.loc[:,Methods_incl].stack().abs().agg(['mean','std'])
a=Comp_E_inc_m_MS.agg(agg_funcs) # hier nochmals, da zwischendurch dicke
d=a.loc['mean',Methods_incl].abs().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)

Evac.MG_strlog("\n    - Relative Deviation D_E,M-A=((E_M - E_A) / E_A)",
               log_mg, printopt=False)
a=Comp_E_inc_m_MA.agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(a.T.to_string(),5), log_mg, printopt=False)
Methods_incl=['B1l','B2l','C2l','D1guw','D1gwt','D2guw','D2gwt']
Evac.MG_strlog(Evac.str_indent("Mean relevant (%s):"%(Methods_incl),5),
               log_mg, printopt=False)
c=Comp_E_inc_m_MA.loc[:,Methods_incl].stack().agg(['mean','std'])
d=a.loc['mean',Methods_incl].agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Absolute Deviation |D_E,M-A|=(|E_M - E_A| / E_A)",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean relevant (%s):"%(Methods_incl),5),
               log_mg, printopt=False)
c=Comp_E_inc_m_MA.loc[:,Methods_incl].stack().abs().agg(['mean','std'])
d=a.loc['mean',Methods_incl].abs().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)

Evac.MG_strlog("\n    - Relative Deviation D_E,C-M=((E_C - E_M) / E_M)",
               log_mg, printopt=False)
a=Comp_E_inc_m_CM.agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(a.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean:",5),
               log_mg, printopt=False)
c=Comp_E_inc_m_CM.stack().agg(['mean','std'])
d=a.loc['mean'].agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Absolute Deviation |D_E,C-M|=(|E_C - E_M| / E_M)",
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Mean:",5),
               log_mg, printopt=False)
c=Comp_E_inc_m_CM.stack().abs().agg(['mean','std'])
d=a.loc['mean'].abs().agg(['mean','std'])
Evac.MG_strlog(Evac.str_indent("  %2.5f \u00B1 %2.5f (Mean of methods: %2.5f \u00B1 %2.5f)"%(*c,*d,),5),
               log_mg, printopt=False)



Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)
Evac.MG_strlog("\n\n  - Population excamination:",
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Distance between mean and median of deviation to methods mean values:",
               log_mg, printopt=False)
a = c_E_inc_m['R']
b=((a-a.mean(axis=0))/a.mean(axis=0)).mean()
c=((a-a.mean(axis=0))/a.mean(axis=0)).median()
d=(c-b).sort_values(ascending=False)
Evac.MG_strlog(Evac.str_indent(d,5), log_mg, printopt=False)
Evac.MG_strlog("\n    - Distance between mean and median of deviation to methods mean values (only M)):",
               log_mg, printopt=False)
a = tg.get_group('M')
a.columns = a.columns.to_series().apply(lambda x: '{0}{1}{2}{3}{4}'.format(*x)).values
b=((a-a.mean(axis=0))/a.mean(axis=0)).mean()
c=((a-a.mean(axis=0))/a.mean(axis=0)).median()
d=(c-b).sort_values(ascending=False)
Evac.MG_strlog(Evac.str_indent(d,5), log_mg, printopt=False)

Evac.MG_strlog("\n    - Coefficient of variation of methods:",
               log_mg, printopt=False)
a = c_E_inc_m['R']
# c=Evac.coefficient_of_variation(a)
c=a.apply(Evac.coefficient_of_variation)
d=c.sort_values()
Evac.MG_strlog(Evac.str_indent(d,5), log_mg, printopt=False)
Evac.MG_strlog("\n    - Coefficient of variation of methods (only M)):",
               log_mg, printopt=False)
a = tg.get_group('M')
a.columns = a.columns.to_series().apply(lambda x: '{0}{1}{2}{3}{4}'.format(*x)).values
c=Evac.coefficient_of_variation(a)
d=c.sort_values()
Evac.MG_strlog(Evac.str_indent(d,5), log_mg, printopt=False)



Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)
Evac.MG_strlog("\n\n  - Compare choosen and conventional YM:",
               log_mg, printopt=False)
Evac.MG_strlog("\n    - Relative Deviation D_E,X-{}=((E_X - E_{}) / E_{}):".format(*[YM_con[2],]*3),
               log_mg, printopt=False)
a = Comp_E_con[['C2Ml','C2Cl','D2Mguw','D2Mgwt']]
b=a.agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\n    - Absolute Deviation |D_E,X-{}|=(|E_X - E_{}| / E_{}):".format(*[YM_con[2],]*3),
               log_mg, printopt=False)
b=a.abs().agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)



Evac.MG_strlog("\n\n "+"="*100, log_mg, printopt=False)

#%%% Einflussanalyse
# Zusätzlich Krümmung aus Kreisbogen:
bl = Bend.Bend_func_legion(name="FSE_fit")
bl.Builder(option="FSE_fixed")
curve_circmid=dfg['Fit_params_dict'].apply(lambda a: Evac.Geo_curve_TBC(bl['w_A']['d0'],
                                                                        a,20))


c_inf_geo = cs.loc(axis=1)['thickness_mean','width_mean', 
                           'Area_CS','Volume','Density_app']
c_inf_add = cs.loc(axis=1)['width_1','width_2','width_3',
                           'thickness_1','thickness_2','thickness_3',
                           'geo_MoI_max','geo_MoI_min','geo_MoI_mid',
                           'geo_curve_max','geo_curve_min','geo_curve_mid',
                           'ind_U_l','ind_U_r',
                           'ind_Fcon_l','ind_Fcon_r',
                           'ind_Fopt_l','ind_Fopt_r',
                           'ind_Rcon_l','ind_Rcon_r',
                           'ind_Ropt_l','ind_Ropt_r']

c_inf_add_new = pd.DataFrame([],index=c_inf_add.index,dtype='float64')
c_inf_add['geo_thick_max']   = c_inf_add[['thickness_1','thickness_2','thickness_3']].max(axis=1)
c_inf_add['geo_thick_min']   = c_inf_add[['thickness_1','thickness_2','thickness_3']].min(axis=1)
c_inf_add_new['thickness_mean'] = c_inf_geo['thickness_mean']
c_inf_add_new['geo_dthick']  = c_inf_add.eval('(geo_thick_max-geo_thick_min)/thickness_2')
c_inf_add['geo_width_max']   = c_inf_add[['width_1','width_2','width_3']].max(axis=1)
c_inf_add['geo_width_min']   = c_inf_add[['width_1','width_2','width_3']].min(axis=1)
c_inf_add_new['width_mean'] = c_inf_geo['width_mean']
c_inf_add_new['geo_dwidth']  = c_inf_add.eval('(geo_width_max-geo_width_min)/width_2')
c_inf_add_new['Area_CS'] = c_inf_geo['Area_CS']
c_inf_add_new['Volume'] = c_inf_geo['Volume']
c_inf_add_new['geo_MoI_mean'] = c_inf_geo.eval('(thickness_mean**3*width_mean)/12')
c_inf_add_new['geo_dMoI']    = c_inf_add.eval('(geo_MoI_max - geo_MoI_min)/geo_MoI_mid')
c_inf_add['geo_curve_mid_circ'] = curve_circmid
c_inf_add_new['geo_curve_mid_circ'] = curve_circmid
c_inf_add_new['geo_curve_max']  = c_inf_add[['geo_curve_max','geo_curve_mid','geo_curve_min']].abs().max(axis=1)
c_inf_add_new['geo_dcurve']  = c_inf_add.eval('(geo_curve_max - geo_curve_min)/geo_curve_mid')
c_inf_add_new['ind_R_max']  = c_inf_add[['ind_Ropt_l','ind_Ropt_r']].abs().max(axis=1)
c_inf_add_new['ind_R_mean'] = c_inf_add[['ind_Ropt_l','ind_Ropt_r']].mean(axis=1).abs()
c_inf_add_new['ind_U_max']  = c_inf_add[['ind_U_l','ind_U_r']].abs().max(axis=1)
c_inf_add_new['ind_U_mean'] = c_inf_add[['ind_U_l','ind_U_r']].mean(axis=1).abs()

# c_inf_gad = pd.concat([c_inf_geo,c_inf_add_new],axis=1)
c_inf_gad = c_inf_add_new

# c_inf_mat = cs.loc(axis=1)['fy':'Wb_opt']
c_inf_mat = cs.loc(axis=1)['Density_app','fy','ey_opt','Wy_opt',
                           'fu','eu_opt','Wu_opt','fb','eb_opt','Wb_opt']

# c_inf=pd.concat([c_inf_geo,c_inf_add_new,c_inf_mat],axis=1)
c_inf=pd.concat([c_inf_add_new,c_inf_mat],axis=1)

# Speicherung stat-data einfluss
t=c_inf.agg(agg_funcs)
t1=c_inf.agg(Evac.confidence_interval)
t1.index=['ci_min','ci_max']
t=pd.concat([t,t1])
t.to_csv(out_full+'_influence_param_stats.csv',sep=';') 

# zusatz log für Krümmung größer als 0.05 mm^-1 (12.11.2022)
Evac.MG_strlog("\n\n  - Curvature in middle greater or equal 0.05:",)
a = c_inf_add[['geo_curve_mid','geo_curve_mid_circ']] 
b = a[a.abs()>= 0.05]
Evac.MG_strlog(Evac.str_indent('Count:\n'+(b.count()).to_string(),3),
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent('Proportion:\n'+(b.count()/a.count()).to_string(),3),
               log_mg, printopt=False)
Evac.MG_strlog("\n\n "+"="*100, log_mg, printopt=False)


c_E_lsq_corr = pd.concat([c_inf,c_E_inc_m['R']],axis=1).corr(method=mcorr)
c_E_inc_corr = pd.concat([c_inf,c_E_inc_m['R']],axis=1).corr(method=mcorr)


fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, 
                              sharex=False, sharey=False, figsize = (6.3,2*3.54))
fig.suptitle('%s\nInfluence on least-square determined YM (only refined range)'%name_Head)
g=sns.heatmap(c_E_lsq_corr.loc[c_inf_gad.columns,c_E_lsq_m['R'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('Influence - geometrical and additional')
ax1.set_xlabel('Determination method')
ax1.set_ylabel('Influence')
ax1.tick_params(axis='x', labelrotation=90, labelsize=5)
ax1.tick_params(axis='y', labelsize=5)
g=sns.heatmap(c_E_lsq_corr.loc[c_inf_mat.columns,c_E_lsq_m['R'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax2)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax2.set_title('Influence - material')
ax2.set_xlabel('Determination method')
ax2.set_ylabel('Influence')
ax2.tick_params(axis='x', labelrotation=90, labelsize=5)
ax1.tick_params(axis='y', labelsize=5)
# fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Inf-E_lsq.pdf')
plt.savefig(out_full+'-Inf-E_lsq.png')
plt.show()

fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, 
                              sharex=False, sharey=False, figsize = (6.3,2*3.54))
fig.suptitle('%s\nInfluence on incremental determined YM (only refined range)'%name_Head)
g=sns.heatmap(c_E_inc_corr.loc[c_inf_gad.columns,c_E_inc_m['R'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('Influence - geometrical and additional')
ax1.set_xlabel('Determination method')
ax1.set_ylabel('Influence')
ax1.tick_params(axis='x', labelrotation=90, labelsize=5)
ax1.tick_params(axis='y', labelsize=5)
g=sns.heatmap(c_E_inc_corr.loc[c_inf_mat.columns,c_E_inc_m['R'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax2)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax2.set_title('Influence - material')
ax2.set_xlabel('Determination method')
ax2.set_ylabel('Influence')
ax2.tick_params(axis='x', labelrotation=90, labelsize=5)
ax1.tick_params(axis='y', labelsize=5)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Inf-E_inc.pdf')
plt.savefig(out_full+'-Inf-E_inc.png')
plt.show()


#%%% Auswertungsbereich E-Modul - neu
a=((c_E_inc['F']-c_E_inc['R'])/c_E_inc['R'])

b=a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                'meanwoso'].droplevel(1,axis=1)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=b, ax=ax1, 
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nDeviation of Youngs Modulus by determination range'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
ax1.set_ylabel('$(E_{fixed}-E_{refined})/E_{refined}$ / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Comp-EvaRange-DYM.pdf')
plt.savefig(out_full+'-Comp-EvaRange-DYM.png')
plt.show()

b=a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                'stdnwoso'].droplevel(1,axis=1)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=b, ax=ax1, 
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nDeviation of coefficient of variation of Youngs Modulus by determination range'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
# ax1.set_ylabel('$(CV_{er,E_{fixed}}-CV_{er,E_{refined}})/CV_{er,E_{refined}}$ / -')
ax1.set_ylabel('$(CV_{er,fixed}-CV_{er,refined})/CV_{er,refined}$ / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Comp-EvaRange-DCV.pdf')
plt.savefig(out_full+'-Comp-EvaRange-DCV.png')
plt.show()

#%%% Methodenvergleich E-Modul
df=pd.melt(c_E_inc_m['R'],
           value_vars=c_E_inc_m['R'].columns)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYoungs Modulus of incremental determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
# ax1.set_ylabel('$\overline{E}_{1.5*IQR}$ / MPa')
ax1.set_ylabel('$E$ / MPa')
ax1.set_ylim(0,5000)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YM-E_inc_R-box.pdf')
plt.savefig(out_full+'-YM-E_inc_R-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="value", data=df, ax=ax1, 
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nYoungs Modulus of incremental determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
ax1.set_ylabel('$E$ / MPa')
ax1.set_ylim(0,5000)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YM-E_inc_R-bar.pdf')
plt.savefig(out_full+'-YM-E_inc_R-bar.png')
plt.show()

df = c_E_inc_m['R']
df = df/df.mean(axis=0)
df=pd.melt(df,
           value_vars=df.columns)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYoungs Modulus per mean of method of incremental determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso/YM_method_mean / -')
ax1.set_ylabel('$E_{Specimen}/\overline{E}_{Method}$ / -')
ax1.set_ylim(0,3)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YMm-E_inc_R-box.pdf')
plt.savefig(out_full+'-YMm-E_inc_R-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="value", data=df, ax=ax1, 
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nYoungs Modulus per mean of method of incremental determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso/YM_method_mean / -')
ax1.set_ylabel('$E_{Specimen}/\overline{E}_{Method}$ / -')
ax1.set_ylim(0,3)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YMm-E_inc_R-bar.pdf')
plt.savefig(out_full+'-YMm-E_inc_R-bar.png')
plt.show()


df = c_E_inc_m['R']
df = (df/df.mean(axis=0)).std(axis=0)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(data=df, ax=ax1)
ax1.set_title('%s\nStandard Deviation of Youngs Modulus per mean of method of incremental determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$\sigma_{E_{Specimen}/\overline{E}_{Method}}$ / -')
ax1.set_ylim(0.80,1.00)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YMmsd-E_inc_R-bar.pdf')
plt.savefig(out_full+'-YMmsd-E_inc_R-bar.png')
plt.show()




df = pd.DataFrame([])
df['mean'] = c_E_inc_m['R'].mean()
id_dr = df.index.str.contains('fu')
df = df.loc[np.invert(id_dr)]
df['dif_opt'] = (df - df.loc[YM_opt[2]])/(df.loc[YM_opt[2]])
df['colors'] = ['red' if x < 0 else 'green' for x in df['dif_opt']]
df.sort_values('dif_opt', inplace=True)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.vlines(x=df.index, ymin=0, ymax=df['dif_opt'], color=df['colors'], alpha=0.4, linewidth=5)
ax1.set_title('%s\nIncremental determined YM compared to Method %s\n(only refined range)'%(name_Head,YM_opt[2]))
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('$\overline{E}_{1.5*IQR}/\overline{E}_{1.5*IQR,%s}$ / -'%YM_opt[2])
ax1.set_ylabel('($(E_{Method} - E_{%s})/E_{%s}$ / -'%(YM_opt[2],YM_opt[2]))
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YMm-E_inc_R-comp_chosen.pdf')
plt.savefig(out_full+'-YMm-E_inc_R-comp_chosen.png')
plt.show()




fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=c_E_lsq_m['R'], ax=ax1, 
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYoungs Modulus of least-square determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
ax1.set_ylabel('$E_{lsq}$ / MPa')
ax1.set_ylim(0,5000)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YM-E_lsq_R-box.pdf')
plt.savefig(out_full+'-YM-E_lsq_R-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=c_E_lsq_m['R'], ax=ax1, 
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nYoungs Modulus of least-square determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
ax1.set_ylabel('$E_{lsq}$ / MPa')
ax1.set_ylim(0,5000)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YM-E_lsq_R-bar.pdf')
plt.savefig(out_full+'-YM-E_lsq_R-bar.png')
plt.show()

#%%% Methodenvergleich stdnorm
df=pd.melt(c_E_inc_stnorm['F'],
           value_vars=c_E_inc_stnorm['F'].columns)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only fixed range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_SD/YM_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_F-box.pdf')
plt.savefig(out_full+'-StdN-E_inc_F-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="value", data=df, ax=ax1, 
                 errwidth=1, capsize=.1)
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only fixed range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_SD/YM_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_F-bar.pdf')
plt.savefig(out_full+'-StdN-E_inc_F-bar.png')
plt.show()



df=pd.melt(c_E_inc_stnorm['R'],
           value_vars=c_E_inc_stnorm['R'].columns)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_SD/YM_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_R-box.pdf')
plt.savefig(out_full+'-StdN-E_inc_R-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="value", data=df, ax=ax1, 
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_SD/YM_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_R-bar.pdf')
plt.savefig(out_full+'-StdN-E_inc_R-bar.png')
plt.show()


df = pd.DataFrame([])
id_dr = c_E_inc_stnorm.columns.droplevel(0).str.contains('fu')
df['mean'] = c_E_inc_stnorm.loc(axis=1)[:,idx[np.invert(id_dr)]].mean()['R']
df['dif_opt'] = (df - df.loc[YM_opt[2]])/(df.loc[YM_opt[2]])
df['colors'] = ['red' if x < 0 else 'green' for x in df['dif_opt']]
df.sort_values('dif_opt', inplace=True)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.vlines(x=df.index, ymin=0, ymax=df['dif_opt'], color=df['colors'], alpha=0.4, linewidth=5)
ax1.set_title('%s\nCoefficient of variation of methods compared to method %s\n(only refined range)'%(name_Head,YM_opt[2]))
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('$\overline{\sigma}_{1.5*IQR}/\overline{\sigma}_{1.5*IQR,%s}$ / -'%YM_opt[2])
# ax1.set_ylabel('($(\sigma_{E_{Method}} - \sigma_{E_{%s}})/\sigma_{E_{%s}}$ / -'%(YM_opt[2],YM_opt[2]))
ax1.set_ylabel('($(CV_{E_{Method}} - CV_{E_{%s}})/CV_{E_{%s}}$ / -'%(YM_opt[2],YM_opt[2]))
ax1.set_ylim(-0.5,4)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_R-comp_chosen.pdf')
plt.savefig(out_full+'-StdN-E_inc_R-comp_chosen.png')
plt.show()


# only M and global
id_Mg = c_E_inc_stnorm.columns.droplevel(0).str.contains('Mg')
id_dr = c_E_inc_stnorm.columns.droplevel(0).str.contains('fu')
c_E_inc_Mg_stnorm=c_E_inc_stnorm.loc(axis=1)[idx[:,:,id_Mg & np.invert(id_dr)]]
df = c_E_inc_Mg_stnorm
df = df.unstack(level=0).reset_index(level=2, drop=True)
# df.index.names=['Range','Method']
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="data", hue="Range", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only global-bending approach)'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_SD/E_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_Mg-box.pdf')
plt.savefig(out_full+'-StdN-E_inc_Mg-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue="Range", data=df, ax=ax1,
                 errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only global-bending approach)'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_SD/E_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
#ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_Mg-bar.pdf')
plt.savefig(out_full+'-StdN-E_inc_Mg-bar.png')
plt.show()

# only M and local
id_Ml = c_E_inc_stnorm.columns.droplevel(0).str.contains('Ml|Me')
id_dr = c_E_inc_stnorm.columns.droplevel(0).str.contains('fu')
c_E_inc_Ml_stnorm=c_E_inc_stnorm.loc(axis=1)[idx[:,:,id_Ml & np.invert(id_dr)]]
df = c_E_inc_Ml_stnorm
df = df.unstack(level=0).reset_index(level=2, drop=True)
# df.index.names=['Range','Method']
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="data", hue="Range", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only local-bending approach)'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_SD/E_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_Ml-box.pdf')
plt.savefig(out_full+'-StdN-E_inc_Ml-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue="Range", data=df, ax=ax1,
                 errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only local-bending approach)'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_SD/E_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
#ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_Ml-bar.pdf')
plt.savefig(out_full+'-StdN-E_inc_Ml-bar.png')
plt.show()


# only A and global
id_Ag = c_E_inc_stnorm.columns.droplevel(0).str.contains('Ag|Sg')
id_dr = c_E_inc_stnorm.columns.droplevel(0).str.contains('fu')
c_E_inc_Ag_stnorm=c_E_inc_stnorm.loc(axis=1)[idx[:,:,id_Ag & np.invert(id_dr)]]
df = c_E_inc_Ag_stnorm
df = df.unstack(level=0).reset_index(level=2, drop=True)
# df.index.names=['Range','Method']
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="data", hue="Range", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only global-complete/w.o. indentation approach)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_SD/E_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_ASg-box.pdf')
plt.savefig(out_full+'-StdN-E_inc_ASg-box.png')
plt.show()
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue="Range", data=df, ax=ax1,
                 errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only global-complete/w.o. indentation approach)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_SD/E_mean / -')
# ax1.set_ylabel('$\sigma_{E_{1.5*IQR}}/\overline{E}_{1.5*IQR}$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
# ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_ASg-bar.pdf')
plt.savefig(out_full+'-StdN-E_inc_ASg-bar.png')
plt.show()


# only A and local
id_Al = c_E_inc_stnorm.columns.droplevel(0).str.contains('Al|Sl')
id_dr = c_E_inc_stnorm.columns.droplevel(0).str.contains('fu')
c_E_inc_Al_stnorm=c_E_inc_stnorm.loc(axis=1)[idx[:,:,id_Al & np.invert(id_dr)]]
df = c_E_inc_Al_stnorm
df = df.unstack(level=0).reset_index(level=2, drop=True)
df.index.names=['Range','Method']
df = df.reset_index(name='data')

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="data", hue="Range", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only local-complete/w.o. indentation approach)'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_SD/E_mean / -')
# ax1.set_ylabel('$\sigma_{E}/E$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_ASl-box.pdf')
plt.savefig(out_full+'-StdN-E_inc_ASl-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue="Range", data=df, ax=ax1,
                 errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nCoefficient of variation of incremental determined YM \n(only local-complete/w.o. indentation approach)'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_SD/E_mean / -')
# ax1.set_ylabel('$\sigma_{E_{1.5*IQR}}/\overline{E}_{1.5*IQR}$ / -')
ax1.set_ylabel('$CV_{E}$ / -')
#ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_ASl-bar.pdf')
plt.savefig(out_full+'-StdN-E_inc_ASl-bar.png')
plt.show()

#%%% Vergleich least-square and incremental

df = Comp_E_lsqinc.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="data", hue='Range', data=df, ax=ax1, showfliers=False,
                  showmeans=False, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.boxplot(x="Method", y="data", hue='Range', data=df, ax=ax1,
#                   showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nYM comparison least-square to incremental determinition'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_lsq/E_inc / -')
ax1.set_ylabel('$(E_{lsq}-E_{inc})/E_{inc}$ / -')
# ax1.set_ylim(0.8,1.2)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-lsqinc-box.pdf')
plt.savefig(out_full+'-EComp-lsqinc-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue='Range', data=df, ax=ax1,
                 errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Determination range')
ax1.set_title('%s\nYM comparison least-square to incremental determinition'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_lsq/E_inc / -')
ax1.set_ylabel('$(E_{lsq}-E_{inc})/E_{inc}$ / -')
# ax1.set_ylim(0,5)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-lsqinc-bar.pdf')
plt.savefig(out_full+'-EComp-lsqinc-bar.png')
plt.show()

#%%% Vergleich M und A/S

id_dr = Comp_E_inc_m_SA.columns.str.contains('fu')
df = Comp_E_inc_m_SA.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="index", y="data", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYM S/A incremental determined YM'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_M/E_A / -')
ax1.set_ylabel('$(E_{S}-E_{A})/E_{A}$ / -')
ax1.set_ylim(-1,4)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_SA-box.pdf')
plt.savefig(out_full+'-EComp-E_inc_SA-box.png')
plt.show()
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="index", y="data", data=df, ax=ax1,
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nYM S/A incremental determined YM'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_M/E_A / -')
ax1.set_ylabel('$(E_{S}-E_{A})/E_{A}$ / -')
ax1.set_ylim(-1,4)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_SA-bar.pdf')
plt.savefig(out_full+'-EComp-E_inc_SA-bar.png')
plt.show()


id_dr = Comp_E_inc_m_MA.columns.str.contains('fu')
df = Comp_E_inc_m_MA.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="index", y="data", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYM M/A incremental determined YM'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_M/E_A / -')
ax1.set_ylabel('$(E_{M}-E_{A})/E_{A}$ / -')
ax1.set_ylim(-1,4)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_MA-box.pdf')
plt.savefig(out_full+'-EComp-E_inc_MA-box.png')
plt.show()
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="index", y="data", data=df, ax=ax1,
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nYM M/A incremental determined YM'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.set_ylabel('$(E_{M}-E_{A})/E_{A}$ / -')
ax1.set_ylim(-1,4)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_MA-bar.pdf')
plt.savefig(out_full+'-EComp-E_inc_MA-bar.png')
plt.show()

id_dr = Comp_E_inc_m_MS.columns.str.contains('fu')
df = Comp_E_inc_m_MS.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="index", y="data", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYM M/S incremental determined YM'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_M/E_A / -')
ax1.set_ylabel('$(E_{M}-E_{S})/E_{S}$ / -')
ax1.set_ylim(-0.3,0.5)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_MS-box.pdf')
plt.savefig(out_full+'-EComp-E_inc_MS-box.png')
plt.show()
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="index", y="data", data=df, ax=ax1,
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nYM M/S incremental determined YM'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.set_ylabel('$(E_{M}-E_{S})/E_{S}$ / -')
ax1.set_ylim(-0.01,0.5)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_MS-bar.pdf')
plt.savefig(out_full+'-EComp-E_inc_MS-bar.png')
plt.show()

id_dr = Comp_E_inc_m_CM.columns.str.contains('fu')
df = Comp_E_inc_m_CM.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="index", y="data", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                             "markeredgecolor":"black", "markersize":"25","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYM C/M incremental determined YM'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_M/E_A / -')
ax1.set_ylabel('$(E_{C}-E_{M})/E_{M}$ / -')
# ax1.set_ylim(0.8,1.5)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_CM-box.pdf')
plt.savefig(out_full+'-EComp-E_inc_CM-box.png')
plt.show()
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="index", y="data", data=df, ax=ax1,
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nYM C/M incremental determined YM'%name_Head)
ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_M/E_A / -')
ax1.set_ylabel('$(E_{C}-E_{M})/E_{M}$ / -')
# ax1.set_ylim(0.8,1.5)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_CM-bar.pdf')
plt.savefig(out_full+'-EComp-E_inc_CM-bar.png')
plt.show()

# df=Comp_E_inc_m_IE.unstack()
# df=df.reset_index(name='data')
# fig, ax1 = plt.subplots()
# ax1.grid(True)
# ax = sns.barplot(x="Method", y="data", hue="Inf", data=df, ax=ax1,
#                  errwidth=1, capsize=.1)
# ax1.set_title('%s\nDeviation of influence-elimination'%name_Head)
# ax1.set_xlabel('Determination method / -')
# ax1.set_ylabel('$D_{E}$ / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylim(-1,2)
# fig.suptitle('')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

# df=Comp_E_inc_m_IE.agg(Evac.coefficient_of_variation).abs().reset_index(name='data')
# fig, ax1 = plt.subplots()
# ax1.grid(True)
# ax = sns.barplot(x="Method", y="data", hue="Inf", data=df, ax=ax1)
# ax1.set_title('%s\nPopulational coefficient of variation of influence-elimination'%name_Head)
# ax1.set_xlabel('Determination method / -')
# ax1.set_ylabel('$|CV_{pop}|$ / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylim(-.1,5)
# fig.suptitle('')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()



#%%%% Einfluss
# c_inf=pd.concat([c_inf_geo,c_inf_add_new],axis=1)

Comp_E_inc_m_MA_corr = pd.concat([c_inf,Comp_E_inc_m_MA],axis=1).corr(method=mcorr)
Comp_E_inc_m_MS_corr = pd.concat([c_inf,Comp_E_inc_m_MS],axis=1).corr(method=mcorr)
Comp_E_inc_m_SA_corr = pd.concat([c_inf,Comp_E_inc_m_SA],axis=1).corr(method=mcorr)

fig, ax1 = plt.subplots()
g=sns.heatmap(Comp_E_inc_m_MA_corr.loc[c_inf.columns,Comp_E_inc_m_MA.columns].round(1),
            center=0,annot=True, annot_kws={"size":5, 'rotation':0},ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('%s\nInfluence on incremental determined YM M/A \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method')
ax1.set_ylabel('Influence')
ax1.tick_params(axis='x', labelrotation=90)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-ECompInf-E_inc_MA.pdf')
plt.savefig(out_full+'-ECompInf-E_inc_MA.png')
plt.show()

fig, ax1 = plt.subplots()
g=sns.heatmap(Comp_E_inc_m_MS_corr.loc[c_inf.columns,Comp_E_inc_m_MS.columns].round(1),
            center=0,annot=True, annot_kws={"size":5, 'rotation':0},ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('%s\nInfluence on incremental determined YM M/S \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method')
ax1.set_ylabel('Influence')
ax1.tick_params(axis='x', labelrotation=90)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-ECompInf-E_inc_MS.pdf')
plt.savefig(out_full+'-ECompInf-E_inc_MS.png')
plt.show()

fig, ax1 = plt.subplots()
g=sns.heatmap(Comp_E_inc_m_SA_corr.loc[c_inf.columns,Comp_E_inc_m_SA.columns].round(1),
            center=0,annot=True, annot_kws={"size":5, 'rotation':0},ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('%s\nInfluence on incremental determined YM S/A \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method')
ax1.set_ylabel('Influence')
ax1.tick_params(axis='x', labelrotation=90)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-ECompInf-E_inc_SA.pdf')
plt.savefig(out_full+'-ECompInf-E_inc_SA.png')
plt.show()



#Kombiniert für VÖ
fig, ax = plt.subplots(ncols=2, nrows=2,
                       gridspec_kw={'width_ratios': [24, 1],
                                    'height_ratios': [3, 1]})
g=sns.heatmap(Comp_E_inc_m_SA_corr.loc[c_inf.columns,Comp_E_inc_m_SA.columns.difference(Methods_excl_names)].round(1),
            center=0,annot=True, annot_kws={"size":5, 'rotation':0},
            ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0,0].set_ylabel('Influence')
ax[0,0].set_xticklabels([])
ax[0,1].tick_params(labelsize=5)
id_dr = Comp_E_inc_m_SA.columns.str.contains('fu')
df = Comp_E_inc_m_SA.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
ax[1,0].grid(True)
h = sns.barplot(x="index", y="data", data=df, ax=ax[1,0],
                 errwidth=1, capsize=.1)
# ax[1,0].set_ylabel('$(E_{S}-E_{A})/E_{A}$ / -')
ax[1,0].set_ylabel('$D_{S,A}$ / -')
ax[1,0].set_xlabel('Determination method')
ax[1,0].tick_params(axis='x', labelrotation=22.5)
fig.suptitle('Influence and deviation of S to A')
fig.delaxes(ax[1,1])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-ECompInf-E_inc_SA-bar.pdf')
plt.savefig(out_full+'-ECompInf-E_inc_SA-bar.png')
plt.show()

fig, ax = plt.subplots(ncols=2, nrows=2,
                       gridspec_kw={'width_ratios': [24, 1],
                                    'height_ratios': [3, 1]})
g=sns.heatmap(Comp_E_inc_m_MS_corr.loc[c_inf.columns,Comp_E_inc_m_MS.columns.difference(Methods_excl_names)].round(1),
            center=0,annot=True, annot_kws={"size":5, 'rotation':0},
            ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0,0].set_ylabel('Influence')
ax[0,0].set_xticklabels([])
ax[0,1].tick_params(labelsize=5)
id_dr = Comp_E_inc_m_MS.columns.str.contains('fu')
df = Comp_E_inc_m_MS.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
ax[1,0].grid(True)
h = sns.barplot(x="index", y="data", data=df, ax=ax[1,0],
                 errwidth=1, capsize=.1)
# ax[1,0].set_ylabel('$(E_{S}-E_{A})/E_{A}$ / -')
ax[1,0].set_ylabel('$D_{M,S}$ / -')
ax[1,0].set_xlabel('Determination method')
ax[1,0].tick_params(axis='x', labelrotation=22.5)
ax[1,0].set_ylim(-0.01,0.11)
fig.suptitle('Influence and deviation of M to S')
fig.delaxes(ax[1,1])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-ECompInf-E_inc_MS-bar.pdf')
plt.savefig(out_full+'-ECompInf-E_inc_MS-bar.png')
plt.show()



#%%% Vergleich zu konventioneller Methode

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=Comp_E_con.loc(axis=1)[np.invert(Comp_E_con.columns.str.contains('fu'))],
                 ax=ax1, showfliers=False,
                  showmeans=False, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYoungs Modulus of methods compared to conventional determination \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$(E_{Method} - E_{%s})/E_{%s}$ / -'%(YM_con[2],YM_con[2]))
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_con-box.pdf')
plt.savefig(out_full+'-EComp-E_inc_con-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=Comp_E_con.loc(axis=1)[np.invert(Comp_E_con.columns.str.contains('fu'))],
                 ax=ax1, errwidth=1, capsize=.1)
ax1.set_title('%s\nYoungs Modulus of methods compared to conventional determination \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$(E_{Method} - E_{%s})/E_{%s}$ / -'%(YM_con[2],YM_con[2]))
ax1.set_ylim(-0.01,2)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_con-bar.pdf')
plt.savefig(out_full+'-EComp-E_inc_con-bar.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=Comp_E_con.loc(axis=1)[Comp_E_con.columns.str.contains('C2|D1|D2')],
                 ax=ax1, errwidth=1, capsize=.1)
ax1.set_title('%s\nYoungs Modulus of methods compared to conventional determination \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$(E_{Method} - E_{%s})/E_{%s}$ / -'%(YM_con[2],YM_con[2]))
ax1.set_ylim(-0.01,2)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-E_inc_con-bar-CD.pdf')
plt.savefig(out_full+'-EComp-E_inc_con-bar-CD.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=Comp_CV_con.loc(axis=1)[np.invert(Comp_CV_con.columns.str.contains('fu'))],
                 ax=ax1, showfliers=False,
                  showmeans=False, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nCoefficient of variation over elastic range of methods\ncompared to conventional determination (only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$(CV_{Method} - CV_{%s})/CV_{%s}$ / -'%(YM_con[2],YM_con[2]))
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-CVComp-E_inc_con-box.pdf')
plt.savefig(out_full+'-CVComp-E_inc_con-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=Comp_CV_con.loc(axis=1)[np.invert(Comp_CV_con.columns.str.contains('fu'))],
                 ax=ax1, errwidth=1, capsize=.1)
ax1.set_title('%s\nCoefficient of variation over elastic range of methods\ncompared to conventional determination (only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$(CV_{Method} - CV_{%s})/CV_{%s}$ / -'%(YM_con[2],YM_con[2]))
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-CVComp-E_inc_con-bar.pdf')
plt.savefig(out_full+'-CVComp-E_inc_con-bar.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=Comp_CV_con.loc(axis=1)[Comp_E_con.columns.str.contains('C2|D1|D2')],
                 ax=ax1, errwidth=1, capsize=.1)
ax1.set_title('%s\nCoefficient of variation over elastic range of methods\ncompared to conventional determination (only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$(CV_{Method} - CV_{%s})/CV_{%s}$ / -'%(YM_con[2],YM_con[2]))
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-CVComp-E_inc_con-bar-CD.pdf')
plt.savefig(out_full+'-CVComp-E_inc_con-bar-CD.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=Comp_CV_con.loc(axis=1)[np.invert(Comp_CV_con.columns.str.contains('fu'))&(Comp_CV_con.columns.str.contains('M'))],
                 ax=ax1, errwidth=1, capsize=.1)
ax1.set_title('%s\nCoefficient of variation over elastic range of methods\ncompared to conventional determination (only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$(CV_{Method} - CV_{%s})/CV_{%s}$ / -'%(YM_con[2],YM_con[2]))
plt.savefig(out_full+'-CVComp-E_inc_con-bar-M.pdf')
plt.savefig(out_full+'-CVComp-E_inc_con-bar-M.png')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%%%% Einfluss
# c_inf=pd.concat([c_inf_geo,c_inf_add_new],axis=1)

Comp_E_con_corr = pd.concat([c_inf,Comp_E_con],axis=1).corr(method=mcorr)

fig, ax1 = plt.subplots()
g=sns.heatmap(Comp_E_con_corr.loc[c_inf.columns,Comp_E_con.columns].round(1),
            center=0, annot=True, annot_kws={"size":5, 'rotation':90},
            xticklabels=1, ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('%s\nInfluence on the deviation of the Youngs Modulus\nin relation to the conventional determination (only refined range)'%name_Head)
ax1.set_xlabel('Determination method')
ax1.set_ylabel('Influence')
ax1.tick_params(axis='x', labelrotation=90, labelsize=5)
ax1.tick_params(axis='y', labelsize=5)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-ECompInf-E_inc_con.pdf')
plt.savefig(out_full+'-ECompInf-E_inc_con.png')
plt.show()



#%%% Vergleich zu the analytischer Biegelinie (Methode G)

df = c_C.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
df = df.loc[np.invert(df.Method.str.contains('fu'))]

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="data", hue='Location', data=df,
                 ax=ax1, showfliers=False,
                 showmeans=False, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
Evac.tick_legend_renamer(ax=ax,renamer={'g':'global','l':'midspan'},title='')
ax1.set_title('%s\nComparison of scaled analytical deformation by methods to measured'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('$\overline{w_{analytical}/w_{measured}}$ / -')
ax1.set_ylabel('$(w_{analytical}-w_{measured})/w_{measured}$ / -')
# ax1.set_ylim(-1.05,1.05)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-wdevana-box.pdf')
plt.savefig(out_full+'-EComp-wdevana-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Method", y="data", hue='Location', data=df[df.Method.str.contains('M')],
                 ax=ax1, showfliers=True, flierprops={"markersize":"2"},
                 showmeans=False, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
Evac.tick_legend_renamer(ax=ax,renamer={'g':'global','l':'midspan'},title='')
ax1.set_title('%s\nComparison of scaled analytical deformation by methods to measured'%name_Head)
ax1.set_xlabel('Determination method (only M) / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('$\overline{w_{analytical}/w_{measured}}$ / -')
ax1.set_ylabel('$(w_{analytical}-w_{measured})/w_{measured}$ / -')
# ax1.set_ylim(-1.05,1.05)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-wdevana_M-box.pdf')
plt.savefig(out_full+'-EComp-wdevana_M-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue='Location', data=df, ax=ax1,
                 errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'g':'global','l':'midspan'},title='')
ax1.set_title('%s\nComparison of scaled analytical deformation by methods to measured'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('$\overline{w_{analytical}/w_{measured}}$ / -')
ax1.set_ylabel('$(w_{analytical}-w_{measured})/w_{measured}$ / -')
# ax1.set_ylim(0,5)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-wdevana-bar.pdf')
plt.savefig(out_full+'-EComp-wdevana-bar.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue='Location', data=df[df.Method.str.contains('M')],
                 ax=ax1, errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'g':'global','l':'midspan'},title='')
ax1.set_title('%s\nComparison of scaled analytical deformation by methods to measured'%name_Head)
ax1.set_xlabel('Determination method (only M) / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('$\overline{w_{analytical}/w_{measured}}$ / -')
ax1.set_ylabel('$(w_{analytical}-w_{measured})/w_{measured}$ / -')
#ax1.set_ylim(-0.25,1.05)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-wdevana_M-bar.pdf')
plt.savefig(out_full+'-EComp-wdevana_M-bar.png')
plt.show()



df = c_C_MC.sort_index(axis=1,level=1).unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
df = df.loc[np.invert(df.Method.str.contains('fu'))]
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue='Location', data=df,
                 ax=ax1, errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'g':'global','l':'midspan'},title='')
ax1.set_title('%s\nComparison of scaled analytical deformation by methods to measured'%name_Head)
ax1.set_xlabel('Determination method (only pure bending (M) and correct shear consideration (C)) / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('$\overline{w_{analytical}/w_{measured}}$ / -')
ax1.set_ylabel('$(w_{analytical}-w_{measured})/w_{measured}$ / -')
#ax1.set_ylim(-0.25,1.05)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-EComp-wdevana_MC-bar.pdf')
plt.savefig(out_full+'-EComp-wdevana_MC-bar.png')
plt.show()



#%% Plot Paper

a=((c_E_inc['F']-c_E_inc['R'])/c_E_inc['R'])
b=a.loc(axis=1)[a.droplevel(1,axis=1).columns.difference(Methods_excl_names),
                'meanwoso'].droplevel(1,axis=1)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(data=b, ax=ax1, 
                 errwidth=1, capsize=.1)
# ax1.set_title('%s\nDeviation of Youngs Modulus by determination range'%name_Head)
ax1.set_xlabel('Evaluation method')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
ax1.set_ylabel('$D_{E,fixed-refined}=(E_{fixed}-E_{refined})/E_{refined}$')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_paper+'Fig04-Comp-EvaRange-DYM.pdf')
plt.savefig(out_paper+'Fig04-Comp-EvaRange-DYM.png')
plt.savefig(out_paper+'Fig04-Comp-EvaRange-DYM.eps')
plt.show()

df = Comp_E_lsqinc.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue='Range', data=df, ax=ax1,
                 errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'F':'fixed','R':'refined'},
                         title='Evaluation range')
# ax1.set_title('%s\nYM comparison least-square to incremental determinition'%name_Head)
ax1.set_xlabel('Evaluation method')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('E_lsq/E_inc / -')
ax1.set_ylabel('$D_{E,lsq-inc}=(E_{lsq}-E_{inc})/E_{inc}$')
# ax1.set_ylim(0,5)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_paper+'Fig05-EComp-lsqinc-bar.pdf')
plt.savefig(out_paper+'Fig05-EComp-lsqinc-bar.png')
plt.savefig(out_paper+'Fig05-EComp-lsqinc-bar.eps')
plt.show()

fig, ax = plt.subplots(ncols=2, nrows=2,
                       gridspec_kw={'width_ratios': [24, 1],
                                    'height_ratios': [1.5, 1]})
plt_cinf_chosen = {'thickness_mean':   "average\nbeam's height",
                    'width_mean':      "average\nbeam's width",
                    'Density_app':     "apperent density",
                    'ind_R_mean':      "average\nsupport indentation\nat elastic limit",
                    'ind_U_mean':      "average\nsupport indentation\nat peak load"}
g=sns.heatmap(Comp_E_inc_m_SA_corr.loc[plt_cinf_chosen.keys(),
                                       Comp_E_inc_m_SA.columns.difference(Methods_excl_names)].round(1),
            center=0,annot=True, annot_kws={"size":5, 'rotation':0},
            ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='y')
ax[0,0].set_ylabel('Contributing factors')
ax[0,0].set_xticklabels([])
ax[0,1].tick_params(labelsize=5)
id_dr = Comp_E_inc_m_SA.columns.str.contains('fu')
df = Comp_E_inc_m_SA.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
ax[1,0].grid(True)
h = sns.barplot(x="index", y="data", data=df, ax=ax[1,0],
                 errwidth=1, capsize=.1)
Evac.tick_label_inserter(ax=h, pos=2, ins='*', axis='x')
ax[1,0].set_ylabel('$D_{E,S-A}$')
ax[1,0].set_xlabel('Evaluation method')
ax[1,0].tick_params(axis='x', labelrotation=22.5)
fig.suptitle('')
fig.delaxes(ax[1,1])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_paper+'Fig06-ECompInf-E_inc_SA-bar.pdf')
plt.savefig(out_paper+'Fig06-ECompInf-E_inc_SA-bar.png')
plt.savefig(out_paper+'Fig06-ECompInf-E_inc_SA-bar.eps')
plt.show()


fig, ax = plt.subplots(ncols=2, nrows=2,
                       gridspec_kw={'width_ratios': [24, 1],
                                    'height_ratios': [1.5, 1]})
plt_cinf_chosen = {'thickness_mean':   "average\nbeam's height",
                    'width_mean':      "average\nbeam's width",
                    'Density_app':     "apperent density",
                    'ind_R_mean':      "average\nsupport indentation\nat elastic limit",
                    'ind_U_mean':      "average\nsupport indentation\nat peak load"}
g=sns.heatmap(Comp_E_inc_m_MS_corr.loc[plt_cinf_chosen.keys(),
                                       Comp_E_inc_m_MS.columns.difference(Methods_excl_names)].round(1),
            center=0,annot=True, annot_kws={"size":5, 'rotation':0},
            ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='y')
ax[0,0].set_ylabel('Contributing factors')
ax[0,0].set_xticklabels([])
ax[0,1].tick_params(labelsize=5)
id_dr = Comp_E_inc_m_MS.columns.str.contains('fu')
df = Comp_E_inc_m_MS.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
ax[1,0].grid(True)
h = sns.barplot(x="index", y="data", data=df, ax=ax[1,0],
                 errwidth=1, capsize=.1)
Evac.tick_label_inserter(ax=h, pos=2, ins='*', axis='x')
ax[1,0].set_ylabel('$D_{E,M-S}$')
ax[1,0].set_xlabel('Evaluation method')
ax[1,0].tick_params(axis='x', labelrotation=22.5)
fig.suptitle('')
tmp=ax[1,0].get_xlim()
ax[1,0].hlines([0.11,0.11,0.11, 0.11, 0.11],
              [-0.4, 0.6, 1.6, 11.6, 12.6],
              [ 0.4, 1.4, 2.4, 12.4, 13.4],
              colors='red', linestyles='solid', linewidth=4.0)
ax[1,0].set_xlim(tmp)
ax[1,0].set_ylim(-0.01,0.11)
t=Comp_E_inc_m_MS.loc(axis=1)[['A2l','A4l','B1l','F4gha','F4gth']].agg(['mean',Evac.confidence_interval])
ftxt=['Trimmed values:',
      '  A2*l: %5.2f (%4.2f-%4.2f)  F4*gha: %5.2f (%4.2f-%4.2f)'%(t.loc['mean','A2l'],
                                        *t.loc['confidence_interval','A2l'],
                                        t.loc['mean','F4gha'],
                                        *t.loc['confidence_interval','F4gha']),
      '  A4*l: %5.2f (%4.2f-%4.2f)  F4*gth: %5.2f (%4.2f-%4.2f)'%(t.loc['mean','A4l'],
                                        *t.loc['confidence_interval','A4l'],
                                        t.loc['mean','F4gth'],
                                        *t.loc['confidence_interval','F4gth']),
      '  B1*l: %5.2f (%4.2f-%4.2f)'%(t.loc['mean','B1l'],
                                      *t.loc['confidence_interval','B1l'])]
ax[1,0].text(2.6,0.065,'\n'.join(ftxt),ha='left',va='bottom', fontsize=5,
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
fig.delaxes(ax[1,1])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_paper+'Fig07-ECompInf-E_inc_MS-bar.pdf')
plt.savefig(out_paper+'Fig07-ECompInf-E_inc_MS-bar.png')
plt.savefig(out_paper+'Fig07-ECompInf-E_inc_MS-bar.eps')
plt.show()


df = c_C_MC.sort_index(axis=1,level=1).unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
df = df.loc[np.invert(df.Method.str.contains('fu'))]
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Method", y="data", hue='Location', data=df,
                 ax=ax1, errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax,renamer={'g':'global','l':'midspan'},title='')
# ax1.set_title('%s\nComparison of scaled analytical deformation by methods to measured'%name_Head)
ax1.set_xlabel('Evaluation method')
ax1.tick_params(axis='x', labelrotation=22.5)
ax1.set_ylabel('$D_{w}=(w_{analytical}-w_{measured})/w_{measured}$')
#ax1.set_ylim(-0.25,1.05)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_paper+'Fig08-EComp-wdevana_MC-bar.pdf')
plt.savefig(out_paper+'Fig08-EComp-wdevana_MC-bar.png')
plt.savefig(out_paper+'Fig08-EComp-wdevana_MC-bar.eps')
plt.show()



#%% Plot Supplementary material
lss = [':', '--', '-.','-']
figsize_sup=(16.0/2.54, 22.0/2.54)

fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=figsize_sup, constrained_layout=True)
axt = sns.histplot(c_inf[['thickness_mean','width_mean']],
                   stat='count', bins=20, ax=ax[0], kde=True, legend=True)
Evac.tick_legend_renamer(ax[0],VIPar_plt_renamer)
# handles = ax[0].legend_.legendHandles[::-1]
# for line, ls, handle in zip(ax[0].lines, lss, handles):
#     line.set_linestyle(ls)
#     handle.set_ls(ls)
# legend_elements=[Line2D([0], [0], color=sns.color_palette("tab10")[0],
#                         ls=lss[0], label=r'$\overline{h}$'),
#                  Line2D([0], [0], color=sns.color_palette("tab10")[1],
#                         ls=lss[1], label=r'$\overline{b}$')]
# ax[0].legend(handles=legend_elements, loc='best')
ax[0].set_title('Mean thickness and width',fontsize=10)
ax[0].set_xlabel('Dimension / mm')
ax[0].set_ylabel('Count')

axt = sns.histplot(c_inf[['geo_dthick','geo_dwidth']],
                   stat='count', bins=20, ax=ax[1], kde=True, legend=True)
Evac.tick_legend_renamer(ax[1],VIPar_plt_renamer)
ax[1].set_title('Deviation of thickness and width along span',fontsize=10)
ax[1].set_xlabel('Deviation / -')
ax[1].set_ylabel('Count')

# tmp=pd.concat([c_inf_add['geo_curve_mid'].abs(),c_inf_add_new['geo_curve_max']],axis=1)
# tmp=(1.0/tmp.mul(c_inf_geo['thickness_mean'],axis=0))
# axt = sns.histplot(tmp[['geo_curve_mid','geo_curve_max']],
#                    stat='count', bins=20, ax=ax[2], kde=True, legend=True)
# Evac.tick_legend_renamer(ax[2],{'geo_curve_mid':'$|R_{mid}|/\overline{h}$',
#                                 'geo_curve_max':'$|R|_{min}/\overline{h}$'})
# tmp=c_inf_add[['geo_curve_mid','geo_curve_max','geo_curve_min']]
# tmp=tmp.mul(c_inf_geo['thickness_mean'],axis=0)
# axt = sns.histplot(tmp[['geo_curve_mid','geo_curve_max','geo_curve_min']],
#                    stat='count', bins=20, ax=ax[2], kde=True, legend=True)
# Evac.tick_legend_renamer(ax[2],{'geo_curve_mid':'$\kappa_{mid}x\overline{h}$',
#                                 'geo_curve_max':'$\kappa_{max}x\overline{h}$',
#                                 'geo_curve_min':'$\kappa_{min}x\overline{h}$'})
# axt = sns.histplot(c_inf_add[['geo_curve_mid','geo_curve_max','geo_curve_min']],
#                    stat='count', bins=20, ax=ax[2], kde=True, legend=True)
# Evac.tick_legend_renamer(ax[2],{'geo_curve_mid':'$\kappa_{mid}$',
#                                 'geo_curve_max':'$\kappa_{max}$',
#                                 'geo_curve_min':'$\kappa_{min}$'})
# tmp=pd.Series(c_inf['geo_dcurve']*c_inf_add['geo_curve_mid'],name='delta')
# axt = sns.histplot(pd.concat([c_inf_add_new['geo_curve_mid_circ'],
#                               c_inf_add['geo_curve_mid'],tmp],axis=1),
#                     stat='count', bins=20, ax=ax[2], kde=True, legend=True)
# Evac.tick_legend_renamer(ax[2],{'geo_curve_mid_circ':'$\kappa_{mid,circle}$',
#                                 'geo_curve_mid':'$\kappa_{mid}$',
#                                 'delta':'$\Delta \kappa_{span}$'})
axt = sns.histplot(pd.concat([c_inf_add_new['geo_curve_mid_circ'],
                              c_inf_add['geo_curve_mid']],axis=1),
                    stat='count', bins=20, ax=ax[2], kde=True, legend=True)
Evac.tick_legend_renamer(ax[2],{'geo_curve_mid_circ':'$\kappa_{mid,circle}$',
                                'geo_curve_mid':'$\kappa_{mid}$'})
# ax[2].set_title('Curvature radius per mean thickness of unloaded specimen',fontsize=10)
# ax[2].set_title('Curvature multiplied with mean thickness of unloaded specimen',fontsize=10)
ax[2].set_title('Curvature of unloaded specimen',fontsize=10)
# ax[2].set_xlabel('Curvature radius per thickness / mm/mm')
# ax[2].set_xlabel('Curvature x thickness / mm/mm')
ax[2].set_xlabel('Curvature / mm$^{-1}$')
ax[2].set_ylabel('Count')

# fig.suptitle('Distribution of geometrical observations',
#              fontsize=12, fontweight="bold")
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'01.png')
plt.savefig(out_supmat+'01.pdf')
plt.show()
plt.close(fig)


fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=figsize_sup, constrained_layout=True)
ax[0].set_title('Curvature of unloaded specimen',fontsize=10)
axt = sns.histplot(c_inf_add[['geo_curve_mid_circ','geo_curve_mid',
                              'geo_curve_max','geo_curve_min']],
                   stat='count', bins=20, ax=ax[0], kde=True, legend=True)
Evac.tick_legend_renamer(ax[0],{'geo_curve_mid_circ':'$\kappa_{mid,circle}$',
                                'geo_curve_mid':'$\kappa_{mid}$',
                                'geo_curve_max':'$\kappa_{max}$',
                                'geo_curve_min':'$\kappa_{min}$'})
ax[0].set_xlabel('Curvature / mm$^{-1}$')
ax[0].set_ylabel('Count')

tmp=c_inf_add[['geo_curve_mid_circ','geo_curve_mid','geo_curve_max','geo_curve_min']]
tmp=tmp.mul(c_inf_geo['thickness_mean'],axis=0)
axt = sns.histplot(tmp[['geo_curve_mid_circ','geo_curve_mid',
                        'geo_curve_max','geo_curve_min']],
                   stat='count', bins=20, ax=ax[1], kde=True, legend=True)
Evac.tick_legend_renamer(ax[1],{'geo_curve_mid_circ':'$\kappa_{mid,circle}x\overline{h}$',
                                'geo_curve_mid':'$\kappa_{mid}x\overline{h}$',
                                'geo_curve_max':'$\kappa_{max}x\overline{h}$',
                                'geo_curve_min':'$\kappa_{min}x\overline{h}$'})
ax[1].set_title('Curvature multiplied with mean thickness of unloaded specimen',fontsize=10)
ax[1].set_xlabel('Curvature x thickness / mm/mm')
ax[1].set_ylabel('Count')

# ax[2].set_title('Curvature radius per mean thickness of unloaded specimen',fontsize=10)
# tmp=pd.concat([c_inf_add['geo_curve_mid'].abs(),c_inf_add_new['geo_curve_max']],axis=1)
# tmp=(1.0/tmp.mul(c_inf_geo['thickness_mean'],axis=0))
# axt = sns.histplot(tmp[['geo_curve_mid','geo_curve_max']],
#                    stat='count', bins=20, binrange=[0,10], ax=ax[2], kde=False, legend=True)
# Evac.tick_legend_renamer(ax[2],{'geo_curve_mid':'$|R_{mid}|/\overline{h}$',
#                                 'geo_curve_max':'$|R|_{min}/\overline{h}$'})
# ax[2].set_xlabel('Curvature radius per thickness / mm/mm')
# ax[2].set_ylabel('Count')
# ax[2].set_xlim([0,10])
ax[2].set_title('Curvature radius per mean thickness of unloaded specimen\n(exclusion of statistical outliers)',fontsize=10)
# tmp=pd.concat([c_inf_add['geo_curve_mid'].abs(),c_inf_add_new['geo_curve_max']],axis=1)
# tmp=(1.0/tmp.mul(c_inf_geo['thickness_mean'],axis=0))
tmp=1.0/c_inf_add['geo_curve_mid'].abs().mul(c_inf_add['thickness_2'],axis=0)
tmp=pd.concat([tmp,1.0/c_inf_add_new[['geo_curve_mid_circ','geo_curve_max']].abs().mul(c_inf_geo['thickness_mean'],axis=0)],axis=1)
tmp.columns=['geo_curve_mid','geo_curve_mid_circ','geo_curve_max']
axt = sns.boxplot(data=tmp[['geo_curve_mid_circ','geo_curve_mid','geo_curve_max']],
                  ax=ax[2], orient='h', showfliers=False, showmeans=False, 
                  meanprops={"marker":"|", "markerfacecolor":"white",
                             "markeredgecolor":"black", "markersize":"25","alpha":0.75})
ftxt=['Mimimal values:',
      '   $|R_{circle}|/\overline{h}$: %4.2f'%(tmp.geo_curve_mid_circ.min()),
      '   $|R_{mid}|/h_{mid}$: %4.2f'%(tmp.geo_curve_mid.min()),
      '   $|R|_{min}/\overline{h}$: %4.2f'%(tmp.geo_curve_max.min())]
ax[2].text(225,2.0,'\n'.join(ftxt),ha='left',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
Evac.tick_label_renamer(ax[2],{'geo_curve_mid_circ':'$|R|_{circle}/\overline{h}$',
                               'geo_curve_mid':'$|R_{mid}|/h_{mid}$',
                               'geo_curve_max':'$|R|_{min}/\overline{h}$'}, axis='y')
ax[2].set_yticklabels(ax[2].get_yticklabels(), rotation=90, va="center")
ax[2].set_xlabel('Curvature radius per thickness / mm/mm')
ax[2].set_ylabel('')
# fig.suptitle('Distribution of geometrical curvature observations',
#              fontsize=12, fontweight="bold")
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'ADD_Curve.png')
plt.savefig(out_supmat+'ADD_Curve.pdf')
plt.show()
plt.close(fig)




fig, ax = plt.subplots(nrows=3, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1],
                                    'height_ratios': [15, 10, 10]},
                       figsize=figsize_sup, constrained_layout=True)
# fig.suptitle('Influence on incremental, in refined range,\ndetermined Youngs Modulus',
#              fontsize=12, fontweight="bold")
g=sns.heatmap(c_E_inc_corr.loc[c_inf_gad.columns,c_E_inc_m['R'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0,0].set_title('Spearman correlation coefficients for geometrical and additional factors')
ax[0,0].set_xlabel('Evaluation method')
ax[0,0].set_ylabel('Contributing factors')
ax[0,0].tick_params(axis='x', labelrotation=90, labelsize=5)
ax[0,0].tick_params(axis='y', labelsize=5)
ax[0,1].tick_params(axis='y', labelsize=5)
g=sns.heatmap(c_E_inc_corr.loc[c_inf_mat.columns,c_E_inc_m['R'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax[1,0],cbar_ax=ax[1,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[1,0].set_title('Spearman correlation coefficients for material factors')
ax[1,0].set_xlabel('Evaluation method')
ax[1,0].set_ylabel('Contributing factors')
ax[1,0].tick_params(axis='x', labelrotation=90, labelsize=5)
ax[1,0].tick_params(axis='y', labelsize=5)
ax[1,1].tick_params(axis='y', labelsize=5)
content=[[r'$h$ - Thickness','$\overline{X}$ - Arithmetic average'],
         [r'$b$ - Width', '$\Delta X$ - Difference between max. and min.'],
         [r'$A_{CS}$ - Cross-section area','$X_{mid}$ - Midspan position'],
         [r'$V$ - Volume','$X_{circle}$ - Approximated as circle'],
         [r'$I$ - Second moment of inertia','$X_{el}$ - Elastic range'],
         [r'$\kappa$ - Geometrical curvature','$X_{y}$ - Yield'],
         [r'$ind$ - Support indentation','$X_{u}$ - Ultimate'],
         [r'$\rho_{app}$ - Apparent density','$X_{b}$ - Break'],
         [r'$f$ - Strength','$X_{opt}$ - Determination via DIC'],
         [r'$\epsilon$ - Strain','  '],
         [r'$W$ - External work','  ']]
axt= ax[2,0].table(content, loc='center', cellLoc='left', rowLoc='center',
                   colWidths=[0.5,0.5],edges='open')
ax[2,0].set_title('Symbols')
axt.auto_set_font_size(False)
axt.set_fontsize(8)
# def set_pad_for_column(col, pad=0.1):
#     cells = axt.get_celld()
#     column = [cell for cell in axt.get_celld() if cell[1] == col]
#     for cell in column:
#         cells[cell].PAD = pad
# set_pad_for_column(col=0, pad=0.01)
# ax[2,0].grid(False)
ax[2,0].axis('off')
ax[2,1].axis('off')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'02.png')
plt.savefig(out_supmat+'02.pdf')
plt.show()



fig, ax = plt.subplots(nrows=3, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1],
                                    'height_ratios': [15, 10, 10]},
                       figsize=figsize_sup, constrained_layout=True)
# fig.suptitle('Influence on the moduli of elasticity determined without (**A*)\nand with (**S*) elimination of the support indentation',
#              fontsize=12, fontweight="bold")
g=sns.heatmap(Comp_E_inc_m_SA_corr.loc[c_inf_gad.columns,
                                       Comp_E_inc_m_SA.columns.difference(Methods_excl_names)].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':0},
              xticklabels=1, ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0,0].set_title('Spearman correlation coefficients for geometrical and additional factors')
# ax[0,0].set_xlabel('Evaluation method')
ax[0,0].set_ylabel('Contributing factors')
ax[0,0].set_xticklabels([])
# ax[0,0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[0,0].tick_params(axis='y', labelsize=6)
ax[0,1].tick_params(axis='y', labelsize=6)
g=sns.heatmap(Comp_E_inc_m_SA_corr.loc[c_inf_mat.columns,
                                       Comp_E_inc_m_SA.columns.difference(Methods_excl_names)].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':0},
              xticklabels=1, ax=ax[1,0],cbar_ax=ax[1,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[1,0].set_title('Spearman correlation coefficients for material factors')
# ax[1,0].set_xlabel('Evaluation method')
ax[1,0].set_ylabel('Contributing factors')
ax[1,0].set_xticklabels([])
# ax[1,0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[1,0].tick_params(axis='y', labelsize=6)
ax[1,1].tick_params(axis='y', labelsize=6)

id_dr = Comp_E_inc_m_SA.columns.str.contains('fu')
df = Comp_E_inc_m_SA.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
ax[2,0].grid(True)
h = sns.barplot(x="index", y="data", data=df, ax=ax[2,0],
                 errwidth=1, capsize=.1)
Evac.tick_label_inserter(ax=h, pos=2, ins='*', axis='x')
ax[2,0].set_title('Relative deviation')
ax[2,0].set_ylabel('$D_{E,S-A}$')
ax[2,0].set_xlabel('Evaluation method')
ax[2,0].tick_params(axis='x', labelrotation=90, labelsize=8)
# fig.delaxes(ax[2,1])
ax[2,1].axis('off')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'03.png')
plt.savefig(out_supmat+'03.pdf')
plt.show()

fig, ax = plt.subplots(nrows=3, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1],
                                    'height_ratios': [15, 10, 10]},
                       figsize=figsize_sup, constrained_layout=True)
# fig.suptitle('Influence on the moduli of elasticity determined without (**S*)\nand with (**M*) elimination of the shear deformation,\nwith support indentation eliminated in both cases',
#              fontsize=12, fontweight="bold")
g=sns.heatmap(Comp_E_inc_m_MS_corr.loc[c_inf_gad.columns,
                                       Comp_E_inc_m_MS.columns.difference(Methods_excl_names)].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':0},
              xticklabels=1, ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0,0].set_title('Spearman correlation coefficients for geometrical and additional factors')
# ax[0,0].set_xlabel('Evaluation method')
ax[0,0].set_ylabel('Contributing factors')
ax[0,0].set_xticklabels([])
# ax[0,0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[0,0].tick_params(axis='y', labelsize=6)
ax[0,1].tick_params(axis='y', labelsize=6)
g=sns.heatmap(Comp_E_inc_m_MS_corr.loc[c_inf_mat.columns,
                                       Comp_E_inc_m_MS.columns.difference(Methods_excl_names)].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':0},
              xticklabels=1, ax=ax[1,0],cbar_ax=ax[1,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[1,0].set_title('Spearman correlation coefficients for material factors')
# ax[1,0].set_xlabel('Evaluation method')
ax[1,0].set_ylabel('Contributing factors')
ax[1,0].set_xticklabels([])
# ax[1,0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[1,0].tick_params(axis='y', labelsize=6)
ax[1,1].tick_params(axis='y', labelsize=6)

id_dr = Comp_E_inc_m_MS.columns.str.contains('fu')
df = Comp_E_inc_m_MS.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
ax[2,0].grid(True)
h = sns.barplot(x="index", y="data", data=df, ax=ax[2,0],
                 errwidth=1, capsize=.1)
Evac.tick_label_inserter(ax=h, pos=2, ins='*', axis='x')
ax[2,0].set_title('Relative deviation')
# ax[2,0].set_ylabel('$D_{E,M-S}$ (logarithmic scale)')
# ax[2,0].set_yscale('log')
ax[2,0].set_ylabel('$D_{E,M-S}$')
tmp=ax[2,0].get_xlim()
ax[2,0].hlines([0.2275,0.2275,0.2275],
                 [-0.4, 0.6, 11.6],
                 [ 0.4, 1.4, 12.4],
                 colors='red', linestyles='solid', linewidth=4.0)
ax[2,0].set_xlim(tmp)
ax[2,0].set_ylim(-0.001,0.23)
t=Comp_E_inc_m_MS.loc(axis=1)[['A2l','A4l','F4gha']].agg(['mean',Evac.confidence_interval])
ftxt=['Trimmed values:',
      '   A2*l: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','A2l'],*t.loc['confidence_interval','A2l']),
      '   A4*l: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','A4l'],*t.loc['confidence_interval','A4l']),
      '   F4*gha: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','F4gha'],*t.loc['confidence_interval','F4gha'])]
ax[2,0].text(2.6,0.16,'\n'.join(ftxt),ha='left',va='bottom', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax[2,0].set_xlabel('Evaluation method')
ax[2,0].tick_params(axis='x', labelrotation=90, labelsize=8)
# fig.delaxes(ax[2,1])
ax[2,1].axis('off')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'04.png')
plt.savefig(out_supmat+'04.pdf')
plt.show()


fig, ax = plt.subplots(nrows=3, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1],
                                    'height_ratios': [15, 10, 10]},
                       figsize=figsize_sup, constrained_layout=True)
# fig.suptitle('Influence on the moduli of elasticity determined without (**A*)\nand with (**M*) elimination of the shear deformation,\nas well as the support indentation',
#              fontsize=12, fontweight="bold")
g=sns.heatmap(Comp_E_inc_m_MA_corr.loc[c_inf_gad.columns,
                                       Comp_E_inc_m_MA.columns.difference(Methods_excl_names)].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':0},
              xticklabels=1, ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0,0].set_title('Spearman correlation coefficients for geometrical and additional factors')
# ax[0,0].set_xlabel('Evaluation method')
ax[0,0].set_ylabel('Contributing factors')
ax[0,0].set_xticklabels([])
# ax[0,0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[0,0].tick_params(axis='y', labelsize=6)
ax[0,1].tick_params(axis='y', labelsize=6)
g=sns.heatmap(Comp_E_inc_m_MA_corr.loc[c_inf_mat.columns,
                                       Comp_E_inc_m_MA.columns.difference(Methods_excl_names)].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':0},
              xticklabels=1, ax=ax[1,0],cbar_ax=ax[1,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[1,0].set_title('Spearman correlation coefficients for material factors')
# ax[1,0].set_xlabel('Evaluation method')
ax[1,0].set_ylabel('Contributing factors')
ax[1,0].set_xticklabels([])
# ax[1,0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[1,0].tick_params(axis='y', labelsize=6)
ax[1,1].tick_params(axis='y', labelsize=6)

id_dr = Comp_E_inc_m_MA.columns.str.contains('fu')
df = Comp_E_inc_m_MA.loc(axis=1)[np.invert(id_dr)].unstack(level=0).reset_index(level=1, drop=True)
df = df.reset_index(name='data')
ax[2,0].grid(True)
h = sns.barplot(x="index", y="data", data=df, ax=ax[2,0],
                 errwidth=1, capsize=.1)
Evac.tick_label_inserter(ax=h, pos=2, ins='*', axis='x')
ax[2,0].set_title('Relative deviation')
# ax[2,0].set_ylabel('$D_{E,M-A}$ (logarithmic scale)')
# ax[2,0].set_yscale('log')
ax[2,0].set_ylabel('$D_{E,M-A}$')
tmp=ax[2,0].get_xlim()
ax[2,0].hlines([1.38,1.38],
                 [-0.4, 0.6],
                 [ 0.4, 1.4],
                 colors='red', linestyles='solid', linewidth=4.0)
ax[2,0].set_xlim(tmp)
ax[2,0].set_ylim(-.41,1.4)
t=Comp_E_inc_m_MA.loc(axis=1)[['A2l','A4l']].agg(['mean',Evac.confidence_interval])
ftxt='Trimmed values:\n  A2*l: %5.2f (%4.2f - %5.2f)\n  A4*l: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','A2l'],*t.loc['confidence_interval','A2l'],
                                                                                      t.loc['mean','A4l'],*t.loc['confidence_interval','A4l'])
ax[2,0].text(-0.35,-0.35,ftxt,ha='left',va='bottom', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
ax[2,0].set_xlabel('Evaluation method')
ax[2,0].tick_params(axis='x', labelrotation=90, labelsize=8)
# fig.delaxes(ax[2,1])
ax[2,1].axis('off')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'05.png')
plt.savefig(out_supmat+'05.pdf')
plt.show()


df=Comp_E_con.loc(axis=1)[np.invert(Comp_E_con.columns.str.contains('fu'))]
df=df.loc(axis=1)[np.invert(df.columns.str.contains('A0Al'))]
fig, ax = plt.subplots(nrows=3,figsize=figsize_sup, constrained_layout=True)
# fig.suptitle('Relative deviation of Youngs Moduli compared to\nconventional determined (Method A0Al)',
#              fontsize=12, fontweight="bold")
g = sns.barplot(data=df.loc(axis=1)[df.columns.str[2:3]=='A'],
                 ax=ax[0], errwidth=1, capsize=.1)
ax[0].set_title('Methods without elimination (**A*)')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=90)
# ax[0].set_ylabel('$D_{E,Method-A0Al} = (E_{Method} - E_{%s})/E_{%s}$'%(YM_con[2],YM_con[2]))
ax[0].set_ylabel('$D_{E,Method-A0Al}$')
# ax[0].set_ylim(-0.01,2)
g = sns.barplot(data=df.loc(axis=1)[df.columns.str[2:3]=='S'],
                 ax=ax[1], errwidth=1, capsize=.1)
ax[1].set_title('Methods with support indentation elimination (**S*)')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=90)
# ax[1].set_ylabel('$D_{E,Method-A0Al} = (E_{Method} - E_{%s})/E_{%s}$'%(YM_con[2],YM_con[2]))
ax[1].set_ylabel('$D_{E,Method-A0Al}$')
# ax[1].set_ylim(-0.01,2)
g = sns.barplot(data=df.loc(axis=1)[(df.columns.str[2:3]=='M')|(df.columns.str[2:3]=='C')],
                 ax=ax[2], errwidth=1, capsize=.1)
ax[2].set_title('Methods with support indentation and shear deformation elimination (**M*|**C*)')
ax[2].set_xlabel('Evaluation method')
ax[2].tick_params(axis='x', labelrotation=90)
# ax[2].set_ylabel('$D_{E,Method-A0Al} = (E_{Method} - E_{%s})/E_{%s}$'%(YM_con[2],YM_con[2]))
ax[2].set_ylabel('$D_{E,Method-A0Al}$')
tmp=ax[2].get_xlim()
ax[2].hlines([3.275,3.275],
             [-0.4, 0.6],
             [ 0.4, 1.4],
             colors='red', linestyles='solid', linewidth=4.0)
ax[2].set_xlim(tmp)
ax[2].set_ylim(-0.01,3.3)
t=df.loc(axis=1)[['A2Ml','A4Ml']].agg(['mean',Evac.confidence_interval])
ftxt='Trimmed values:\n  A2Ml: %5.2f (%4.2f - %5.2f)\n  A4Ml: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','A2Ml'],*t.loc['confidence_interval','A2Ml'],
                                                                                     t.loc['mean','A4Ml'],*t.loc['confidence_interval','A4Ml'])
ax[2].text(1.81,2.21,ftxt,ha='left',va='bottom', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'06.png')
plt.savefig(out_supmat+'06.pdf')
plt.show()

#%% Extra (Oveview Mat-params and Regression)

def agg_add_ci(pdo, agg_funcs):
    a=pdo.agg(agg_funcs)
    a1=pdo.agg(Evac.confidence_interval)
    a1.index=['ci_min','ci_max']
    a=pd.concat([a,a1])
    return a

#%%% Overview
txt="\n "+"="*100
txt+=("\n Overview:\n")
Evac.MG_strlog(txt,log_mg,1,printopt=False)
# a=cs.loc(axis=1)[[YM_con_str,YM_opt_str,'fy','fu','ey_opt','eu_opt','Density_app']].agg(agg_funcs)
# a1=cs.loc(axis=1)[[YM_con_str,YM_opt_str,'fy','fu','ey_opt','eu_opt','Density_app']].agg(Evac.confidence_interval)
# a1.index=['ci_min','ci_max']
# a=pd.concat([a,a1])
# # a=cs.loc(axis=1)[[YM_con_str,YM_opt_str,'fy','fu','Density_app']].describe()
a=agg_add_ci(cs.loc(axis=1)[[YM_con_str,YM_opt_str,'fy','fu','ey_opt','eu_opt','Density_app']],agg_funcs)
Evac.MG_strlog(a.to_string(),log_mg,1,printopt=False)

txt=("\n\n Overview E (inc+R) Methods (A0Al and **M* wo. fu):\n")
Evac.MG_strlog(txt,log_mg,1,printopt=False)
a=['A0Al','A2Ml','A4Ml','B1Ml','B2Ml','C2Ml','C2Cl',
   'D1Mguw','D1Mgwt','D2Mguw','D2Mgwt',
   'E4Ml','E4Me','E4Mg',
   'F4Mgha','F4Mgth',
   'G3Mg','G3Cg']
a = ['E_inc_R_%s_meanwoso' % (x,) for x in a]
a=agg_add_ci(cs.loc(axis=1)[a],agg_funcs)
Evac.MG_strlog(a.T.to_string(),log_mg,1,printopt=False)

txt=("\n\n Overview CV (inc+R) Methods (A0Al and **M* wo. fu):\n")
Evac.MG_strlog(txt,log_mg,1,printopt=False)
a=['A0Al','A2Ml','A4Ml','B1Ml','B2Ml','C2Ml','C2Cl',
   'D1Mguw','D1Mgwt','D2Mguw','D2Mgwt',
   'E4Ml','E4Me','E4Mg',
   'F4Mgha','F4Mgth',
   'G3Mg','G3Cg']
a = ['E_inc_R_%s_stdnwoso' % (x,) for x in a]
a=agg_add_ci(cs.loc(axis=1)[a],agg_funcs)
Evac.MG_strlog(a.T.to_string(),log_mg,1,printopt=False)


txt=("\n\n Overview material params:\n")
Evac.MG_strlog(txt,log_mg,1,printopt=False)
a=agg_add_ci(c_inf_mat,agg_funcs)
Evac.MG_strlog(a.T.to_string(),log_mg,1,printopt=False)



#%%% Regression 


txt="\n "+"-"*100
txt+=("\n Linear regression:\n")
formula_in_dia=True
Lin_reg_df =pd.DataFrame([],dtype='O')

def MG_linRegStats(Y,X,Y_txt,X_txt):
    des_txt = ("\n  - Linear relation between %s and %s:\n   "%(Y_txt,X_txt))
    tmp  = Evac.YM_sigeps_lin(Y, X)
    smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
    out=pd.Series([*tmp,smst,des_txt],index=['s','c','Rquad','fit','smstat','Description'])
    return out

Lin_reg_df['Econvsfu'] = MG_linRegStats(cs[YM_con_str], cs['fu'],
                                        "Youngs Modulus (conventional)",
                                        "ultimate strength")
Lin_reg_df['Eoptvsfu'] = MG_linRegStats(cs[YM_opt_str], cs['fu'],
                                        "Youngs Modulus (optical)",
                                        "ultimate strength")

Lin_reg_df['Econvsfy'] = MG_linRegStats(cs[YM_con_str], cs['fy'],
                                        "Youngs Modulus (conventional)",
                                        "yield strength")
Lin_reg_df['Eoptvsfy'] = MG_linRegStats(cs[YM_opt_str], cs['fy'],
                                        "Youngs Modulus (optical)",
                                        "yield strength")
# 48-17_LuPeCo_cl13b - eu opt = NaN
Lin_reg_df['Econvseuopt'] = MG_linRegStats(cs.loc[~cs['eu_opt'].isna()][YM_con_str], 
                                           cs.loc[~cs['eu_opt'].isna()]['eu_opt'],
                                        "Youngs Modulus (conventional)",
                                        "ultimate strain (optical)")
Lin_reg_df['Eoptvseuopt'] = MG_linRegStats(cs.loc[~cs['eu_opt'].isna()][YM_opt_str],
                                           cs.loc[~cs['eu_opt'].isna()]['eu_opt'],
                                        "Youngs Modulus (optical)",
                                        "ultimate strain (optical)")
Lin_reg_df['Econvseyopt'] = MG_linRegStats(cs[YM_con_str], cs['ey_opt'],
                                        "Youngs Modulus (conventional)",
                                        "yield strain (optical)")
Lin_reg_df['Eoptvseyopt'] = MG_linRegStats(cs[YM_opt_str], cs['ey_opt'],
                                        "Youngs Modulus (optical)",
                                        "yield strain (optical)")



Lin_reg_df['EconvsRho'] = MG_linRegStats(cs[YM_con_str], cs['Density_app'],
                                         "Youngs Modulus (conventional)",
                                         "apparent density")
Lin_reg_df['EoptvsRho'] = MG_linRegStats(cs[YM_opt_str], cs['Density_app'],
                                         "Youngs Modulus (optical)",
                                         "apparent density")
Lin_reg_df['fuvsRho'] = MG_linRegStats(cs['fu'], cs['Density_app'],
                                       "ultimate strength",
                                       "apparent density")
Lin_reg_df['fyvsRho'] = MG_linRegStats(cs['fy'], cs['Density_app'],
                                       "yield strength",
                                       "apparent density")

for i in Lin_reg_df.columns:
    txt += Lin_reg_df.loc['Description',i]
    txt += Evac.str_indent(Evac.fit_report_adder(*Lin_reg_df.loc[['fit','Rquad'],i]))
    txt += Evac.str_indent(Lin_reg_df.loc['smstat',i])

Evac.MG_strlog(txt,log_mg,1,printopt=False)




fig, ax1 = plt.subplots()
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Econvsfu'].iloc[:3]).split(',')
    txt = r'$E_{A0Al}$ = %s $f_{u}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{A0Al}$'
ax = sns.regplot(x="fu", y=YM_con_str, data=cs, label=txt, ax=ax1)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Eoptvsfu'].iloc[:3]).split(',')
    txt = r'$E_{D2Mgwt}$ = %s $f_{u}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{D2Mgwt}$'        
ax = sns.regplot(x="fu", y=YM_opt_str, data=cs, label=txt, ax=ax1)
ax1.legend()
# ax1.set_title('%s\nYoungs-modulus vs. ultimate strength'%name_Head)
ax1.set_xlabel('$f_{u}$ / MPa')
ax1.set_ylabel('$E$ / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(out_full+'-Reg-E_fu.pdf')
# plt.savefig(out_full+'-Reg-E_fu.png')
plt.show()

fig, ax1 = plt.subplots()
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Econvsfy'].iloc[:3]).split(',')
    txt = r'$E_{A0Al}$ = %s $f_{y}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{A0Al}$'
ax = sns.regplot(x="fu", y=YM_con_str, data=cs, label=txt, ax=ax1)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Eoptvsfy'].iloc[:3]).split(',')
    txt = r'$E_{D2Mgwt}$ = %s $f_{y}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{D2Mgwt}$'        
ax = sns.regplot(x="fu", y=YM_opt_str, data=cs, label=txt, ax=ax1)
ax1.legend()
# ax1.set_title('%s\nYoungs-modulus vs. ultimate strength'%name_Head)
ax1.set_xlabel('$f_{y}$ / MPa')
ax1.set_ylabel('$E$ / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(out_full+'-Reg-E_fu.pdf')
# plt.savefig(out_full+'-Reg-E_fu.png')
plt.show()


fig, ax1 = plt.subplots()
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['EconvsRho'].iloc[:3]).split(',')
    txt = r'$E_{A0Al}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt='$E_{A0Al}$'
ax = sns.regplot(x="Density_app", y=YM_con_str, data=cs,
                 label=txt, ax=ax1)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['EoptvsRho'].iloc[:3]).split(',')
    txt = r'$E_{D2Mgwt}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt='$E_{D2Mgwt}$'
ax = sns.regplot(x="Density_app", y=YM_opt_str, data=cs,
                 label=txt, ax=ax1)
ax1.legend()
# ax1.set_title('%s\nYoungs-modulus vs. density'%name_Head)
ax1.set_xlabel(r'$\rho_{app}$ / (g/cm³)')
ax1.set_ylabel(r'$E$ / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(out_full+'-Reg-E_Rho.pdf')
# plt.savefig(out_full+'-Reg-E_Rho.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['fuvsRho'].iloc[:3]).split(',')
    txt = r'$f_{u}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$f_{u}$'
ax = sns.regplot(x="Density_app", y='fu', data=cs, label=txt, ax=ax1)
# ax1.set_title('%s\nUltimate strength vs. density'%name_Head)
ax1.set_xlabel(r'$\rho_{app}$ / (g/cm³)')
ax1.set_ylabel(r'$f_{u}$ / MPa')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(out_full+'-Reg-fu_Rho.pdf')
# plt.savefig(out_full+'-Reg-fu_Rho.pdf')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['fyvsRho'].iloc[:3]).split(',')
    txt = r'$f_{y}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$f_{y}$'
ax = sns.regplot(x="Density_app", y='fy', data=cs, label=txt, ax=ax1)
# ax1.set_title('%s\nUltimate strength vs. density'%name_Head)
ax1.set_xlabel(r'$\rho_{app}$ / (g/cm³)')
ax1.set_ylabel(r'$f_{y}$ / MPa')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(out_full+'-Reg-fu_Rho.pdf')
# plt.savefig(out_full+'-Reg-fu_Rho.pdf')
plt.show()



# fig, ax = plt.subplots(ncols=1,nrows=3,
#                        figsize=figsize_sup, constrained_layout=True)
# if formula_in_dia:
#     txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['EconvsRho'].iloc[:3]).split(',')
#     txt = r'$E_{A0Al}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
# else:
#     txt='$E_{A0Al}$'
# axt = sns.regplot(x="Density_app", y=YM_con_str, data=cs,
#                  label=txt, ax=ax[0])
# if formula_in_dia:
#     txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['EoptvsRho'].iloc[:3]).split(',')
#     txt = r'$E_{D2Mgwt}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
# else:
#     txt='$E_{D2Mgwt}$'
# axt = sns.regplot(x="Density_app", y=YM_opt_str, data=cs,
#                  label=txt, ax=ax[0])
# ax[0].legend()
# ax[0].set_title('Modulus of elasticity vs. apparent density')
# ax[0].set_xlabel(r'$\rho_{app}$ / (g/cm³)')
# ax[0].set_ylabel(r'$E$ / MPa')

# if formula_in_dia:
#     txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Econvsfu'].iloc[:3]).split(',')
#     txt = r'$E_{A0Al}$ = %s $f_{u}$ + %s ($R²$ = %s)'%(*txt,)
# else:
#     txt = '$E_{A0Al}$'
# axt = sns.regplot(x="fu", y=YM_con_str, data=cs, label=txt, ax=ax[1])
# if formula_in_dia:
#     txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Eoptvsfu'].iloc[:3]).split(',')
#     txt = r'$E_{D2Mgwt}$ = %s $f_{u}$ + %s ($R²$ = %s)'%(*txt,)
# else:
#     txt = '$E_{D2Mgwt}$'        
# axt = sns.regplot(x="fu", y=YM_opt_str, data=cs, label=txt, ax=ax[1])
# ax[1].legend()
# ax[1].set_title('Modulus of elasticity vs. ultimate strength')
# ax[1].set_xlabel('$f_{u}$ / MPa')
# ax[1].set_ylabel('$E$ / MPa')

# if formula_in_dia:
#     txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Econvseuopt'].iloc[:3]).split(',')
#     txt = r'$E_{A0Al}$ = %s $\epsilon_{u,opt}$ + %s ($R²$ = %s)'%(*txt,)
# else:
#     txt = '$E_{A0Al}$'
# axt = sns.regplot(x="eu_opt", y=YM_con_str, data=cs, label=txt, ax=ax[2])
# if formula_in_dia:
#     txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Eoptvseuopt'].iloc[:3]).split(',')
#     txt = r'$E_{D2Mgwt}$ = %s $\epsilon_{u,opt}$ + %s ($R²$ = %s)'%(*txt,)
# else:
#     txt = '$E_{D2Mgwt}$'        
# axt = sns.regplot(x="eu_opt", y=YM_opt_str, data=cs, label=txt, ax=ax[2])
# ax[2].legend()
# ax[2].set_title('Modulus of elasticity vs. strain at ultimate stress')
# ax[2].set_xlabel('$\epsilon_{u,opt}$ / -')
# ax[2].set_ylabel('$E$ / MPa')
# tmp=ax[2].get_ylim()
# ax[2].set_ylim(-500,tmp[1])
# fig.suptitle('')
# plt.savefig(out_supmat+'RR.png')
# plt.savefig(out_supmat+'RR.pdf')
# plt.show()
# plt.close(fig)


fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=figsize_sup, constrained_layout=True)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['EconvsRho'].iloc[:3]).split(',')
    txt = r'$E_{A0Al}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt='$E_{A0Al}$'
axt = sns.regplot(x="Density_app", y=YM_con_str, data=cs,
                 ci=95, seed=0, truncate=True, label=txt, ax=ax[0])
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['EoptvsRho'].iloc[:3]).split(',')
    txt = r'$E_{D2Mgwt}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt='$E_{D2Mgwt}$'
axt = sns.regplot(x="Density_app", y=YM_opt_str, data=cs,
                 ci=95, seed=0, truncate=True, label=txt, ax=ax[0])
ax[0].legend()
ax[0].set_title('Modulus of elasticity vs. apparent density')
ax[0].set_xlabel(r'$\rho_{app}$ / (g/cm³)')
ax[0].set_ylabel(r'$E$ / MPa')
#ax[0].set_xscale("log")
ax[0].set_yscale("log")
tmp=ax[0].get_ylim()
ax[0].set_ylim(10,10000)

if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Econvsfu'].iloc[:3]).split(',')
    txt = r'$E_{A0Al}$ = %s $f_{u}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{A0Al}$'
axt = sns.regplot(x="fu", y=YM_con_str, data=cs, 
                  ci=95, seed=0, truncate=True,label=txt, ax=ax[1])
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Eoptvsfu'].iloc[:3]).split(',')
    txt = r'$E_{D2Mgwt}$ = %s $f_{u}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{D2Mgwt}$'        
axt = sns.regplot(x="fu", y=YM_opt_str, data=cs, 
                  ci=95, seed=0, truncate=True, label=txt, ax=ax[1])
ax[1].legend()
ax[1].set_title('Modulus of elasticity vs. ultimate strength')
ax[1].set_xlabel('$f_{u}$ / MPa')
ax[1].set_ylabel('$E$ / MPa')
#ax[1].set_xscale("log")
ax[1].set_yscale("log")
tmp=ax[1].get_ylim()
ax[1].set_ylim(10,10000)

if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Econvseuopt'].iloc[:3]).split(',')
    txt = r'$E_{A0Al}$ = %s $\epsilon_{u,opt}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{A0Al}$'
axt = sns.regplot(x="eu_opt", y=YM_con_str,
                  ci=95, seed=0, truncate=True, data=cs, label=txt, ax=ax[2])
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Eoptvseuopt'].iloc[:3]).split(',')
    txt = r'$E_{D2Mgwt}$ = %s $\epsilon_{u,opt}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{D2Mgwt}$'        
axt = sns.regplot(x="eu_opt", y=YM_opt_str,
                  ci=95, seed=0, truncate=True, data=cs, label=txt, ax=ax[2])
ax[2].legend()
ax[2].set_title('Modulus of elasticity vs. strain at ultimate stress')
ax[2].set_xlabel('$\epsilon_{u,opt}$ / -')
ax[2].set_ylabel('$E$ / MPa')
a#x[2].set_xscale("log")
ax[2].set_yscale("log")
tmp=ax[2].get_ylim()
ax[2].set_ylim(10,10000)
fig.suptitle('')
plt.savefig(out_supmat+'RR.png')
plt.savefig(out_supmat+'RR.pdf')
plt.show()
plt.close(fig)

#%% Close Log
log_mg.close()

