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
                     'geo_curve_cim': r'$\kappa_{mid,circle}$',
                     'ind_R_max': r'$ind_{el,max}$','ind_R_mean': r'$\overline{ind}_{el}$',
                     'ind_U_max': r'$ind_{u,max}$','ind_U_mean': r'$\overline{ind}_{u}$'}
Donor_dict={"LEIULANA_52-17":"PT2",
            "LEIULANA_67-17":"PT3",
            "LEIULANA_57-17":"PT4",
            "LEIULANA_48-17":"PT5",
            "LEIULANA_22-17":"PT6",
            "LEIULANA_60-17":"PT7"}

path = "F:/Mess_TBT_Verf/Auswertung/Complete/"
name_paper="Paper_"
name_verf="Verf_"
name_supmat="SUP_"
name_in   = "Verif_TBT-Summary_all"
name_out  = "Verif_TBT-Conclusion"
name_Head = "Youngs Modulus determination methods - Verification"
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



# YM_con=['inc','R','A0Al','meanwoso']
# YM_opt=['inc','R','D2Mgwt','meanwoso']
# Fixed 25.10.22
YM_con=['inc','F','A0Al','meanwoso']
YM_opt=['inc','F','D2Mgwt','meanwoso']
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
out_verf= os.path.abspath(path+name_verf)
out_paper= os.path.abspath(path+name_paper)
out_supmat= os.path.abspath(path+name_supmat)
# path_doda = 'F:/Messung/000-PARAFEMM_Patientendaten/PARAFEMM_Donordata_full.xlsx'
h5_conc = 'Summary'
h5_data = 'Add_/Measurement'
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
data_read.close()


del dfa['Number']
dfa.Failure_code  = Evac.list_cell_compiler(dfa.Failure_code)
dfa['statistics'] = Evac.list_interpreter(dfa.Failure_code, no_stats_fc)

#fehlende Einfluss (Geometrie-) Werte
dfa['thickness_mean']=dfa.loc(axis=1)[dfa.columns.str.startswith('thick')].mean(axis=1)
dfa['width_mean']=dfa.loc(axis=1)[dfa.columns.str.startswith('width')].mean(axis=1)
dfa['Density_app']=dfa.eval('Mass_org/geo_Vol_int')*1000
VIParams_rename={'geo_Vol_int':'Volume', 'geo_ACS_mean':'Area_CS'}
dfa.rename(columns=VIParams_rename,inplace=True)

# h=dfa.Origin
# for i in relist:
#     h=h.str.replace(i,'')
# h2=h.map(Locdict)
# if (h2.isna()).any(): print('Locdict have missing/wrong values! \n   (Lnr: %s)'%['{:d}'.format(i) for i in h2.loc[h2.isna()].index])
# dfa.insert(3,'Origin_short',h)
# dfa.insert(4,'Origin_sshort',h2)
# del h, h2

cs = dfa.loc[dfa.statistics]
a=cs.Designation.apply(lambda x: [x[0:3],x[3:5],x[5],x[6]])
a=pd.DataFrame(a.to_list(), index=a.index, columns=['a','Kind','Vers','Rot'])
cs=pd.concat([cs,a[['Kind','Vers','Rot']]],axis=1)
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
h_eva = cs.groupby('Type')[cs_num_cols].agg(['size','min','max','mean','std'],
                                       **{'skipna':True,'numeric_only':True,'ddof':1})
h_eva_add = cs.groupby('Type')[cs_num_cols].agg([Evac.stat_outliers],
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
# E_inc_m_R_MI  = c_E_inc_m['R']
# Fixed 25.10.22
E_inc_m_R_MI  = c_E_inc_m['F']
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
# a=c_E_inc.loc(axis=1)['R',:,['meanwoso','stdnwoso']].droplevel(axis=1,level=[0])
# Fixed 25.10.22
a=c_E_inc.loc(axis=1)['F',:,['meanwoso','stdnwoso']].droplevel(axis=1,level=[0])
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
# a=c_E.loc(axis=1)[idx[:,:,:,['E','R','meanwoso','stdnwoso']]].agg(Evac.stat_outliers)
# Fixed 25.10.22
a=c_E.loc(axis=1)[idx[:,:,:,['E','F','meanwoso','stdnwoso']]].agg(Evac.stat_outliers)
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
# a=c_E_inc.loc(axis=1)['R',:,'meanwoso'].droplevel(axis=1,level=[0,2]).agg(['mean','std'])
# Fixed 25.10.22
a=c_E_inc.loc(axis=1)['F',:,'meanwoso'].droplevel(axis=1,level=[0,2]).agg(['mean','std'])
a=(a.loc['std']/a.loc['mean']).sort_values()
Evac.MG_strlog(Evac.str_indent(a.T.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\n    - Youngs modulus' coefficient of variation without statistical outliers (std/mean) (inc + R):",
               log_mg, printopt=False)
# b=c_E_inc.loc(axis=1)['R',:,'meanwoso'].droplevel(axis=1,level=[0,2]).agg([Evac.meanwoso,Evac.stdwoso])
# Fixed 25.10.22
b=c_E_inc.loc(axis=1)['F',:,'meanwoso'].droplevel(axis=1,level=[0,2]).agg([Evac.meanwoso,Evac.stdwoso])
b=(b.loc['stdwoso']/b.loc['meanwoso']).sort_values()
Evac.MG_strlog(Evac.str_indent(b.T.to_string(),5), log_mg, printopt=False)


Evac.MG_strlog("\n\n  - Comparison of methods to conventional:",
               log_mg, printopt=False)
# a=c_E_inc.loc(axis=1)['R',:,'meanwoso'].droplevel(axis=1,level=[0,2])
# Fixed 25.10.22
a=c_E_inc.loc(axis=1)['F',:,'meanwoso'].droplevel(axis=1,level=[0,2])
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
# a = c_E_inc_m['R']
# Fixed 25.10.22
a = c_E_inc_m['F']
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
# a = c_E_inc_m['R']
# Fixed 25.10.22
a = c_E_inc_m['F']
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
c_inf_geo = cs.loc(axis=1)['thickness_mean','width_mean', 
                           'Area_CS','Volume','Density_app']
c_inf_add = cs.loc(axis=1)['width_1','width_2','width_3',
                           'thickness_1','thickness_2','thickness_3',
                           'geo_MoI_max','geo_MoI_min','geo_MoI_mid',
                           'geo_curve_max','geo_curve_min','geo_curve_mid',
                           'geo_curve_cim',
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
c_inf_add_new['geo_curve_cim'] = c_inf_add['geo_curve_cim']
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
                           'fu','eu_opt','Wu_opt']
                           # 'fu','eu_opt','Wu_opt','fb','eb_opt','Wb_opt']

# c_inf=pd.concat([c_inf_geo,c_inf_add_new,c_inf_mat],axis=1)
c_inf=pd.concat([c_inf_add_new,c_inf_mat],axis=1)

# Speicherung stat-data einfluss
t=c_inf.agg(agg_funcs)
t1=c_inf.agg(Evac.confidence_interval)
t1.index=['ci_min','ci_max']
t=pd.concat([t,t1])
t.to_csv(out_full+'_influence_param_stats.csv',sep=';') 

# c_E_lsq_corr = pd.concat([c_inf,c_E_inc_m['R']],axis=1).corr(method=mcorr)
# c_E_inc_corr = pd.concat([c_inf,c_E_inc_m['R']],axis=1).corr(method=mcorr)
# Fixed 25.10.22
c_E_lsq_corr = pd.concat([c_inf,c_E_inc_m['F']],axis=1).corr(method=mcorr)
c_E_inc_corr = pd.concat([c_inf,c_E_inc_m['F']],axis=1).corr(method=mcorr)


fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, 
                              sharex=False, sharey=False, figsize = (6.3,2*3.54))
fig.suptitle('%s\nInfluence on least-square determined YM (only refined range)'%name_Head)
# g=sns.heatmap(c_E_lsq_corr.loc[c_inf_gad.columns,c_E_lsq_m['R'].columns].round(1),
# Fixed 25.10.22
g=sns.heatmap(c_E_lsq_corr.loc[c_inf_gad.columns,c_E_lsq_m['F'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('Influence - geometrical and additional')
ax1.set_xlabel('Determination method')
ax1.set_ylabel('Influence')
ax1.tick_params(axis='x', labelrotation=90, labelsize=5)
ax1.tick_params(axis='y', labelsize=5)
# g=sns.heatmap(c_E_lsq_corr.loc[c_inf_mat.columns,c_E_lsq_m['R'].columns].round(1),
# Fixed 25.10.22
g=sns.heatmap(c_E_lsq_corr.loc[c_inf_mat.columns,c_E_lsq_m['F'].columns].round(1),
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
# g=sns.heatmap(c_E_inc_corr.loc[c_inf_gad.columns,c_E_inc_m['R'].columns].round(1),
# Fixed 25.10.22
g=sns.heatmap(c_E_inc_corr.loc[c_inf_gad.columns,c_E_inc_m['F'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('Influence - geometrical and additional')
ax1.set_xlabel('Determination method')
ax1.set_ylabel('Influence')
ax1.tick_params(axis='x', labelrotation=90, labelsize=5)
ax1.tick_params(axis='y', labelsize=5)
# g=sns.heatmap(c_E_inc_corr.loc[c_inf_mat.columns,c_E_inc_m['R'].columns].round(1),
# Fixed 25.10.22
g=sns.heatmap(c_E_inc_corr.loc[c_inf_mat.columns,c_E_inc_m['F'].columns].round(1),
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


#%%% Verfication
#%%%% Log
# YMMetoI=[YM_con_str,'E_inc_R_C2Ml_meanwoso',YM_opt_str]
# Fixed 25.10.22
YMMetoI=[YM_con_str,'E_inc_F_C2Ml_meanwoso',YM_opt_str]

Evac.MG_strlog("\n\n "+"="*100, log_mg, printopt=False)
Evac.MG_strlog("\nVerification", log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Overview:",3), 
               log_mg, printopt=False)
a=cs.groupby('Type')[['Density_app']+YMMetoI].agg(['mean','std']).T
Evac.MG_strlog(Evac.str_indent(a.to_string(),3), log_mg, printopt=False)

Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("PMMA:",3), 
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Statistics for con. and opt.:",5),
               log_mg, printopt=False)
a=cs.loc[cs.Type=='ACRYL'][YMMetoI].agg(agg_funcs)
Evac.MG_strlog(Evac.str_indent(a.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Deviation by rotation:",5),
               log_mg, printopt=False)
b=cs.loc[cs.Type=='ACRYL'].copy(deep=True)
b.index=pd.MultiIndex.from_frame(b[['Type','Kind','Vers','Rot']])
c=b.loc[idx[:,:,:,'A'],YMMetoI].droplevel(3)
d=b.loc[idx[:,:,:,'B'],YMMetoI].droplevel(3)
e=(d-c)/c
Evac.MG_strlog(Evac.str_indent(e.T.to_string(),5), log_mg, printopt=False)

Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Kind of specimen geometry:",3), 
               log_mg, printopt=False)
b=cs.copy(deep=True)
b.index=pd.MultiIndex.from_frame(b[['Type','Kind','Vers','Rot']])
b.drop(['Type','Kind','Vers','Rot'],axis=1,inplace=True)
c=b[YMMetoI].groupby(['Type','Kind']).agg(['min','max','mean','std'])
Evac.MG_strlog(Evac.str_indent(c.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Deviation to Rectangle-mean (01) (E-E_01)/E_01)):",5),
               log_mg, printopt=False)
b=cs.copy(deep=True)
b.index=pd.MultiIndex.from_frame(b[['Type','Kind','Vers','Rot']])
c=b[YMMetoI]
d=c.groupby(['Type','Kind']).mean().loc(axis=0)[idx[:,'01']].droplevel(1)
e=(c-d)/d
Evac.MG_strlog(Evac.str_indent(e.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Deviation by rotation:",5),
               log_mg, printopt=False)
c=b.loc[idx[:,:,:,'A'],YMMetoI].droplevel(3)
d=b.loc[idx[:,:,:,'B'],YMMetoI].droplevel(3)
e=(d-c)/c
Evac.MG_strlog(Evac.str_indent(e.to_string(),5), log_mg, printopt=False)

#%%%% Plot

def sns_ppwMMeb(ax, data, x,y, hue=None,
                dodge=0.2, join=False, palette=None,
                markers=['o','P'], scale=1, barsabove=True, capsize=2,
                controlout=False):
    axt = sns.pointplot(data=data, 
                         x=x, y=y, hue=hue,
                         ax=ax, join=join, dodge=dodge, legend=False,scale=scale,
                         markers=markers, palette=palette,
                         ci=None, errwidth=1, capsize=.1)
    if hue is None:
        # erragg = data.groupby(x).agg(['mean','min','max'])[y]
        erragg = data.groupby(x).agg(['mean','min','max'],sort=False)[y]
    else:
        # erragg = data.groupby([hue,x]).agg(['mean','min','max'])[y]
        erragg = data.groupby([hue,x]).agg(['mean','min','max'],sort=False)[y]
    # very qnd!!! (sonst falsch zugeordnet, prüfen!!!)
    if hue is not None:
        orgind=[data[hue].drop_duplicates().to_list(),data[x].drop_duplicates().to_list()]
        erragg=erragg.sort_index().loc(axis=0)[pd.MultiIndex.from_product(orgind)]
        
    errors=erragg.rename({'mean':'M','min':'I','max':'A'},axis=1)
    errors=errors.eval('''Min = M-I
                          Max = A-M''').loc(axis=1)[['Min','Max']].T
    i=0
    for point_pair in axt.collections:
        if hue is None:
            if i<1:
                i+=1
                colors=point_pair.get_facecolor()[0]
                x_coords = []
                y_coords = []
                for x, y in point_pair.get_offsets():
                    x_coords.append(x)
                    y_coords.append(y)
                ax.errorbar(x_coords, y_coords, yerr=errors.values,
                            c=colors, fmt=' ', zorder=-1, barsabove=barsabove, capsize=capsize)
        elif (i<=len(errors.columns.get_level_values(0).drop_duplicates())-1):
            errcol=errors.columns.get_level_values(0).drop_duplicates()[i]
            i+=1
            colors=point_pair.get_facecolor()[0]
            x_coords = []
            y_coords = []
            for x, y in point_pair.get_offsets():
                x_coords.append(x)
                y_coords.append(y)
            ax.errorbar(x_coords, y_coords, yerr=errors.loc(axis=1)[errcol].values,
                        c=colors, fmt=' ', zorder=-1, barsabove=barsabove, capsize=capsize)
    if controlout:
        return erragg

# b=cs.copy(deep=True)
# b.index=pd.MultiIndex.from_frame(b[['Type','Kind','Vers','Rot']])
# b.drop(['Type','Kind','Vers','Rot'],axis=1,inplace=True)
# c=b.loc[idx[:,:,:,'A'],[YM_con_str,YM_opt_str]].droplevel(3)
# d=b.loc[idx[:,:,:,'B'],[YM_con_str,YM_opt_str]].droplevel(3)
# e=(d-c)/c

# fig, ax1 = plt.subplots()
# ax = sns.scatterplot(data=b.loc['UVR',YM_opt_str].reset_index(), 
#                      x='Kind', hue='Rot', y=YM_opt_str, ax=ax1)
# fig.suptitle('')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

b=cs.copy(deep=True)
b.index=pd.MultiIndex.from_frame(b[['Type','Kind','Vers','Rot']])
c=b[[YM_con_str,YM_opt_str]]
c=pd.concat([b[YM_con_str],
             # b.loc(axis=1)[b.columns.str.contains(r'E_inc_R_D2.*wt_meanwoso')]],
# Fixed 25.10.22
             b.loc(axis=1)[b.columns.str.contains(r'E_inc_F_D2.*wt_meanwoso')]],
            axis=1)
df=pd.melt(c.reset_index(), id_vars=['Type','Kind','Rot'], value_vars=c.columns)
df.rename({'variable':'Method'},axis=1,inplace=True)
df.Method=df.Method.str.split('_',expand=True)[3]

fig, ax = plt.subplots(ncols=1,nrows=2,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Youngs Modulus for different materials, geometries and methods',
             fontsize=12, fontweight="bold")
axt = sns.barplot(data=df.loc[df.Type=='UVR'], 
                     x='Kind', hue='Method', y='value', 
                     ax=ax[0], errwidth=1, capsize=.1)
ax[0].set_title('UV-Resine (LCD)')
ax[0].set_xlabel('Specimen geometry type / -')
ax[0].set_ylabel('E / MPa')
axt = sns.barplot(data=df.loc[df.Type=='PLA'], 
                     x='Kind', hue='Method', y='value', 
                     ax=ax[1], errwidth=1, capsize=.1)
ax[1].set_title('PLA (FDM)')
ax[1].set_xlabel('Specimen geometry type / -')
ax[1].set_ylabel('E / MPa')
plt.savefig(out_verf+'-E_Geo_Met-bar.pdf')
plt.savefig(out_verf+'-E_Geo_Met-bar.png')
plt.show()

fig, ax = plt.subplots(ncols=1,nrows=2,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Youngs Modulus for different materials, geometries and methods',
             fontsize=12, fontweight="bold")
# axt = sns.pointplot(data=df.loc[df.Type=='UVR'], 
#                      x='Kind', hue='Method', y='value', 
#                      ax=ax[0], join=True, dodge=0.2, scale=1,
#                      errorbar=("pi", 100), errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[0], data=df.loc[df.Type=='UVR'],
                 x='Kind', hue='Method', y='value',
                 dodge=0.2, join=True, 
                 markers='o',scale=1,barsabove=True, capsize=4,
                 controlout=False)
ax[0].set_title('UV-Resine (LCD)')
ax[0].set_xlabel('Specimen geometry type / -')
ax[0].set_ylabel('E / MPa')
# axt = sns.pointplot(data=df.loc[df.Type=='PLA'], 
#                      x='Kind', hue='Method', y='value', 
#                      ax=ax[1], join=True, dodge=0.2, scale=1,
#                      errorbar=("pi", 100), errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[1], data=df.loc[df.Type=='PLA'],
                 x='Kind', hue='Method', y='value',
                 dodge=0.2, join=True, 
                 markers='o',scale=1,barsabove=True, capsize=4,
                 controlout=False)
ax[1].set_title('PLA (FDM)')
ax[1].set_xlabel('Specimen geometry type / -')
ax[1].set_ylabel('E / MPa')
plt.savefig(out_verf+'-E_Geo_Met-per.pdf')
plt.savefig(out_verf+'-E_Geo_Met-per.png')
plt.show()

# Anpassen pi -> minmax
fig, ax1 = plt.subplots()
ax1.grid(True)
df=pd.melt(cs, id_vars=['Type','Kind'], value_vars=YM_opt_str)
df.sort_values(['Type','Kind'],inplace=True)
fig.suptitle('Youngs Modulus for different materials and geometries\n (Method D2Mgwt)',
             fontsize=12, fontweight="bold")
# ax = sns.boxplot(x="Kind", y="value", hue="Type", data=df, ax=ax1, 
#                   showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="Origin_sshort", y="value", hue="variable",
#                     data=df, ax=ax1, dodge=True, edgecolor="black",
#                     linewidth=.5, alpha=.5, size=2)
axt = sns.pointplot(data=df.loc[(df.Type=='PLA')+(df.Type=='ACRYL')], 
                      x='Kind', hue='Type', y='value', 
                      ax=ax1, join=True, dodge=0.2, scale=1, legend=False,
                      errorbar=("pi", 100), errwidth=1, capsize=.1)
# errc=sns_ppwMMeb(ax=ax1, data=df.loc[df.Type=='ACRYL'],
#                  x='Kind', y='value',
#                  dodge=0.2, join=True, 
#                  palette=[sns.color_palette("tab10")[1]],
#                  markers='o',scale=1,barsabove=True, capsize=4,
#                  controlout=False)
# errc=sns_ppwMMeb(ax=ax1, data=df.loc[df.Type=='PLA'],
#                  x='Kind', y='value',
#                  dodge=0.2, join=True, 
#                  palette=[sns.color_palette("tab10")[1]],
#                  markers='o',scale=1,barsabove=True, capsize=4,
#                  controlout=False)
ax1.set_xlabel('Kind / -')
ax1.set_ylabel('PMMA and PLA: $E$ / MPa')
ax2=ax1.twinx()
axt = sns.pointplot(data=df.loc[df.Type=='UVR'], 
                      x='Kind', hue='Type', y='value', 
                      ax=ax2, join=True, dodge=0.2, legend=False,scale=1,
                      palette=[sns.color_palette("tab10")[3]],
                      errorbar=("pi", 100), errwidth=1, capsize=.1)
# errc=sns_ppwMMeb(ax=ax2, data=df.loc[df.Type=='UVR'],
#                  x='Kind', y='value',
#                  dodge=0.2, join=True, 
#                  palette=[sns.color_palette("tab10")[3]],
#                  markers='o',scale=1,barsabove=True, capsize=4,
#                  controlout=False)
ax2.set_ylabel('UV-Resine: $E$ / MPa')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.84),title='Type')
ax1.legend([],[], frameon=False)
ax2.legend([],[], frameon=False)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_verf+'-E_Geo_opt-per.pdf')
plt.savefig(out_verf+'-E_Geo_opt-per.png')
plt.show()




c=b[YMMetoI]
d=c.groupby(['Type','Kind']).mean().loc(axis=0)[idx[:,'01']].droplevel(1)
e=(c-d)/d
df=pd.melt(e.reset_index(),id_vars=['Type','Kind','Rot'],
           value_vars=YMMetoI)
df['variable']=df.variable.str.split('_',expand=True)[3]
df.rename({'variable':'Method','Rot':'Orientation','value':'E'},axis=1,inplace=True)
fig, ax = plt.subplots(ncols=1,nrows=2,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
axt = sns.scatterplot(data=df.loc[df.Type=='UVR'], 
                     x='Kind', hue='Method', y='E',style='Orientation',
                     markers=['P','X'], ax=ax[0],s=50, alpha=0.7)
ax[0].set_title('UV-Resine (LCD)')
ax[0].set_ylabel('$D_{E,Geo-\overline{01}}$')
ax[0].set_xlabel('Kind of specimen geometry')
axt = sns.scatterplot(data=df.loc[df.Type=='PLA'], 
                     x='Kind', hue='Method', y='E',style='Orientation',
                     markers=['P','X'], ax=ax[1],s=50, alpha=0.7)
ax[1].set_title('PLA (FDM)')
ax[1].set_ylabel('$D_{E,Geo-\overline{01}}$')
ax[1].set_xlabel('Kind of specimen geometry')
fig.suptitle('%s\nYoungs-modulus deviation to mean of rectangle (01)'%name_Head,
             fontsize=12, fontweight="bold")
plt.savefig(out_verf+'-E_D01Geo_Met-sca.pdf')
plt.savefig(out_verf+'-E_D01Geo_Met-sca.png')
plt.show()

b=cs.copy(deep=True)
b.index=pd.MultiIndex.from_frame(b[['Type','Kind','Vers','Rot']])
c=b.loc[idx[:,:,:,'A'],YMMetoI].droplevel(3)
d=b.loc[idx[:,:,:,'B'],YMMetoI].droplevel(3)
e=(d-c)/c
e.sort_index(level=0,inplace=True)
df=pd.melt(e.reset_index(),id_vars=['Type','Kind'],
           value_vars=YMMetoI)
df['variable']=df.variable.str.split('_',expand=True)[3]
df.rename({'variable':'Method','value':'E'},axis=1,inplace=True)

fig, ax = plt.subplots(constrained_layout=True)
axt = sns.scatterplot(data=df, 
                     x='Kind', hue='Method', y='E', style='Type',
                     markers=['D','P','X'], ax=ax,s=50, alpha=0.7)
ax.set_ylabel('$(E_{B}-E_{A})/E_{A}$ / -')
ax.set_xlabel('Kind of specimen geometry')
fig.suptitle('%s\nYoungs-modulus deviation by orientation'%name_Head,
             fontsize=12, fontweight="bold")
plt.savefig(out_verf+'-E_DABGeo_Met-sca.pdf')
plt.savefig(out_verf+'-E_DABGeo_Met-sca.png')
plt.show()


df=E_inc_m_M.copy(deep=True)
df.columns=E_inc_m_M.columns.to_series().apply(lambda x: '{0}{1}{2}{3}'.format(*x)).values
id_dr = df.columns.str.contains('fu')
df = df.loc(axis=1)[np.invert(id_dr)]
df0=df.loc[cs.Type=='ACRYL']
df0=df0.unstack(level=0).reset_index(level=1, drop=True).reset_index(name='data')
df1=df.loc[cs.Type=='UVR']
df1=df1.unstack(level=0).reset_index(level=1, drop=True).reset_index(name='data')
df2=df.loc[cs.Type=='PLA']
df2=df2.unstack(level=0).reset_index(level=1, drop=True).reset_index(name='data')

fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Youngs Modulus (only M) for different methods and materials',
             fontsize=12, fontweight="bold")
ax[0].grid(True)
h = sns.barplot(x="index", y="data", data=df0, ax=ax[0],
                 errwidth=1, capsize=.1)
ax[0].set_title('PMMA')
ax[0].set_ylabel('$E_{M}$ / MPa')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[1].grid(True)
h = sns.barplot(x="index", y="data", data=df1, ax=ax[1],
                 errwidth=1, capsize=.1)
ax[1].set_title('UV-Resine (LCD)')
ax[1].set_ylabel('$E_{M}$ / MPa')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[2].grid(True)
h = sns.barplot(x="index", y="data", data=df2, ax=ax[2],
                 errwidth=1, capsize=.1)
ax[2].set_title('PLA (FDM)')
ax[2].set_ylabel('$E_{M}$ / MPa')
ax[2].set_xlabel('Evaluation method')
ax[2].tick_params(axis='x', labelrotation=90, labelsize=8)
plt.savefig(out_verf+'-E_Mat_MMet-bar.pdf')
plt.savefig(out_verf+'-E_Mat_MMet-bar.png')
plt.show()

fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Youngs Modulus (only M) for different methods and materials',
             fontsize=12, fontweight="bold")
ax[0].grid(True)
h = sns.boxplot(x="index", y="data", data=df0, ax=ax[0], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                           "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax[0].set_title('PMMA')
ax[0].set_ylabel('$E_{M}$ / MPa')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[1].grid(True)
h = sns.boxplot(x="index", y="data", data=df1, ax=ax[1], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                           "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax[1].set_title('UV-Resine (LCD)')
ax[1].set_ylabel('$E_{M}$ / MPa')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[2].grid(True)
h = sns.boxplot(x="index", y="data", data=df2, ax=ax[2], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                           "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax[2].set_title('PLA (FDM)')
ax[2].set_ylabel('$E_{M}$ / MPa')
ax[2].set_xlabel('Evaluation method')
ax[2].tick_params(axis='x', labelrotation=90, labelsize=8)
plt.savefig(out_verf+'-E_Mat_MMet-box.pdf')
plt.savefig(out_verf+'-E_Mat_MMet-box.png')
plt.show()




# df=pd.melt(pd.concat([c_E_inc_m['R'],cs['Type']],axis=1),
#             id_vars='Type',
#             value_vars=c_E_inc_m['R'].columns.difference(Methods_excl_names))
# Fixed 25.10.22
df=pd.melt(pd.concat([c_E_inc_m['F'],cs['Type']],axis=1),
            id_vars='Type',
            value_vars=c_E_inc_m['F'].columns.difference(Methods_excl_names))
# fig, ax1 = plt.subplots()
# ax1.grid(True)
# ax = sns.boxplot(x="variable", y="value", data=df, ax=ax1, 
#                   showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
#                                              "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# # ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
# ax1.set_title('%s\nYoungs Modulus of incremental determined YM \n(only refined range)'%name_Head)
# ax1.set_xlabel('Determination method / -')
# ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('$E$ / MPa')
# fig.suptitle('')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Youngs Modulus for different methods and materials',
             fontsize=12, fontweight="bold")
ax[0].grid(True)
h = sns.boxplot(x="variable", y="value", data=df[df.Type=='ACRYL'], ax=ax[0], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                           "markeredgecolor":"black", "markersize":"12","alpha":0.75})

ax[0].set_title('PMMA')
ax[0].set_ylabel('$E$ / MPa')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[1].grid(True)
h = sns.boxplot(x="variable", y="value", data=df[df.Type=='UVR'], ax=ax[1], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                           "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax[1].set_title('UV-Resine (LCD)')
ax[1].set_ylabel('$E$ / MPa')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[2].grid(True)
h = sns.boxplot(x="variable", y="value", data=df[df.Type=='PLA'], ax=ax[2], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                             "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax[2].set_title('PLA (FDM)')
ax[2].set_ylabel('$E$ / MPa')
ax[2].set_xlabel('Evaluation method')
ax[2].tick_params(axis='x', labelrotation=90, labelsize=8)
plt.savefig(out_verf+'-E_Mat_allMet-box.pdf')
plt.savefig(out_verf+'-E_Mat_allMet-box.png')
plt.show()






#%%%% Test F-R u-l
data_read = pd.HDFStore(path+name_in+'.h5','r')
dfEt=data_read['Add_/E_inc_df']
dfVIP=data_read['Add_/VIP']
data_read.close()

cEt=pd.DataFrame([])
for i in dfEt.index:
    tmpE=dfEt.loc[i]
    tmpV=dfVIP.loc[i]
    Evar=pd.DataFrame([],columns=['l_F','l_R','u_F','u_R'])
    Evar.loc['con']=pd.Series({'l_F':(tmpV.loc['FlA',0],tmpV.loc['FlB',0]),
                      'l_R':(tmpV.loc['RlA',0],tmpV.loc['RlB',0]),
                      'u_F':(tmpV.loc['FuA',0],tmpV.loc['FuB',0]),
                      'u_R':(tmpV.loc['RuA',0],tmpV.loc['RuB',0])})
    Evar.loc['opt']=pd.Series({'l_F':(tmpV.loc['FlA',1],tmpV.loc['FlB',1]),
                      'l_R':(tmpV.loc['RlA',1],tmpV.loc['RlB',1]),
                      'u_F':(tmpV.loc['FuA',1],tmpV.loc['FuB',1]),
                      'u_R':(tmpV.loc['RuA',1],tmpV.loc['RuB',1])})
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
    del tmpE,tmpV, Evar, cols_con, tmpEr
    
cEt.index=pd.MultiIndex.from_frame(cs.loc[cEt.index,['Type','Kind','Vers','Rot']])
cEt.columns = cEt.columns.str.split('_', expand=True)
cEt.columns.names=['Load','Range','Method']
cEt=cEt.reorder_levels(['Method','Load','Range'],axis=1).sort_index(axis=1,level=0)
cEt.to_csv(out_full+'-E_lu_FR.csv',sep=';')
cEE=cEt.loc(axis=1)[['A0Al','D2Mgwt'],:,:]
tmp=cEt.columns.levels[0].to_series()
Methods_incl_M=tmp[(tmp.str[2]=='M')*(~tmp.str.contains('fu'))].to_list()

#%%%% Logs
Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)
Evac.MG_strlog("\nWith load/unload and fixed/refined", log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Overview:",3), 
               log_mg, printopt=False)
a=cEE.groupby('Type').agg(['mean','std']).T
Evac.MG_strlog(Evac.str_indent(a.to_string(),3), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Deviation to Rectangle-mean (01) (E-E_01)/E_01)):",5),
               log_mg, printopt=False)
d=cEE.groupby(['Type','Kind']).mean().loc(axis=0)[idx[:,'01']].droplevel(1)
e=(cEE-d)/d
Evac.MG_strlog(Evac.str_indent(e.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("Deviation by rotation:",5),
               log_mg, printopt=False)
c=cEE.loc[idx[:,:,:,'A']].droplevel(3)
d=cEE.loc[idx[:,:,:,'B']].droplevel(3)
e=(d-c)/c
Evac.MG_strlog(Evac.str_indent(e.to_string(),5), log_mg, printopt=False)


Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)
Evac.MG_strlog("\nAbsolute deviation from chosen Method (D2Mgwt):", log_mg, printopt=False)
Evac.MG_strlog("\nPMMA (Acryl):", log_mg, printopt=False)
tmp=pd.Series(cEt.columns.droplevel([1,2])).drop_duplicates()
# tmp=tmp.loc[tmp.apply(lambda x: ((x[2]=='M')or(x[2]=='C'))and(x[1]!='4'))].values
tmp=tmp.loc[tmp.apply(lambda x: ((x[2]=='M')or(x[2]=='C')))].values
a=cEt.loc['ACRYL'].loc(axis=1)['D2Mgwt',:,'F'].groupby(axis=1,level=0).mean()
b=cEt.loc['ACRYL'].loc(axis=1)[tmp,:,'F']
c=b.subtract(a.D2Mgwt,axis=0).abs().divide(a.D2Mgwt,axis=0).groupby(axis=1,level=0).mean()
d=c.mean().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\nPLA only rectangle (01):", log_mg, printopt=False)
a=cEt.loc['PLA','01'].loc(axis=1)['D2Mgwt',:,'F'].groupby(axis=1,level=0).mean()
b=cEt.loc['PLA','01'].loc(axis=1)[tmp,:,'F']
c=b.subtract(a.D2Mgwt,axis=0).abs().divide(a.D2Mgwt,axis=0).groupby(axis=1,level=0).mean()
d=c.mean().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\nPLA all geometrys:", log_mg, printopt=False)
a=cEt.loc['PLA'].loc(axis=1)['D2Mgwt',:,'F'].groupby(axis=1,level=0).mean()
b=cEt.loc['PLA'].loc(axis=1)[tmp,:,'F']
c=b.subtract(a.D2Mgwt,axis=0).abs().divide(a.D2Mgwt,axis=0).groupby(axis=1,level=0).mean()
d=c.mean().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)

Evac.MG_strlog("\n\n "+"-"*100, log_mg, printopt=False)
Evac.MG_strlog("\nMean absolute deviation from chosen Method (D2Mgwt)-loading range only:", log_mg, printopt=False)
Evac.MG_strlog("\nPMMA (Acryl):", log_mg, printopt=False)
tmp=pd.Series(cEt.columns.droplevel([1,2])).drop_duplicates()
# tmp=tmp.loc[tmp.apply(lambda x: ((x[2]=='M')or(x[2]=='C'))and(x[1]!='4'))].values
tmp=tmp.loc[tmp.apply(lambda x: ((x[2]=='M')or(x[2]=='C')))].values
a=cEt.loc['ACRYL'].loc(axis=1)['D2Mgwt','l','F']
b=cEt.loc['ACRYL'].loc(axis=1)[tmp,'l','F']
c=b.subtract(a,axis=0).abs().divide(a,axis=0).groupby(axis=1,level=0).mean()
d=c.mean().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\nPLA only rectangle (01):", log_mg, printopt=False)
a=cEt.loc['PLA','01'].loc(axis=1)['D2Mgwt','l','F']
b=cEt.loc['PLA','01'].loc(axis=1)[tmp,'l','F']
c=b.subtract(a,axis=0).abs().divide(a,axis=0).groupby(axis=1,level=0).mean()
d=c.mean().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\nPLA all geometrys:", log_mg, printopt=False)
a=cEt.loc['PLA'].loc(axis=1)['D2Mgwt','l','F']
b=cEt.loc['PLA'].loc(axis=1)[tmp,'l','F']
c=b.subtract(a,axis=0).abs().divide(a,axis=0).groupby(axis=1,level=0).mean()
d=c.mean().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)

Evac.MG_strlog("\nMaximum absolute deviation from chosen Method (D2Mgwt)-loading range only:", log_mg, printopt=False)
Evac.MG_strlog("\nPMMA (Acryl):", log_mg, printopt=False)
tmp=pd.Series(cEt.columns.droplevel([1,2])).drop_duplicates()
# tmp=tmp.loc[tmp.apply(lambda x: ((x[2]=='M')or(x[2]=='C'))and(x[1]!='4'))].values
tmp=tmp.loc[tmp.apply(lambda x: ((x[2]=='M')or(x[2]=='C')))].values
a=cEt.loc['ACRYL'].loc(axis=1)['D2Mgwt','l','F']
b=cEt.loc['ACRYL'].loc(axis=1)[tmp,'l','F']
c=b.subtract(a,axis=0).abs().divide(a,axis=0).groupby(axis=1,level=0).max()
d=c.max().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\nPLA only rectangle (01):", log_mg, printopt=False)
a=cEt.loc['PLA','01'].loc(axis=1)['D2Mgwt','l','F']
b=cEt.loc['PLA','01'].loc(axis=1)[tmp,'l','F']
c=b.subtract(a,axis=0).abs().divide(a,axis=0).groupby(axis=1,level=0).max()
d=c.max().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)
Evac.MG_strlog("\nPLA all geometrys:", log_mg, printopt=False)
a=cEt.loc['PLA'].loc(axis=1)['D2Mgwt','l','F']
b=cEt.loc['PLA'].loc(axis=1)[tmp,'l','F']
c=b.subtract(a,axis=0).abs().divide(a,axis=0).groupby(axis=1,level=0).max()
d=c.max().sort_values()
Evac.MG_strlog(Evac.str_indent(d.to_string(),5), log_mg, printopt=False)


#%%%% Plots
df=pd.melt(cEE.reset_index(level=0),id_vars='Type')
df['Dir_Range']=df['Load']+df['Range']

fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Youngs Modulus for different methods and materials',
             fontsize=12, fontweight="bold")
h = sns.boxplot(x="Method", y="value", hue='Dir_Range',
                data=df[df.Type=='ACRYL'], ax=ax[0], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                           "markeredgecolor":"black", "markersize":"12","alpha":0.75})
Evac.tick_legend_renamer(ax=ax[0],
                         renamer={'lF':'load and fixed','lR':'load and refined',
                                  'uF':'unload and fixed','uR':'unload and refined',},
                         title='Loading and evaluation range')
ax[0].set_title('PMMA')
ax[0].set_ylabel('$E$ / MPa')
ax[0].set_xlabel('Evaluation method')
h = sns.boxplot(x="Method", y="value", hue='Dir_Range',
                data=df[df.Type=='UVR'], ax=ax[1], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                           "markeredgecolor":"black", "markersize":"12","alpha":0.75})
Evac.tick_legend_renamer(ax=ax[1],
                         renamer={'lF':'load and fixed','lR':'load and refined',
                                  'uF':'unload and fixed','uR':'unload and refined',},
                         title='Loading and evaluation range')
ax[1].set_title('UV-Resine (LCD)')
ax[1].set_ylabel('$E$ / MPa')
ax[1].set_xlabel('Evaluation method')
h = sns.boxplot(x="Method", y="value", hue='Dir_Range',
                data=df[df.Type=='PLA'], ax=ax[2], 
                showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
                                             "markeredgecolor":"black", "markersize":"12","alpha":0.75})
Evac.tick_legend_renamer(ax=ax[2],
                         renamer={'lF':'load and fixed','lR':'load and refined',
                                  'uF':'unload and fixed','uR':'unload and refined',},
                         title='Loading and evaluation range')
ax[2].set_title('PLA (FDM)')
ax[2].set_ylabel('$E$ / MPa')
ax[2].set_xlabel('Evaluation method')
plt.savefig(out_verf+'-E_Mat_luFr-box.pdf')
plt.savefig(out_verf+'-E_Mat_luFr-box.png')
plt.show()

# Anpassen pi -> minmax
df=pd.melt(cEE.reset_index(level=[0,1]),id_vars=['Type','Kind'])
df['Dir_Range']=df['Load']+df['Range']

fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Youngs Modulus for different materials and geometries\n (Method D2Mgwt)',
             fontsize=12, fontweight="bold")
axt = sns.pointplot(data=df.loc[(df.Type=='ACRYL')], 
                     x='Kind', hue='Dir_Range', y='value', 
                     ax=ax[0], join=True, dodge=True, scale=1, legend=False,
                     markers=['o','s','P','X'],
                     linestyles=['-','-.','--',':'],
                     errorbar=("pi", 100), errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=ax[0],
                         renamer={'lF':'load and fixed','lR':'load and refined',
                                  'uF':'unload and fixed','uR':'unload and refined',},
                         title='Loading and evaluation range')
ax[0].set_title('PMMA')
ax[0].set_xlabel('Kind / -')
ax[0].set_ylabel('$E$ / MPa')
axt = sns.pointplot(data=df.loc[(df.Type=='UVR')], 
                     x='Kind', hue='Dir_Range', y='value', 
                     ax=ax[1], join=True, dodge=True, scale=1, legend=False,
                     markers=['o','s','P','X'],
                     linestyles=['-','-.','--',':'],
                     errorbar=("pi", 100), errwidth=1, capsize=.1)
ax[1].legend([],[], frameon=False)
ax[1].set_title('UV-Resine (LCD)')
ax[1].set_xlabel('Kind / -')
ax[1].set_ylabel('$E$ / MPa')
axt = sns.pointplot(data=df.loc[(df.Type=='PLA')], 
                     x='Kind', hue='Dir_Range', y='value', 
                     ax=ax[2], join=True, dodge=True, scale=1, legend=False,
                     markers=['o','s','P','X'],
                     linestyles=['-','-.','--',':'],
                     errorbar=("pi", 100), errwidth=1, capsize=.1)
ax[2].legend([],[], frameon=False)
ax[2].set_title('PLA (FDM)')
ax[2].set_xlabel('Kind / -')
ax[2].set_ylabel('$E$ / MPa')
plt.savefig(out_verf+'-E_Geo_opt_luFr-per.pdf')
plt.savefig(out_verf+'-E_Geo_opt_luFr-per.png')
plt.show()

# Anpassen pi -> minmax
df=pd.melt(cEt.loc(axis=1)[['A0Al']+Methods_incl_M,:,'F'].reset_index(level=0),id_vars='Type')
df=df.loc[df['value']>0]
fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Youngs Modulus for different methods and materials',
             fontsize=12, fontweight="bold")
ax[0].grid(True)
axt = sns.pointplot(data=df[df.Type=='ACRYL'], 
                     x='Method', y='value', hue='Load',
                     ax=ax[0], join=False, dodge=True, legend=False,scale=1,
                     markers=['o','P'],
                     errorbar=("pi", 100), errwidth=1, capsize=.1)
ax[0].axvline(0.5,0,1, color='grey',ls='--')
Evac.tick_legend_renamer(ax=ax[0],
                         renamer={'l':'load','u':'unload'},
                         title='Loading')
ax[0].set_title('PMMA')
ax[0].set_ylabel('$E_{M}$ / MPa')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[1].grid(True)
axt = sns.pointplot(data=df[df.Type=='UVR'], 
                     x='Method', y='value', hue='Load',
                     ax=ax[1], join=False, dodge=True, legend=False,scale=1,
                     markers=['o','P'],
                     errorbar=("pi", 100), errwidth=1, capsize=.1)
ax[1].axvline(0.5,0,1, color='grey',ls='--')
Evac.tick_legend_renamer(ax=ax[1],
                         renamer={'l':'load','u':'unload'},
                         title='Loading')
ax[1].set_title('UV-Resine (LCD)')
ax[1].set_ylabel('$E_{M}$ / MPa')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=90, labelsize=8)
ax[2].grid(True)
axt = sns.pointplot(data=df[df.Type=='PLA'], 
                     x='Method', y='value', hue='Load',
                     ax=ax[2], join=False, dodge=True, legend=False,scale=1,
                     markers=['o','P'],
                     errorbar=("pi", 100), errwidth=1, capsize=.1)
ax[2].axvline(0.5,0,1, color='grey',ls='--')
Evac.tick_legend_renamer(ax=ax[2],
                         renamer={'l':'load','u':'unload'},
                         title='Loading')
ax[2].set_title('PLA (FDM)')
ax[2].set_ylabel('$E_{M}$ / MPa')
ax[2].set_xlabel('Evaluation method')
ax[2].tick_params(axis='x', labelrotation=90, labelsize=8)
plt.savefig(out_verf+'-E_Mat_MMet_lu-per.pdf')
plt.savefig(out_verf+'-E_Mat_MMet_lu-per.png')
plt.show()
#%%%% Plots SUP
figsize_sup=(16.0/2.54, 22.0/2.54)
# Methods_chosen=['A0Al',
#                 'A2Al','B1Al','C2Al','D1Aguw','D1Agwt','D2Aguw','D2Agwt','G3Ag',
#                 'A2Sl','B1Sl','C2Sl','D1Sguw','D1Sgwt','D2Sguw','D2Sgwt','G3Sg',
#                 'A2Ml','B1Ml','C2Ml','C2Cl','D1Mguw','D1Mgwt','D2Mguw','D2Mgwt','G3Mg','G3Cg']
Methods_chosen=['A0Al',
                'C2Al','D1Aguw','D1Agwt','D2Aguw','D2Agwt',
                'C2Sl','D1Sguw','D1Sgwt','D2Sguw','D2Sgwt',
                'C2Ml','C2Cl','D1Mguw','D1Mgwt','D2Mguw','D2Mgwt']
Methods_chosen_short=['A0Al','C2Ml','D2Mgwt']


tmp=pd.Series(cEt.columns.droplevel([1,2])).drop_duplicates()
tmp=tmp.loc[tmp.apply(lambda x: ((x[2]=='M')or(x[2]=='C'))and(x[4:]!='fu'))].values
a=cEt.loc(axis=0)[:,'01'].loc(axis=1)['D2Mgwt',:,'F'].groupby(axis=1,level=0).mean()
b=cEt.loc(axis=0)[:,'01'].loc(axis=1)[tmp,:,'F']
c=b.subtract(a.D2Mgwt,axis=0).divide(a.D2Mgwt,axis=0)
df=pd.melt(c.reset_index(level=[0,1]),id_vars=['Type','Kind'])
fig, ax = plt.subplots(ncols=1,nrows=2,
                       figsize=figsize_sup, constrained_layout=True)
# fig.suptitle('Deviation of methods with complete exclusion of falsifiieng influences (**M*/**C*)',
#               fontsize=12, fontweight="bold")
# axt = sns.pointplot(data=df[df.Type=='ACRYL'], 
#                       x='Method', y='value',
#                       ax=ax[0], join=False, dodge=True, legend=False,scale=1,
#                       errorbar=("pi", 100), errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[0], data=df[df.Type=='ACRYL'],
                 x='Method', y='value', hue=None,
                 dodge=0.2, markers=['o','P'],scale=1,barsabove=True, capsize=2,
                 controlout=False)
ax[0].set_title('PMMA')
ax[0].set_ylabel('$D_{E,Method-D2Mgwt}=(E_{Method}-E_{D2Mgwt})/E_{D2Mgwt}$')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=90, labelsize=8)
a=cEt.loc['PLA'].loc(axis=1)['D2Mgwt',:,'F'].groupby(axis=1,level=0).mean()
b=cEt.loc['PLA'].loc(axis=1)[tmp,:,'F']
c=b.subtract(a.D2Mgwt,axis=0).divide(a.D2Mgwt,axis=0)
df=pd.melt(c.reset_index(level=[0,1,2]),id_vars=['Kind','Vers','Rot'])
df2=df.loc[df.Kind=='01'].copy(deep=True)
df['Geometry']='All'
df2['Geometry']='01'
df3=pd.concat([df2,df])
# axt = sns.pointplot(data=df3, 
#                       x='Method', y='value', hue='Geometry',
#                       ax=ax[1], join=False, dodge=True, legend=False,scale=1,
#                       markers=['o','P'],
#                       errorbar=("pi", 100), errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[1], data=df3,
                 x='Method', y='value', hue='Geometry',
                 dodge=0.2, markers=['o','P'],scale=1,barsabove=True, capsize=2,
                 controlout=False)
ax[1].set_title('PLA (FDM)')
ax[1].set_ylabel('$D_{E,Method-D2Mgwt}=(E_{Method}-E_{D2Mgwt})/E_{D2Mgwt}$')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=90, labelsize=8)
plt.savefig(out_verf+'-SUP-DE_choosen_MMet_lu-per.pdf')
plt.savefig(out_verf+'-SUP-DE_choosen_MMet_lu-per.png')
plt.show()





df=pd.melt(cEt.loc(axis=1)[Methods_chosen,:,'F'].reset_index(level=0),id_vars='Type')
df=df.loc[df['value']>0]
fig, ax = plt.subplots(ncols=1,nrows=2,
                       figsize=figsize_sup, constrained_layout=True)
# fig.suptitle('Youngs Modulus of chosen methods',
#              fontsize=12, fontweight="bold")
# axt = sns.pointplot(data=df[df.Type=='ACRYL'], 
#                      x='Method', y='value', hue='Load',
#                      ax=ax[0], join=False, dodge=True, legend=False,scale=1,
#                      markers=['o','P'],
#                      errorbar=("pi", 100), errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[0], data=df[df.Type=='ACRYL'],
                 x='Method', y='value', hue='Load',
                 dodge=0.2, markers=['o','P'],scale=1,barsabove=True, capsize=2,
                 controlout=True)
Evac.tick_legend_renamer(ax=ax[0],
                         renamer={'l':'load','u':'unload'},
                         title='Loading')
ax[0].set_title('PMMA')
ax[0].set_ylabel('$E$ / MPa')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=90, labelsize=8)
# axt = sns.pointplot(data=df[df.Type=='PLA'], 
#                      x='Method', y='value', hue='Load',
#                      ax=ax[1], join=False, dodge=True, legend=False,scale=1,
#                      markers=['o','P'],
#                      errorbar=("pi", 100), errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[1], data=df[df.Type=='PLA'],
                 x='Method', y='value', hue='Load',
                 dodge=0.2, markers=['o','P'],scale=1,barsabove=True, capsize=2,
                 controlout=True)
Evac.tick_legend_renamer(ax=ax[1],
                         renamer={'l':'load','u':'unload'},
                         title='Loading')
ax[1].set_title('PLA (FDM)')
ax[1].set_ylabel('$E$ / MPa')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=90, labelsize=8)
for i in range(ax.size):
    # for j in [0.5,8.5,16.5]:
    for j in [0.5,5.5,10.5]:
        ax[i].axvline(j,0,1, color='grey',ls='--')
plt.savefig(out_verf+'-SUP-E_Mat_cMet_lu-per.pdf')
plt.savefig(out_verf+'-SUP-E_Mat_cMet_lu-per.png')
plt.show()




df = c_E_inc_m.loc(axis=1)['F',Methods_chosen].droplevel(0,axis=1).loc[cs.Type=='PLA']
df = pd.concat([c_inf,df],axis=1).corr(method=mcorr)
fig, ax = plt.subplots(nrows=2, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1],
                                    'height_ratios': [19, 10]},
                       figsize=figsize_sup, constrained_layout=True)
# fig.suptitle('Influence on incremental, in loading range,\ndetermined Youngs Modulus',
#              fontsize=12, fontweight="bold")
g=sns.heatmap(df.loc[c_inf_gad.columns.to_list()+['Density_app','fu','eu_opt','Wu_opt'],
                     Methods_chosen].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
for j in [1,6,11]:
    ax[0,0].axvline(j,0,1, color='white',ls='-')
ax[0,0].set_title('Spearman correlation coefficients')
ax[0,0].set_xlabel('Evaluation method')
ax[0,0].set_ylabel('Contributing factors')
ax[0,0].tick_params(axis='x', labelrotation=90, labelsize=5)
ax[0,0].tick_params(axis='y', labelsize=5)
ax[0,1].tick_params(axis='y', labelsize=5)
content=[[r'$h$ - Thickness','$\overline{X}$ - Arithmetic average'],
         [r'$b$ - Width', '$\Delta X$ - Difference between max. and min.'],
         [r'$A_{CS}$ - Cross-section area','$X_{mid}$ - Midspan position'],
         [r'$V$ - Volume','$X_{circle}$ - Approximated as circle'],
         [r'$I$ - Second moment of inertia','$X_{el}$ - Elastic range'],
         [r'$\kappa$ - Geometrical curvature','$X_{y}$ - Yield'],
         [r'$ind$ - Support indentation','$X_{u}$ - Ultimate'],
         [r'$\rho_{app}$ - Apparent density','$X_{opt}$ - Determination via DIC'],
         [r'$f$ - Strength','   '],
         [r'$\epsilon$ - Strain','  '],
         [r'$W$ - External work','  ']]
axt= ax[1,0].table(content, loc='center', cellLoc='left', rowLoc='center',
                   colWidths=[0.5,0.5],edges='open')
ax[1,0].set_title('Symbols')
axt.auto_set_font_size(False)
axt.set_fontsize(8)
ax[1,0].axis('off')
ax[1,1].axis('off')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_verf+'-SUP-Inf_El_PLA.pdf')
plt.savefig(out_verf+'-SUP-Inf_El_PLA.png')
plt.show()



d=cEt.groupby(['Type','Kind']).mean().loc(axis=0)[idx[:,'01']].droplevel(1)
e=(cEt-d)/d
df=pd.melt(e.loc(axis=1)[Methods_chosen_short,:,'F'].reset_index(),
           id_vars=['Type','Kind','Vers','Rot'])
df['Orientation and loading']=df.Rot+df.Load.replace({'l':' load','u':' unload'})
# df.sort_values('Orientation and Rotation')
# df.rename({'Rot':'Orientation','value':'E'},axis=1,inplace=True)
fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=figsize_sup, constrained_layout=True)
# axt = sns.scatterplot(data=df.loc[(df.Type=='PLA')*(df.Method=='A0Al')], 
#                      x='Kind', hue='Orientation and loading', y='value',style='Orientation and loading',
#                      markers=['o','s','P','X'], ax=ax[0], s=50, alpha=0.7)
axt = sns.pointplot(data=df.loc[(df.Type=='PLA')*(df.Method=='A0Al')], 
                     x='Kind', hue='Orientation and loading', y='value',style='Orientation and loading',
                     dodge=True, join=False, ax=ax[0],
                     markers=['o','s','P','X'], s=50, alpha=0.7)
ax[0].set_title('Method A0Al')
ax[0].set_ylabel('$D_{E,Geo-\overline{01}}$')
ax[0].set_xlabel('Kind of specimen geometry')
axt = sns.pointplot(data=df.loc[(df.Type=='PLA')*(df.Method=='C2Ml')], 
                     x='Kind', hue='Orientation and loading', y='value',style='Orientation and loading',
                     dodge=True, join=False, ax=ax[1],
                     markers=['o','s','P','X'], s=50, alpha=0.7)
ax[1].set_title('Method C2Ml')
ax[1].set_ylabel('$D_{E,Geo-\overline{01}}$')
ax[1].set_xlabel('Kind of specimen geometry')
axt = sns.pointplot(data=df.loc[(df.Type=='PLA')*(df.Method=='D2Mgwt')], 
                     x='Kind', hue='Orientation and loading', y='value',style='Orientation and loading',
                     dodge=True, join=False, ax=ax[2],
                     markers=['o','s','P','X'], s=50, alpha=0.7)
ax[2].set_title('Method D2Mgwt')
ax[2].set_ylabel('$D_{E,Geo-\overline{01}}$')
ax[2].set_xlabel('Kind of specimen geometry')
# fig.suptitle('%s\nYoungs-modulus deviation of PLA specimens to mean of rectangle (01)'%name_Head,
              # fontsize=12, fontweight="bold")
plt.savefig(out_verf+'-SUP-E_D01Geo_PLA-sca.pdf')
plt.savefig(out_verf+'-SUP-E_D01Geo_PLA-sca.png')
plt.show()


c=cEt.loc[idx[:,:,:,'A']].droplevel(3)
d=cEt.loc[idx[:,:,:,'B']].droplevel(3)
e=(d-c)/c
e.sort_index(level=0,inplace=True)
df=pd.melt(e.loc(axis=1)[Methods_chosen_short,:,'F'].reset_index(),
           id_vars=['Type','Kind','Vers'])
df.rename({'variable':'Method','Load':'Loading','value':'E'},axis=1,inplace=True)
df['Loading']=df.Loading.replace({'l':'load','u':'unload'})

fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=figsize_sup, constrained_layout=True)
axt = sns.pointplot(data=df.loc[(df.Type=='PLA')*(df.Method=='A0Al')], 
                     x='Kind', hue='Loading', y='E', style='Loading',
                     markers=['P','X'],
                     dodge=True, join=False,  ax=ax[0],s=50, alpha=0.7)
ax[0].set_title('Method A0Al')
ax[0].set_ylabel('$D_{E,B-A}$')
ax[0].set_xlabel('Kind of specimen geometry')
axt = sns.pointplot(data=df.loc[(df.Type=='PLA')*(df.Method=='C2Ml')], 
                     x='Kind', hue='Loading', y='E', style='Loading',
                     markers=['P','X'],
                     dodge=True, join=False,  ax=ax[1],s=50, alpha=0.7)
ax[1].set_title('Method C2Ml')
ax[1].set_ylabel('$D_{E,B-A}$')
ax[1].set_xlabel('Kind of specimen geometry')
axt = sns.pointplot(data=df.loc[(df.Type=='PLA')*(df.Method=='D2Mgwt')], 
                     x='Kind', hue='Loading', y='E', style='Loading',
                     markers=['P','X'], 
                     dodge=True, join=False, ax=ax[2],s=50, alpha=0.7)
ax[2].set_title('Method D2Mgwt')
ax[2].set_ylabel('$D_{E,B-A}$')
ax[2].set_xlabel('Kind of specimen geometry')
# fig.suptitle('%s\nYoungs-modulus deviation of PLA specimens by orientation'%name_Head,
#              fontsize=12, fontweight="bold")
plt.savefig(out_verf+'-SUP-E_DABGeo_PLA-sca.pdf')
plt.savefig(out_verf+'-SUP-E_DABGeo_PLA-sca.png')
plt.show()


fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=figsize_sup, constrained_layout=True)
df= c_C.loc[cs.Type=='PLA'].loc(axis=1)[:,Methods_chosen[:6]]
df = df.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
axt = sns.barplot(x="Method", y="data", hue='Location', data=df,
                 ax=ax[0], errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=axt,renamer={'g':'global','l':'midspan'},title='')
ax[0].set_title('Methods without elimination (**A*)')
# ax[0].set_ylabel('$D_{w}=(w_{analytical}-w_{measured})/w_{measured}$')
ax[0].set_ylabel('$D_{w,analytical-measured}$')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=0, labelsize=8)
df= c_C.loc[cs.Type=='PLA'].loc(axis=1)[:,Methods_chosen[6:11]]
df = df.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
axt = sns.barplot(x="Method", y="data", hue='Location', data=df,
                 ax=ax[1], errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=axt,renamer={'g':'global','l':'midspan'},title='')
ax[1].set_title('Methods with support indentation elimination (**S*)')
ax[1].set_ylabel('$D_{w,analytical-measured}$')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=0, labelsize=8)
df= c_C.loc[cs.Type=='PLA'].loc(axis=1)[:,Methods_chosen[11:]]
df = df.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
axt = sns.barplot(x="Method", y="data", hue='Location', data=df,
                 ax=ax[2], errwidth=1, capsize=.1)
Evac.tick_legend_renamer(ax=axt,renamer={'g':'global','l':'midspan'},title='')
ax[2].set_title('Methods with support indentation and shear deformation elimination (**M*|**C*)')
ax[2].set_ylabel('$D_{w,analytical-measured}$')
ax[2].set_xlabel('Evaluation method')
ax[2].tick_params(axis='x', labelrotation=0, labelsize=8)
plt.savefig(out_verf+'-SUP-w_Dam_PLA-sca.pdf')
plt.savefig(out_verf+'-SUP-w_Dam_PLA-sca.png')
plt.show()



fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=figsize_sup, constrained_layout=True)
df= c_C.loc[cs.Type=='PLA'].loc(axis=1)[:,Methods_chosen[:6]]
df = df.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
# axt = sns.barplot(x="Method", y="data", hue='Location', data=df,
#                  ax=ax[0], errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[0], data=df,
                 x="Method", y="data", hue='Location',
                 dodge=0.2, join=False, 
                 markers=['o','P'],scale=1,barsabove=True, capsize=4,
                 controlout=False)
Evac.tick_legend_renamer(ax=ax[0],renamer={'g':'global','l':'midspan'},title='')
ax[0].set_title('Methods without elimination (**A*)')
# ax[0].set_ylabel('$D_{w}=(w_{analytical}-w_{measured})/w_{measured}$')
ax[0].set_ylabel('$D_{w,analytical-measured}$')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=0, labelsize=8)
df= c_C.loc[cs.Type=='PLA'].loc(axis=1)[:,Methods_chosen[6:11]]
df = df.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
# axt = sns.barplot(x="Method", y="data", hue='Location', data=df,
#                  ax=ax[1], errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[1], data=df,
                 x="Method", y="data", hue='Location',
                 dodge=0.2, join=False, 
                 markers=['o','P'],scale=1,barsabove=True, capsize=4,
                 controlout=False)
Evac.tick_legend_renamer(ax=ax[1],renamer={'g':'global','l':'midspan'},title='')
ax[1].set_title('Methods with support indentation elimination (**S*)')
ax[1].set_ylabel('$D_{w,analytical-measured}$')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=0, labelsize=8)
df= c_C.loc[cs.Type=='PLA'].loc(axis=1)[:,Methods_chosen[11:]]
df = df.unstack(level=0).reset_index(level=2, drop=True)
df = df.reset_index(name='data')
# axt = sns.barplot(x="Method", y="data", hue='Location', data=df,
#                  ax=ax[2], errwidth=1, capsize=.1)
errc=sns_ppwMMeb(ax=ax[2], data=df,
                 x="Method", y="data", hue='Location',
                 dodge=0.2, join=False, 
                 markers=['o','P'],scale=1,barsabove=True, capsize=4,
                 controlout=False)
Evac.tick_legend_renamer(ax=ax[2],renamer={'g':'global','l':'midspan'},title='')
ax[2].set_title('Methods with support indentation and shear deformation elimination (**M*|**C*)')
ax[2].set_ylabel('$D_{w,analytical-measured}$')
ax[2].set_xlabel('Evaluation method')
ax[2].tick_params(axis='x', labelrotation=0, labelsize=8)
plt.savefig(out_verf+'-SUP-w_Dam_PLA-per.pdf')
plt.savefig(out_verf+'-SUP-w_Dam_PLA-per.png')
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
# df=pd.melt(c_E_inc_m['R'],
#            value_vars=c_E_inc_m['R'].columns)
# Fixed 25.10.22
df=pd.melt(c_E_inc_m['F'],
           value_vars=c_E_inc_m['F'].columns)
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
# ax1.set_ylim(0,5000)
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
# ax1.set_ylim(0,5000)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YM-E_inc_R-bar.pdf')
plt.savefig(out_full+'-YM-E_inc_R-bar.png')
plt.show()

# df = c_E_inc_m['R']
# Fixed 25.10.22
df = c_E_inc_m['F']
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
# ax1.set_ylim(0,3)
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
# ax1.set_ylim(0,3)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YMm-E_inc_R-bar.pdf')
plt.savefig(out_full+'-YMm-E_inc_R-bar.png')
plt.show()


# df = c_E_inc_m['R']
# Fixed 25.10.22
df = c_E_inc_m['F']
df = (df/df.mean(axis=0)).std(axis=0)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(data=df, ax=ax1)
ax1.set_title('%s\nStandard Deviation of Youngs Modulus per mean of method of incremental determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('$\sigma_{E_{Specimen}/\overline{E}_{Method}}$ / -')
# ax1.set_ylim(0.80,1.00)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YMmsd-E_inc_R-bar.pdf')
plt.savefig(out_full+'-YMmsd-E_inc_R-bar.png')
plt.show()




df = pd.DataFrame([])
# df['mean'] = c_E_inc_m['R'].mean()
# Fixed 25.10.22
df['mean'] = c_E_inc_m['F'].mean()
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
# ax = sns.boxplot(data=c_E_lsq_m['R'], ax=ax1, 
#                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# Fixed 25.10.22
ax = sns.boxplot(data=c_E_lsq_m['F'], ax=ax1, 
                 showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(x="variable", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYoungs Modulus of least-square determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
ax1.set_ylabel('$E_{lsq}$ / MPa')
# ax1.set_ylim(0,5000)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-YM-E_lsq_R-box.pdf')
plt.savefig(out_full+'-YM-E_lsq_R-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
# ax = sns.barplot(data=c_E_lsq_m['R'], ax=ax1, 
#                  errwidth=1, capsize=.1)
# Fixed 25.10.22
ax = sns.barplot(data=c_E_lsq_m['F'], ax=ax1, 
                 errwidth=1, capsize=.1)
ax1.set_title('%s\nYoungs Modulus of least-square determined YM \n(only refined range)'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.tick_params(axis='x', labelrotation=90)
# ax1.set_ylabel('YM_meanwoso / MPa')
ax1.set_ylabel('$E_{lsq}$ / MPa')
# ax1.set_ylim(0,5000)
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
# ax1.set_ylim(0,1)
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
# ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_F-bar.pdf')
plt.savefig(out_full+'-StdN-E_inc_F-bar.png')
plt.show()



# df=pd.melt(c_E_inc_stnorm['R'],
#            value_vars=c_E_inc_stnorm['R'].columns)
# Fixed 25.10.22
df=pd.melt(c_E_inc_stnorm['F'],
           value_vars=c_E_inc_stnorm['F'].columns)
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
# ax1.set_ylim(0,1)
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
# ax1.set_ylim(0,1)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-StdN-E_inc_R-bar.pdf')
plt.savefig(out_full+'-StdN-E_inc_R-bar.png')
plt.show()


df = pd.DataFrame([])
id_dr = c_E_inc_stnorm.columns.droplevel(0).str.contains('fu')
# df['mean'] = c_E_inc_stnorm.loc(axis=1)[:,idx[np.invert(id_dr)]].mean()['R']
# Fixed 25.10.22
df['mean'] = c_E_inc_stnorm.loc(axis=1)[:,idx[np.invert(id_dr)]].mean()['F']
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
# ax1.set_ylim(-0.5,4)
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
# ax1.set_ylim(0,1)
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
# ax1.set_ylim(0,1)
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
# ax1.set_ylim(0,1)
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
# ax1.set_ylim(0,1)
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
# ax1.set_ylim(-1,4)
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
# ax1.set_ylim(-1,4)
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
# ax1.set_ylim(-1,4)
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
# ax1.set_ylim(-1,4)
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
# ax1.set_ylim(-0.3,0.5)
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
# ax1.set_ylim(-0.01,0.5)
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
# ax[1,0].set_ylim(-0.01,0.11)
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
# ax1.set_ylim(-0.01,2)
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
# ax1.set_ylim(-0.01,2)
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
# tmp=ax[1,0].get_xlim()
# ax[1,0].hlines([0.11,0.11,0.11, 0.11, 0.11],
#               [-0.4, 0.6, 1.6, 11.6, 12.6],
#               [ 0.4, 1.4, 2.4, 12.4, 13.4],
#               colors='red', linestyles='solid', linewidth=4.0)
# ax[1,0].set_xlim(tmp)
# ax[1,0].set_ylim(-0.01,0.11)
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

fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
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
tmp=pd.Series(c_inf['geo_dcurve']*c_inf_add['geo_curve_mid'],name='delta')
axt = sns.histplot(pd.concat([c_inf_add_new['geo_curve_cim'],
                              c_inf_add['geo_curve_mid'],tmp],axis=1),
                    stat='count', bins=20, ax=ax[2], kde=True, legend=True)
Evac.tick_legend_renamer(ax[2],{'geo_curve_cim':'$\kappa_{circle}$',
                                'geo_curve_mid':'$\kappa_{mid}$',
                                'delta':'$\Delta \kappa_{span}$'})
# ax[2].set_title('Curvature radius per mean thickness of unloaded specimen',fontsize=10)
# ax[2].set_title('Curvature multiplied with mean thickness of unloaded specimen',fontsize=10)
ax[2].set_title('Curvature of unloaded specimen',fontsize=10)
# ax[2].set_xlabel('Curvature radius per thickness / mm/mm')
# ax[2].set_xlabel('Curvature x thickness / mm/mm')
ax[2].set_xlabel('Curvature / mm$^{-1}$')
ax[2].set_ylabel('Count')

fig.suptitle('Distribution of geometrical observations',
             fontsize=12, fontweight="bold")
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'01.png')
plt.savefig(out_supmat+'01.pdf')
plt.show()
plt.close(fig)


fig, ax = plt.subplots(ncols=1,nrows=3,
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)

ax[0].set_title('Curvature of unloaded specimen',fontsize=10)
axt = sns.histplot(c_inf_add[['geo_curve_cim','geo_curve_mid',
                              'geo_curve_max','geo_curve_min']],
                   stat='count', bins=20, ax=ax[0], kde=True, legend=True)
Evac.tick_legend_renamer(ax[0],{'geo_curve_cim': r'$\kappa_{mid,circle}$',
                                'geo_curve_mid':'$\kappa_{mid}$',
                                'geo_curve_max':'$\kappa_{max}$',
                                'geo_curve_min':'$\kappa_{min}$'})
ax[0].set_xlabel('Curvature / mm$^{-1}$')
ax[0].set_ylabel('Count')

tmp=c_inf_add[['geo_curve_cim','geo_curve_mid','geo_curve_max','geo_curve_min']]
tmp=tmp.mul(c_inf_geo['thickness_mean'],axis=0)
axt = sns.histplot(tmp[['geo_curve_cim','geo_curve_mid','geo_curve_max','geo_curve_min']],
                   stat='count', bins=20, ax=ax[1], kde=True, legend=True)
Evac.tick_legend_renamer(ax[1],{'geo_curve_cim':'$\kappa_{circle}x\overline{h}$',
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
tmp=pd.concat([tmp,1.0/c_inf_add_new[['geo_curve_cim','geo_curve_max']].abs().mul(c_inf_geo['thickness_mean'],axis=0)],axis=1)
tmp.columns=['geo_curve_mid','geo_curve_cim','geo_curve_max']
axt = sns.boxplot(data=tmp[['geo_curve_cim','geo_curve_mid','geo_curve_max']],
                  ax=ax[2], orient='h', showfliers=False, showmeans=True, 
                  meanprops={"marker":"|", "markerfacecolor":"white",
                             "markeredgecolor":"black", "markersize":"25","alpha":0.75})
ftxt=['Mimimal values:',
      '   $|R_{circle}|/\overline{h}$: %4.2f'%(tmp.geo_curve_cim.min()),
      '   $|R_{mid}|/h_{mid}$: %4.2f'%(tmp.geo_curve_mid.min()),
      '   $|R|_{min}/\overline{h}$: %4.2f'%(tmp.geo_curve_max.min())]
ax[2].text(10000,1.0,'\n'.join(ftxt),ha='left',va='center', 
           bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
Evac.tick_label_renamer(ax[2],{'geo_curve_cim':'$|R|_{circle}/\overline{h}$',
                               'geo_curve_mid':'$|R_{mid}|/h_{mid}$',
                               'geo_curve_max':'$|R|_{min}/\overline{h}$'}, axis='y')
ax[2].set_yticklabels(ax[2].get_yticklabels(), rotation=90, va="center")
ax[2].set_xlabel('Curvature radius per thickness / mm/mm')
ax[2].set_ylabel('')
fig.suptitle('Distribution of geometrical curvature observations',
             fontsize=12, fontweight="bold")
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'ADD_Curve.png')
plt.savefig(out_supmat+'ADD_Curve.pdf')
plt.show()
plt.close(fig)




fig, ax = plt.subplots(nrows=3, ncols=2, 
                       gridspec_kw={'width_ratios': [29, 1],
                                    'height_ratios': [14, 10, 10]},
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Influence on incremental, in refined range,\ndetermined Youngs Modulus',
             fontsize=12, fontweight="bold")
# g=sns.heatmap(c_E_inc_corr.loc[c_inf_gad.columns,c_E_inc_m['R'].columns].round(1),
# Fixed 25.10.22
g=sns.heatmap(c_E_inc_corr.loc[c_inf_gad.columns,c_E_inc_m['F'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax[0,0],cbar_ax=ax[0,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[0,0].set_title('Geometrical and additional')
ax[0,0].set_xlabel('Evaluation method')
ax[0,0].set_ylabel('Contributing factors')
ax[0,0].tick_params(axis='x', labelrotation=90, labelsize=5)
ax[0,0].tick_params(axis='y', labelsize=5)
ax[0,1].tick_params(axis='y', labelsize=5)
# g=sns.heatmap(c_E_inc_corr.loc[c_inf_mat.columns,c_E_inc_m['R'].columns].round(1),
# Fixed 25.10.22
g=sns.heatmap(c_E_inc_corr.loc[c_inf_mat.columns,c_E_inc_m['F'].columns].round(1),
              center=0, annot=True, annot_kws={"size":5, 'rotation':90},
              xticklabels=1, ax=ax[1,0],cbar_ax=ax[1,1])
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax[1,0].set_title('Material')
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
                                    'height_ratios': [14, 10, 10]},
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Influence on the moduli of elasticity determined without (**A*)\nand with (**S*) elimination of the support indentation',
             fontsize=12, fontweight="bold")
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
                                    'height_ratios': [14, 10, 10]},
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Influence on the moduli of elasticity determined without (**S*)\nand with (**M*) elimination of the shear deformation,\nwith support indentation eliminated in both cases',
             fontsize=12, fontweight="bold")
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
# tmp=ax[2,0].get_xlim()
# ax[2,0].hlines([0.225,0.225,0.225],
#                  [-0.4, 0.6, 11.6],
#                  [ 0.4, 1.4, 12.4],
#                  colors='red', linestyles='solid', linewidth=4.0)
# ax[2,0].set_xlim(tmp)
# ax[2,0].set_ylim(-0.001,0.23)
# t=Comp_E_inc_m_MS.loc(axis=1)[['A2l','A4l','F4gha']].agg(['mean',Evac.confidence_interval])
# ftxt=['Trimmed values:',
#       '   A2*l: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','A2l'],*t.loc['confidence_interval','A2l']),
#       '   A4*l: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','A4l'],*t.loc['confidence_interval','A4l']),
#       '   F4g*ha: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','F4gha'],*t.loc['confidence_interval','F4gha'])]
# ax[2,0].text(2.6,0.16,'\n'.join(ftxt),ha='left',va='bottom', 
#            bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
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
                                    'height_ratios': [14, 10, 10]},
                       figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Influence on the moduli of elasticity determined without (**A*)\nand with (**M*) elimination of the shear deformation,\nas well as the support indentation',
             fontsize=12, fontweight="bold")
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
# ax[2,0].hlines([1.35,1.35],
#                  [-0.4, 0.6],
#                  [ 0.4, 1.4],
#                  colors='red', linestyles='solid', linewidth=4.0)
# ax[2,0].set_xlim(tmp)
# ax[2,0].set_ylim(-.41,1.4)
# t=Comp_E_inc_m_MA.loc(axis=1)[['A2l','A4l']].agg(['mean',Evac.confidence_interval])
# ftxt='Trimmed values:\n  A2*l: %5.2f (%4.2f - %5.2f)\n  A4*l: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','A2l'],*t.loc['confidence_interval','A2l'],
#                                                                                       t.loc['mean','A4l'],*t.loc['confidence_interval','A4l'])
# ax[2,0].text(-0.35,-0.35,ftxt,ha='left',va='bottom', 
#            bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
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
fig, ax = plt.subplots(nrows=3,figsize=(16.0/2.54, 24.5/2.54), constrained_layout=True)
fig.suptitle('Relative deviation of Youngs Moduli compared to\nconventional determined (Method A0Al)',
             fontsize=12, fontweight="bold")
g = sns.barplot(data=df.loc(axis=1)[df.columns.str[2:3]=='A'],
                 ax=ax[0], errwidth=1, capsize=.1)
ax[0].set_title('Methods without elimination (**A*)')
ax[0].set_xlabel('Evaluation method')
ax[0].tick_params(axis='x', labelrotation=90)
ax[0].set_ylabel('$D_{E,Method-A0Al} = (E_{Method} - E_{%s})/E_{%s}$'%(YM_con[2],YM_con[2]))
# ax[0].set_ylim(-0.01,2)
g = sns.barplot(data=df.loc(axis=1)[df.columns.str[2:3]=='S'],
                 ax=ax[1], errwidth=1, capsize=.1)
ax[1].set_title('Methods with support indentation elimination (**S*)')
ax[1].set_xlabel('Evaluation method')
ax[1].tick_params(axis='x', labelrotation=90)
ax[1].set_ylabel('$D_{E,Method-A0Al} = (E_{Method} - E_{%s})/E_{%s}$'%(YM_con[2],YM_con[2]))
# ax[1].set_ylim(-0.01,2)
g = sns.barplot(data=df.loc(axis=1)[(df.columns.str[2:3]=='M')|(df.columns.str[2:3]=='C')],
                 ax=ax[2], errwidth=1, capsize=.1)
ax[2].set_title('Methods with support indentation and shear deformation elimination (**M*|**C*)')
ax[2].set_xlabel('Evaluation method')
ax[2].tick_params(axis='x', labelrotation=90)
ax[2].set_ylabel('$D_{E,Method-A0Al} = (E_{Method} - E_{%s})/E_{%s}$'%(YM_con[2],YM_con[2]))
# tmp=ax[2].get_xlim()
# ax[2].hlines([3.25,3.25],
#              [-0.4, 0.6],
#              [ 0.4, 1.4],
#              colors='red', linestyles='solid', linewidth=4.0)
# ax[2].set_xlim(tmp)
# ax[2].set_ylim(-0.01,3.3)
# t=df.loc(axis=1)[['A2Ml','A4Ml']].agg(['mean',Evac.confidence_interval])
# ftxt='Trimmed values:\n  A2Ml: %5.2f (%4.2f - %5.2f)\n  A4Ml: %5.2f (%4.2f - %5.2f)'%(t.loc['mean','A2Ml'],*t.loc['confidence_interval','A2Ml'],
                                                                                     # t.loc['mean','A4Ml'],*t.loc['confidence_interval','A4Ml'])
# ax[2].text(1.81,2.21,ftxt,ha='left',va='bottom', 
#            bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_supmat+'06.png')
plt.savefig(out_supmat+'06.pdf')
plt.show()

#%% Close Log
log_mg.close()

