# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:56:34 2023

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
# plt_Fig_dict={'tight':True, 'show':True, 
#               'save':False, 's_types':["pdf","png"], 
#               'clear':True, 'close':True}
#%% Functions
def pd_agg_custom(pdo, agg_funcs=['mean',Evac.meanwoso,'median',
                                   'std',Evac.coefficient_of_variation, 
                                   Evac.stdwoso, Evac.coefficient_of_variation_woso,
                                   'max','min',Evac.confidence_interval], 
                  numeric_only=False, 
                  af_ren={'coefficient_of_variation_woso':'CVwoso',
                          'coefficient_of_variation':'CV'},
                  af_unp={'confidence_interval': ['CImin','CImax']}):
    def unpack_aggval(o, n, rn):
        tmp=o.loc[n].apply(lambda x: pd.Series([*x], index=rn))
        i = o.index.get_indexer_for([n])[0]
        s=pd.concat([o.iloc[:i],tmp.T,o.iloc[i+1:]],axis=0)
        return s
    pda = Evac.pd_agg(pd_o=pdo, agg_funcs=agg_funcs, numeric_only=numeric_only)
    for i in af_unp.keys():
        pda = unpack_aggval(pda, i, af_unp[i])
    if len(af_ren.keys())>0:
        pda = pda.rename(af_ren)
    return pda

#%% Einlesen und auswählen
data="S1"
# data="S2"
# data="Complete"

no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
               'B01.1','B01.2','B01.3', 'B02.3',
               'C01.1','C01.2','C01.3', 'C02.3',
               'D01.1','D01.2','D01.3', 'D02.3',
               'F01.1','F01.2','F01.3', 'F02.3',
               'G01.1','G01.2','G01.3', 'G02.3']
VIPar_plt_renamer = {'fy':'$f_{y}$','fu':'$f_{u}$','fb':'$f_{b}$',
                     'ey_con':r'$\epsilon_{y,con}$','Wy_con':'$W_{y,con}$',
                     'eu_con':r'$\epsilon_{u,con}$','Wu_con':'$W_{u,con}$',
                     'eb_con':r'$\epsilon_{b,con}$','Wb_con':'$W_{b,con}$',
                     'ey_opt':r'$\epsilon_{y,opt}$','Wy_opt':'$W_{y,opt}$',
                     'eu_opt':r'$\epsilon_{u,opt}$','Wu_opt':'$W_{u,opt}$',
                     'eb_opt':r'$\epsilon_{b,opt}$','Wb_opt':'$W_{b,opt}$',
                     # 'E_con': '$E_{con}$','E_opt': '$E_{opt}$',
                     'Density_app': r'$\rho_{app}$',
                     'Length_test': r'$l_{test}$','Length': r'$l$',
                     # 'thickness_mean': r'$\overline{t}$',
                     # 'width_mean': r'$\overline{w}$',
                     'thickness_mean': r'$\overline{h}$',
                     'thickness_2': r'$h_{mid}$','geo_dthick': r'$\Delta h/h_{mid}$',
                     'width_mean': r'$\overline{b}$',
                     'width_2': r'$b_{mid}$','geo_dwidth': r'$\Delta b/b_{mid}$',
                     'thickness_1': r'$h_{left}$','thickness_3': r'$h_{right}$',
                     'width_1': r'$b_{left}$','width_3': r'$b_{right}$',
                     'Area_CS': r'$\overline{A}_{CS}$', 'Volume': r'$\overline{V}$', 
                     'VolTot': r'$V_{tot}$',
                     'MoI_mid': r'$I_{mid}$','geo_MoI_mean':r'$\overline{I}$','geo_dMoI': r'$\Delta I/I_{mid}$',
                     'geo_curve_max': r'$\kappa_{max}$','geo_dcurve': r'$\Delta \kappa/\kappa_{mid}$',
                     'geo_curve_mid_circ': r'$\kappa_{mid,circle}$',
                     'ind_R_max': r'$ind_{el,max}$','ind_R_mean': r'$\overline{ind}_{el}$',
                     'ind_U_max': r'$ind_{u,max}$','ind_U_mean': r'$\overline{ind}_{u}$',
                     'Mass': r'$m$','MassW': r'$m_{water}$',
                     'DMWtoA': r'$D_{w,fresh}$','DMWtoG': r'$w_{s}$',
                     'RHenv': r'$\phi_{asp}$', 'RHstore':r'$\phi_{env}$',
                     'WC_vol': r'$\Phi$', 'WC_gra': r'$w$',
                     'WC_vol_toA':r'$\Delta_{\Phi,fresh}$','WC_vol_rDA':r'$D_{\Phi,fresh}$',
                     'WC_gra_toA':r'$\Delta_{w,fresh}$','WC_gra_rDA':r'$D_{w,fresh}$',
                     'lu_F_mean': r'$\overline{E}_{lr}$','lu_F_ratio': r'${E}_{r}/{E}_{l}$',
                     'DEFlutoB': r'$D_{E,sat}$','DEFlutoG': r'$D_{E,dry}$',
                     # 'Hyst_An': r'$H_{n}$','DHAntoB': r'$D_{H_{n},sat}$','DHAntoG': r'$D_{H_{n},dry}$',
                     # 'Hyst_APn': r'$H_{n}$','DHAPntoB': r'$D_{H_{n},sat}$','DHAPntoG': r'$D_{H_{n},dry}$'}
                     'HA': r'$H$','HAn': r'$H_{n}$',
                     'DHAntoB': r'$D_{H_{n},sat}$','DHAntoG': r'$D_{H_{n},dry}$'}

protpaths = pd.DataFrame([],dtype='string')
combpaths = pd.DataFrame([],dtype='string')
protpaths.loc['S1','name_prot'] = "Zesbo_Kiefer_Schwein_Protocol_220630.xlsx"
protpaths.loc['S1','path_eva2'] = "Complete/"
protpaths.loc['S1','fname_in']  = "CBPig_TBT-Summary_all"
protpaths.loc['S1','fname_out'] = "CBpig_TBT-Conclusion"
protpaths.loc['S1','name'] = "Pig"


# protpaths.loc['Complete','name_prot'] = ""
# protpaths.loc['Complete','path_eva2'] = "Complete/"
# protpaths.loc['Complete','fname_in']  = "CBPig_TBT-Summary_all"
# protpaths.loc['Complete','fname_out'] = "CBpig_TBT-Conclusion"
# protpaths.loc['Complete','name'] = "two series"

protpaths.loc[:,'path_main']    = "F:/ZESBO/Kiefer/1_Schwein/"
protpaths.loc[:,'path_eva1']    = "Auswertung/"

combpaths['prot'] = protpaths['path_main']+protpaths['path_eva1']+protpaths['name_prot']
combpaths['in']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']+protpaths['fname_in']
combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']+protpaths['fname_out']
# path_doda=os.path.abspath(os.path.join(protpaths.loc['Complete','path_main'],
#                                        protpaths.loc['Complete','path_eva1'],
#                                        'MM-CB_Donordata_full.xlsx'))

# name_Head = "Mandibula Compact Bone (%s, %s)\n"%(protpaths.loc[data,'name'],
#                                                            date.today().strftime("%d.%m.%Y"))
name_Head = ""

out_full= combpaths.loc[data,'out']
h5_conc = 'Summary'
h5_data = '/Add_/Measurement'

YM_con=['inc','R','A0Al','meanwoso']
YM_opt=['inc','R','D2Mgwt','meanwoso']
YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
YM_opt_str='E_{}_{}_{}_{}'.format(*YM_opt)
Methods_excl_names=["F4Agfu","F4Sgfu","F4Mgfu",
                    "F4gfu"]
Methods_excl_str='|'.join(Methods_excl_names)



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

dfa['Side_LR']=dfa.Origin.str.split(' ', expand=True)[2]
dfa['Side_pd']=dfa.Origin.str.split(' ', expand=True)[3]
# dfa['Series']=dfa.Designation.str[3]
# dfa['Numsch']=dfa.Designation.apply(lambda x: '{0}'.format(x[-3:]))
dfa.Failure_code  = Evac.list_cell_compiler(dfa.Failure_code)
dfa['statistics'] = Evac.list_interpreter(dfa.Failure_code, no_stats_fc)
dfa['Density_app']=dfa['Mass_org']/dfa['geo_Vol_mean']

dfs=dfa.loc[dfa.statistics]
# #Spender:
# doda = pd.read_excel(path_doda, skiprows=range(1,2), index_col=0)



#%% Data-Export
rel_col_com=['Designation','Donor','Origin','Side_LR','Side_pd']
rel_col_geo=['thickness_1','thickness_2','thickness_3',
             'width_1','width_2','width_3','Length','geo_Vol_mean']
rel_col_add=['Mass_org','Density_app']
rel_col_mec=['statistics','fy','ey_opt','fu','eu_opt',
             YM_con_str.replace('R','F'),YM_opt_str.replace('R','F'),
             YM_con_str,YM_opt_str]
dfc = dfa[rel_col_com+rel_col_geo+rel_col_add+rel_col_mec]
# dft_comb_rel.loc[~dft_comb_rel.statistics, ['fu','eu_opt']] = np.nan # SettingWithCopyWarning
dfc=dfc.rename({YM_con_str.replace('R','F'):'E_con_F',
                YM_opt_str.replace('R','F'):'E_opt_F',
                YM_con_str:'E_con_R',
                YM_opt_str:'E_opt_R'}, axis=1)

writer = pd.ExcelWriter(out_full+'.xlsx', engine = 'xlsxwriter')
# tmp=dft_comb_rel.rename(VIPar_plt_renamer,axis=1)
dfc.to_excel(writer, sheet_name='Conclusion')
tmp=pd_agg_custom(dfc.loc[dfc['statistics']],numeric_only=True)
tmp.T.to_excel(writer, sheet_name='Descriptive')
writer.close()

#%%Logging - descriptive

#%%% Plots
tmp=dfa.loc(axis=1)[dfa.columns.str.contains(r'^E_inc_.*_meanwoso$')].copy(deep=True)
tmp.columns = tmp.columns.str.split('_', expand=True)
tmp.columns.names=['Parameter','Determination','Range','Method','Value']

tmp2 = tmp.droplevel([0,1,4],axis=1)
tmp2=tmp2.unstack().reset_index()
fig, ax1 = plt.subplots()
ax = sns.barplot(data=tmp2, 
                 x='Method',y=0, hue='Range', ax=ax1, 
                 seed=0, errwidth=1, capsize=0.2)
ax1.set_title('%sYoungs moduli by determination method'%name_Head)
ax1.set_xlabel('Determination method / -')
ax1.set_ylabel(r'$E$ / MPa')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylim([0,5000])
fig.suptitle('')
Evac.plt_handle_suffix(fig,path=out_full+'-Box-YM-allMethods',**plt_Fig_dict)


# fig, ax1 = plt.subplots()
# ax = sns.boxplot(data=tmp,ax=ax1,
#                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", 
#                                             "markeredgecolor":"black", "markersize":"12","alpha":0.75})
# ax = sns.swarmplot(data=tmp, ax=ax1, 
#                    dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
# ax1.axvline(0.5,0,1, color='grey',ls='--')
# ax1.text(5,1.1,"Desorption",ha='center',va='center', 
#            bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
# ax1.axvline(6.0,0,1, color='grey',ls='--')
# ax1.text(7,1.1,"Adsorption",ha='center',va='center', 
#            bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
# ax1.set_title('%sWater content based on dry mass of the different manipulation variants'%name_Head)
# ax1.set_xlabel('Procedure step / -')
# ax1.set_ylabel(r'$D_{mass,Water}$ / -')
# ax1.set_yscale("log")
# ax1.set_yticks([0.1,1.0])
# ax1.grid(True, which='both', axis='y')
# fig.suptitle('')
# Evac.plt_handle_suffix(fig,path=out_full+'-Box-DMWtoG-ext',**plt_Fig_dict)