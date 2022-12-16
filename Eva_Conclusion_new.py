# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:01:15 2021

@author: mgebhard
"""


import os
import copy
import pandas as pd
import math
import numpy as np
from scipy import stats as spstat
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import Eva_common as Evac

plt.rcParams['figure.figsize'] = [6.3,3.54]
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size']= 8.0
#pd.set_option('display.expand_frame_repr', False)
plt.rcParams['lines.linewidth']= 1.0
plt.rcParams['lines.markersize']= 4.0
plt.rcParams['markers.fillstyle']= 'none'


# =============================================================================
#%% Einlesen und auswählen
Version="211203"
ptype="TBT"
ptype="ACT"
ptype="ATT"

# no_stats_fc = ['1.11','1.12','1.21','1.22','1.31','2.21','3.11','3.21','3.31']
no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
               'B01.1','B01.2','B01.3', 'B02.3',
               'C01.1','C01.2','C01.3', 'C02.3',
               'D01.1','D01.2','D01.3', 'D02.3',
               'F01.1','F01.2','F01.3', 'F02.3',
               'G01.1','G01.2','G01.3', 'G02.3']
               # 'G02.1','G02.2','G02.3']
# var_suffix = ["A","B","C","D"] #Suffix of variants of measurements (p.E. diffferent moistures)
var_suffix = [""] #Suffix of variants of measurements (p.E. diffferent moistures)

# if ptype=="TBT": VIParams=["Rohdichte","E2","ED2","fy","ey","Wy","fu","eu","Wu"]
# if ptype=="ACT": VIParams=["Rohdichte","E2","fy","ey","Wy","fu","eu","Wu"]
# if ptype=="ATT": VIParams=["Rohdichte","E","fy","ey","fu","eu"]
VIParF = {'ey_con':100,'eu_con':100,'ey_opt':100,'eu_opt':100,} # Faktor für VIParams in log-Ausgabe (nur wenn ungleich 1) [e_ in %, W_ in mJ/mm³]
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
                     'MoI_mid': r'$I_{mid}$',
                     'thickness_mean': r'$t_{mean}$', 'width_mean': r'$w_{mean}$',
                     'Area_CS': r'$A_{CS}$', 'Volume': r'$V$',
                     'Fy':'$F_{y}$','sy_con':'$s_{y,con}$',
                     'Fu':'$F_{u}$','su_con':'$s_{u,con}$',
                     'Fb':'$F_{u}$','sb_con':'$s_{b,con}$',
                     'D_con':r'$D_{con}$'}
Donor_dict={"LEIULANA_52-17":"PT2",
            "LEIULANA_67-17":"PT3",
            "LEIULANA_57-17":"PT4",
            "LEIULANA_48-17":"PT5",
            "LEIULANA_22-17":"PT6",
            "LEIULANA_60-17":"PT7"}

VIParams_don=["Sex","Age","Storagetime","BMI",
              "ICDCodes","Special_Anamnesis","Note_Anamnesis",
              "Fixation","Note_Fixation"]
if ptype=="TBT":   
    path = "D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/"+Version+"/TBT/"
    name_in   = "B3-B7_TBT-Summary"
    name_out  = "B3-B7_TBT-Conclusion"
    name_Head = "Compact bone"
    # YM_con=['inc','R','A0Al','mean']
    # YM_opt=['inc','R','G1Mgwt','mean']
    YM_con=['inc','R','A0Al','meanwoso']
    # # YM_opt=['inc','R','G1Mgwt','meanwoso']
    # YM_opt=['inc','R','G2Mgwt','meanwoso'] # Methodenumbenennung 211203
    YM_opt=['inc','R','D2Mgwt','meanwoso']
    YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
    YM_opt_str='E_{}_{}_{}_{}'.format(*YM_opt)
    VIParams_gen=["Designation","Origin","Donor"]
    VIParams_geo=["thickness_mean","width_mean",
                  "Area_CS","Volume","geo_MoI_mid","Density_app"]
    # VIParams_mat=["fy","ey_opt","Wy_opt","fu","eu_opt","Wu_opt",YM_con_str,YM_opt_str]
    VIParams_mat=["fy","ey_opt","Wy_opt",
                  "fu","eu_opt","Wu_opt",
                  "fb","eb_opt","Wb_opt",
                  YM_con_str,YM_opt_str]
    VIParams_rename = {'geo_MoI_mid':'MoI_mid',
                       YM_con_str:'E_con',YM_opt_str:'E_opt'}
elif ptype=="ACT":
    path = "D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/"+Version+"/ACT/"
    name_in   = "B3-B7_ACT-Summary"
    name_out  = "B3-B7_ACT-Conclusion"
    name_Head = "Trabecular bone"
    # YM_con=['inc','R','A0Al','mean']
    # YM_opt=['inc','R','G1Mgwt','mean']
    # YM_con=['inc','R','A0Al','meanwoso']
    YM_con=['lsq','R','A0Al','E']
    # YM_opt=['inc','R','G1Mgwt','meanwoso']
    YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
    # YM_opt_str='E_{}_{}_{}_{}'.format(*YM_opt)
    VIParams_gen=["Designation","Origin","Donor","Direction_test"]
    VIParams_geo=["Length_test",
                  "Area_CS","Volume","Density_app"]
    # VIParams_mat=["fy","ey_con","Wy_con","fu","eu_con","Wu_con",YM_con_str]
    VIParams_mat=["fy","ey_con","Wy_con",
                  "fu","eu_con","Wu_con",
                  "fb","eb_con","Wb_con",
                  YM_con_str]
    VIParams_rename = {'geo_MoI_mid':'MoI_mid',
                       YM_con_str:'E_con'}
elif ptype=="ATT":
    path = "D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/"+Version+"/ATT/"
    name_in   = "B3-B7_ATT-Summary"
    name_out  = "B3-B7_ATT-Conclusion"
    name_Head = "Soft tissue"
    # YM_con=['inc','R','A0Al','mean']
    # YM_opt=['inc','R','G1Mgwt','mean']
    # YM_con=['inc','R','A0Al','meanwoso']
    YM_con=['lsq','R','A0Al','E']
    # YM_opt=['inc','R','G1Mgwt','meanwoso']
    YM_con_str='E_{}_{}_{}_{}'.format(*YM_con)
    # YM_opt_str='E_{}_{}_{}_{}'.format(*YM_opt)
    D_con_str='D_{}_{}_{}'.format(*YM_con[:-1])
    VIParams_gen=["Designation","Origin","Donor"]
    VIParams_geo=["Area_CS","Volume","Density_app"]
    # VIParams_mat=["fy","ey_con","Wy_con","fu","eu_con","Wu_con",YM_con_str,D_con_str]
    # VIParams_mat=["fy","ey_con","Wy_con","fu","eu_con","Wu_con",YM_con_str,
    #               "Fy","sy_con","Fu","su_con",D_con_str]
    VIParams_mat=["fy","ey_con","Wy_con",
                  "fu","eu_con","Wu_con",
                  "fb","eb_con","Wb_con",
                  YM_con_str,
                  "Fy","sy_con","Fu","su_con","Fb","sb_con",D_con_str]
    VIParams_rename = {D_con_str:'D_con',
                       YM_con_str:'E_con'}
    
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
             # 'Ligamentum pectineale':                       'Pecti',
             'Ligamentum pectineum':                        'Pecti',
             'Ligamenta sacroiliaca anteriora':             'SaIlia',
             'Ligamenta sacroiliaca posteriora':            'SaIlip',
             'Ligamentum sacroiliacum posterior longum':    'SaIlil',
             'Membrana obturatoria':                        'MObt',
             # 'Fascia glutea maximus':                       'FGM',
             'Fascia glutea':                               'FGlu',
             'Fascia lumbalis':                             'FLum',
             'Fascia crescent':                             'FCres',
             'Fascia endopelvina':                          'FEnPe',
             # 'Fascia thoracolumbalis':                      'FTCL',
             # 'Fascia thoracolumbalis deep layer':           'FTCLdL'}
             'Fascia thoracolumbalis lamina superficalis':  'FTCLls',
             'Fascia thoracolumbalis lamina profunda':      'FTCLlp'}
else: print("Failure ptype!!!")

out_full= os.path.abspath(path+name_out)
path_doda = os.path.abspath('F:/Messung/000-PARAFEMM_Patientendaten/PARAFEMM_Donordata_full.xlsx')
h5_conc = 'Summary'
h5_data = 'Test_Data'
VIParams = copy.deepcopy(VIParams_geo)
VIParams.extend(VIParams_mat)

log_mg=open(out_full+'.log','w')
Evac.MG_strlog(name_out, log_mg, printopt=False)
Evac.MG_strlog("\n   Paths:", log_mg, printopt=False)
Evac.MG_strlog("\n   - in:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(os.path.abspath(path+name_in+'.h5')), log_mg, printopt=False)
Evac.MG_strlog("\n   - out:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(out_full), log_mg, printopt=False)
Evac.MG_strlog("\n   - donor:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(path_doda), log_mg, printopt=False)
Evac.MG_strlog("\n   Donors:"+Evac.str_indent('\n{}'.format(pd.Series(Donor_dict).to_string()),5), log_mg, printopt=False)


data_read = pd.HDFStore(path+name_in+'.h5','r')
dfa=data_read.select(h5_conc)
dft=data_read.select(h5_data)
data_read.close()

del dfa['No']
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


cs = dfa.loc[dfa.statistics]
cs_num_cols = cs.select_dtypes(include=['int','float']).columns
cs_num_cols = cs_num_cols[np.invert(cs_num_cols.str.startswith('OPT_'))] # exclude Options

cs_short_gen = cs[VIParams_gen]
# cs_short_gen.columns=pd.MultiIndex.from_product([["General"],cs_short_gen.columns],
#                                                 names=["Main","Parameter"])
cs_short_geo = cs[VIParams_geo]
# cs_short_geo.columns=pd.MultiIndex.from_product([["Geometry"],cs_short_geo.columns],
#                                                 names=["Main","Parameter"])
cs_short_mat = cs[VIParams_mat]
# cs_short_mat.columns=pd.MultiIndex.from_product([["Material"],cs_short_mat.columns],
#                                                 names=["Main","Parameter"])
cs_short = pd.concat([cs_short_gen,cs_short_geo,cs_short_mat],axis=1)
cols_short = cs_short.columns
cs_short.rename(columns=VIParams_rename,inplace=True)
VIParams_short=[VIParams_rename.get(item,item)  for item in VIParams]

#%% Statistische Auswertungen
agg_funcs = ['size','min','max','mean','median','std']

# c_eva=cs[cs_num_cols].agg(agg_funcs,
#          **{'skipna':True,'numeric_only':True,'ddof':1})
c_eva=Evac.pd_agg(cs,agg_funcs,True)


c_eva.loc['ouliers']      = cs[cs_num_cols].agg(Evac.stat_outliers,
                                            **{'option':'IQR', 'span':1.5,
                                               'out':'all', 'outsort' : 'ascending'})
c_eva.loc['ouliers_high'] = cs[cs_num_cols].agg(Evac.stat_outliers,
                                            **{'option':'IQR', 'span':1.5,
                                               'out':'higher', 'outsort' : 'ascending'})
c_eva.loc['ouliers_low']  = cs[cs_num_cols].agg(Evac.stat_outliers,
                                            **{'option':'IQR', 'span':1.5,
                                               'out':'lower', 'outsort' : 'descending'})
c_eva.loc['ouliers_not']  = cs[cs_num_cols].agg(Evac.stat_outliers,
                                            **{'option':'IQR', 'span':1.5,
                                               'out':'inner', 'outsort' : None})

c_short_eva=Evac.pd_agg(cs_short,agg_funcs,True)
if ptype=="ATT":
    c_short_Type_eva=Evac.pd_agg(pd.concat([cs_short,cs['Type']],axis=1).groupby('Type'),agg_funcs,True)
    c_short_eva = pd.concat([c_short_eva,c_short_Type_eva.stack()])
    c_short_eva_Fas=c_short_Type_eva.stack().loc['Fascia']
    c_short_eva_Lig=c_short_Type_eva.stack().loc['Ligament']
#%% Zusammenfassung nach Spender
doda = pd.read_excel(path_doda, skiprows=range(1,2), index_col=0)
# d_eva = cs.groupby('Donor')[cs_num_cols].agg(agg_funcs)
doda['Date_Test']=pd.to_datetime(dfa.loc[dfa.statistics].groupby('Donor')['Date_test'].max())
doda['Storagetime']=(doda['Date_Test']-doda['Date_Death']).dt.days
# d_eva_c = Evac.pd_agg(cs.groupby('Donor'),agg_funcs,True)
d_eva_c = Evac.pd_agg(cs.groupby('Donor')[cs_num_cols],agg_funcs,True)
# d_eva = d_eva.stack()
# d_eva.loc[pd.IndexSlice[:,'mean'],:].droplevel(1)
d_eva_c = pd.concat([doda,d_eva_c],axis=1)

d_eva_short = Evac.pd_agg(cs_short.groupby('Donor'),agg_funcs,True)
d_eva_short = pd.concat([doda.loc[d_eva_short.index,VIParams_don],d_eva_short],axis=1)


#%% Zusammenfassung nach Ort (Auswertung):
h_eva = cs.groupby('Origin_short')[cs_num_cols].agg(agg_funcs,
                                       **{'skipna':True,'numeric_only':True,'ddof':1})
h_eva_add = cs.groupby('Origin_short')[cs_num_cols].agg([Evac.stat_outliers],
                                                      **{'option':'IQR', 'span':1.5,
                                                          'out':'all', 'outsort' : 'ascending'})
h_eva_add = h_eva_add.rename(columns={'stat_outliers':'outliers'})


h_eva_c=h_eva.join(h_eva_add).stack()
h_eva_c.columns=h_eva.columns.get_level_values(0).drop_duplicates()


h_eva_short = Evac.pd_agg(pd.concat([cs_short,cs['Origin_short']],axis=1).groupby('Origin_short'),agg_funcs,True)
h_eva_short = h_eva_short.stack()

if ptype=="ACT":
    h_dir_eva=cs.groupby(['Origin_short','Direction_test'])[cs_num_cols].agg(agg_funcs)
    h_dir_eva_short = Evac.pd_agg(pd.concat([cs_short,cs['Origin_short']],axis=1).groupby(['Origin_short','Direction_test']),agg_funcs,True)
    h_dir_eva_short = h_dir_eva_short.stack()
# h_eva=h_eva.stack()
#%% Speicherung
dfa.append(c_eva,sort=False).to_csv(out_full+'.csv',sep=';') 
h_eva_c.to_csv(out_full+'-Loc.csv',sep=';')

cs_short.append(c_short_eva,sort=False).to_csv(out_full+'-short.csv',sep=';')
if ptype=="ATT": 
    cs_short[cs.Type == 'Fascia'  ].append(c_short_eva_Fas,sort=False).to_csv(out_full+'-Fas-short.csv',sep=';')
    cs_short[cs.Type == 'Ligament'].append(c_short_eva_Lig,sort=False).to_csv(out_full+'-Lig-short.csv',sep=';')
h_eva_short.to_csv(out_full+'-Loc-short.csv',sep=';')
if ptype=="ACT": h_dir_eva_short.to_csv(out_full+'-Loc-Dir-short.csv',sep=';')

data_store = pd.HDFStore(out_full+'.h5')
data_store['Material_Data'] = dfa   # write to HDF5
data_store['Test_Data'] = dft   # write to HDF5
data_store['Conclusion'] = c_eva   # write to HDF5
data_store['Conclusion_Donor'] = d_eva_c   # write to HDF5
data_store['Conclusion_Location'] = h_eva_c   # write to HDF5
data_store.close()

#%% Kontrolle
if False:
    print('='*100)
    print('Kontrolle %s'%YM_con_str)
    t = c_eva.loc['ouliers'].loc[c_eva.loc['ouliers'].index.str.contains(YM_con_str)]
    print(t.values)
    print(cs.loc[t.values.tolist()[0]][YM_con_str])
    tb = 'E_inc_R_A0Al_meanwoso'
    print('Kontrolle %s'%tb)
    t = c_eva.loc['ouliers'].loc[c_eva.loc['ouliers'].index.str.contains(tb)]
    print(t.values)
    print(cs.loc[t.values.tolist()[0]][tb])
    print('='*100)
#%% Logging
Evac.MG_strlog("\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog("\n Statistics per location: (Mean ± SD)",
               log_mg,1,printopt=False)
Evac.MG_strlog("\n"+" "*44+"Origin;  N"+("; {:>18}"*len(VIParams_short)).format(*VIParams_short),
               log_mg,1,printopt=False)
for i in h_eva_short.index.get_level_values(0).drop_duplicates().values:
    txt="\n%50s;%3d"%(i,h_eva_short.loc[pd.IndexSlice[i,'size']][-1])
    for j in VIParams_short:
        t=h_eva_short.loc(axis=0)[pd.IndexSlice[i,['mean','std']]][j].values
        if len(t)==1: t=np.append(t,0)
        if j in VIParF:
            # txt+=(";%.3f \u00B1 %.3f" %(h_eva_short.loc[pd.IndexSlice[i,'mean'],j]*VIParF[j],h_eva_short.loc[pd.IndexSlice[i,'std'],j]*VIParF[j]))
            txt+=(";%8.3f \u00B1 %8.3f" %(*(t*VIParF[j]),))
        else:
            txt+=(";%8.3f \u00B1 %8.3f" %(*t,))
            
    Evac.MG_strlog(txt,log_mg,1,printopt=False)
    
# if ptype=="ACT":
#     Evac.MG_strlog("\nStatistics per location and direction: (Mean ± SD-Mean)",
#                log_mg,1,printopt=False)
#     log_mg.write("\n                                           Herkunft; Ex; Ey; Ez;" )
#     for i in h_dir_eva.index.get_level_values(0).drop_duplicates().values:
#         txt="\n %50s"%(i)
#         for j in h_dir_eva.index.get_level_values(1).drop_duplicates().values:
#             # txt+=(";%.3f \u00B1 %.3f" %(h_dir_eva.loc[pd.IndexSlice[i,j,'Mean'],'E2'],h_dir_eva.loc[pd.IndexSlice[i,j,'SD'],'E2']))    
#             txt+=(";%.3f \u00B1 %.3f" %(h_dir_eva.loc[pd.IndexSlice[i,j,'Mean'],'E2'],h_dir_eva.loc[pd.IndexSlice[i,j,'SDM'],'E2']))                
#         log_mg.write(txt)


# if ptype=="ACT":
#     txt= ('\n   E-Modul (gesamt): %.3f +/- %.3f' %(dfa.loc[dfa.statistics].mean(skipna=True,numeric_only=True)['E2'],dfa.loc[dfa.statistics].std(skipna=True,numeric_only=True)['E2']))
#     txt+=('\n   E-Modul (x):      %.3f +/- %.3f' %(dfa.loc[(dfa['Prüfrichtung']=='x')&(dfa.statistics)].mean(skipna=True,numeric_only=True)['E2'],dfa.loc[(dfa['Prüfrichtung']=='x')&(dfa.statistics)].std(skipna=True,numeric_only=True)['E2']))
#     txt+=('\n   E-Modul (y):      %.3f +/- %.3f' %(dfa.loc[(dfa['Prüfrichtung']=='y')&(dfa.statistics)].mean(skipna=True,numeric_only=True)['E2'],dfa.loc[(dfa['Prüfrichtung']=='y')&(dfa.statistics)].std(skipna=True,numeric_only=True)['E2']))
#     txt+=('\n   E-Modul (z):      %.3f +/- %.3f' %(dfa.loc[(dfa['Prüfrichtung']=='z')&(dfa.statistics)].mean(skipna=True,numeric_only=True)['E2'],dfa.loc[(dfa['Prüfrichtung']=='z')&(dfa.statistics)].std(skipna=True,numeric_only=True)['E2']))
#     print(txt)
#     log_mg.write("\n\nStatistics for directions:")
#     log_mg.write(txt)
#%% HTML plot
if False:
        
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
    config = {
      'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 500,
        'width': 700,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
        },
      'editable': True}
    
    fig = go.Figure()
    fig.update_layout(title={'text':"<b> Ultimate normed stress strain curves </b> <br>(%s)" %name_Head,
                              'y':.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      xaxis_title="strain / ultimate strain / -",
                      yaxis_title="stress / ultimate stress / -",
                      legend_title='<b> Specimen </b>')
    # Add traces
    for i in dft.index:
        End_step=dft[i].loc[dft[i].VIP_m.str.contains('U').fillna(False)].index[0]
        fig.add_trace(go.Scatter(x=dft[i].Strain.loc[:End_step]/dft[i].Strain.loc[End_step],
                                  y=dft[i].Stress.loc[:End_step]/dft[i].Stress.loc[End_step],
                                  mode='lines',line={'dash':'dash'},
                                  name=i.split('_')[2]+' | '+i.split('_')[4],
                                  legendgroup = 'conventional'))
        fig.add_trace(go.Scatter(x=dft[i].Strain_opt_c_M.loc[:End_step]/dft[i].Strain_opt_c_M.loc[End_step],
                                  y=dft[i].Stress.loc[:End_step]/dft[i].Stress.loc[End_step],
                                  mode='lines',line={'dash':'dot'},
                                  name=i.split('_')[2]+' | '+i.split('_')[4],
                                  legendgroup = 'optical'))
        
    fig.show(config=config)
    
    fig = go.Figure()
    fig.update_layout(title={'text':"<b> Stress strain curves </b> <br>(%s)" %name_Head,
                              'y':.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      xaxis_title="Strain / -",
                      yaxis_title="Stress / MPa",
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


#%%% Verteilung
Evac.MG_strlog("\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog("\n Distribution tests:",
               log_mg,1,printopt=False)
txt=("\n  - Statistical test (Shapiro-Wilks):")
for i in VIParams:
    stat, p = spstat.shapiro(cs.loc[cs.statistics].loc[:,i].dropna())
    txt+=("\n    Statistics of %24s = %.3e, p= %.3e" % (i,stat, p))
    if p > alpha:
        txt+=("    -> Sample looks Gaussian (fail to reject H0)")
    else: 
        txt+=("    -> Sample does not look Gaussian (reject H0)")
Evac.MG_strlog(txt,log_mg,1,printopt=False)

txt=("\n  - Statistical test (D’Agostino’s K^2):")
for i in VIParams:
    stat, p = spstat.normaltest(cs.loc[cs.statistics].loc[:,i].dropna())
    txt+=("\n    Statistics of %24s = %.3e, p= %.3e" % (i,stat, p))
    if p > alpha:
        txt+=("    -> Sample looks Gaussian (fail to reject H0)")
    else: 
        txt+=("    -> Sample does not look Gaussian (reject H0)")
Evac.MG_strlog(txt,log_mg,1,printopt=False)

#%%% Correlation

cs_short_corr = cs_short.corr(method=mcorr)

fig, ax1 = plt.subplots(figsize = (6.3,2*3.54))
g=sns.heatmap(cs_short_corr.round(1),
            center=0,annot=True, annot_kws={"size":5, 'rotation':90},ax=ax1)
Evac.tick_label_renamer(ax=g, renamer=VIPar_plt_renamer, axis='both')
ax1.set_title('%s\nInfluence of material data'%name_Head)
ax1.set_xlabel('Influences')
ax1.set_ylabel('Influences')
ax1.tick_params(axis='x', labelrotation=90)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Inf-Mat_Mat.pdf')
plt.savefig(out_full+'-Inf-Mat_Mat.png')
plt.show()

if False:
    # sns.pairplot(cs_short)
    # sns.pairplot(cs_short,kind='kde')
    g=sns.pairplot(cs_short.merge(cs.Origin_sshort, left_index=True, right_index=True),
                 hue='Origin_sshort')
    g.fig.suptitle('%s\nInfluence of material data'%name_Head)
    g.savefig(out_full+'-Inf-Mat_Mat-pp.pdf')
    g.savefig(out_full+'-Inf-Mat_Mat-pp.png')
    # g.show()

#%%% Cyclic Testing Influence
if ptype == "ATT":
    
    cs_cfl = cs[['cyc_f_lo','cyc_f_hi']].div(cs.fu,axis=0).dropna()
    
    txt="\n "+"="*100
    txt+=("\n Cyclic-loading influence: (%d samples)"%cs.OPT_pre_load_cycles[cs.OPT_pre_load_cycles>0].size)
    txt += Evac.str_indent('- cyclic loading stress level (related to ultimate strength):')
    txt += Evac.str_indent('- preload (aim: 0.10):',6)
    txt += Evac.str_indent('  {0:.3f} ± {1:.3f} ({2:.3f}-{3:.3f})'.format(*cs_cfl.cyc_f_lo.agg(['mean','std','min','max'])),6)
    txt += Evac.str_indent('- cyclic (aim: 0.30):',6)
    txt += Evac.str_indent('  {0:.3f} ± {1:.3f} ({2:.3f}-{3:.3f})'.format(*cs_cfl.cyc_f_hi.agg(['mean','std','min','max'])),6)

    cs_E = cs.loc(axis=1)[cs.columns.str.startswith('E_lsq')]
    cs_E.columns = cs_E.columns.str.split('_', expand=True)
    cs_E = cs_E.droplevel(0,axis=1)
    # cs_E.columns.names=['Determination','Cycle','Method','Parameter']
    idx = pd.IndexSlice
    cs_E_lsq_m=cs_E.loc(axis=1)[idx['lsq',:,'A0Al','E']].droplevel([0,2,3],axis=1)
    cs_E_lsq_m_pR=cs_E_lsq_m.loc(axis=1)[cs_E_lsq_m.columns.str.contains(r'^C',regex=True)]
    cs_E_lsq_m_pR=cs_E_lsq_m_pR.div(cs_E_lsq_m['F'],axis=0)
    # cs_E_lsq_m_pR=cs_E_lsq_m_pR.div(cs_E_lsq_m['R'],axis=0)
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

    txt += Evac.str_indent('- cyclic related to final Youngs Modulus:')
    txt += Evac.str_indent('- load (ascending):',6)
    txt += Evac.str_indent(cs_E_lsq_m_pR_rise.agg(['mean','std',
                                                   Evac.meanwoso,Evac.stdwoso,
                                                   'min','max']).T,9)
    txt += Evac.str_indent('statistical outliers: %d'%len(cs_E_lsq_m_pR_rise_so),9)
    txt += Evac.str_indent(cs.Failure_code[cs_E_lsq_m_pR_rise_so],9)
    
    txt += Evac.str_indent('- unload (descending):',6)
    txt += Evac.str_indent(cs_E_lsq_m_pR_fall.agg(['mean','std',
                                                   Evac.meanwoso,Evac.stdwoso,
                                                   'min','max']).T,9)
    txt += Evac.str_indent('statistical outliers: %d'%len(cs_E_lsq_m_pR_fall_so),9)
    txt += Evac.str_indent(cs.Failure_code[cs_E_lsq_m_pR_fall_so],12)
    txt += Evac.str_indent('- plastic strain after cycle:')
    txt += Evac.str_indent(cs_epl_m.agg(['mean','std',
                                         Evac.meanwoso,Evac.stdwoso,
                                         'min','max']).T,9)
    txt += Evac.str_indent('statistical outliers: %d'%len(cs_epl_m_so),9)
    txt += Evac.str_indent(cs.Failure_code[cs_epl_m_so],12)
    Evac.MG_strlog(txt,log_mg,1,printopt=False)
    
    
    fig, ax1 = plt.subplots()
    sns.scatterplot(data=cs_cfl,x='cyc_f_lo',y='cyc_f_hi',
                    label='real stress level',ax=ax1)
    ax1.set_title('%s\nPre vs. cyclic stress level'%name_Head)
    ax1.set_xlabel('pre stress/ultimate stress / -')
    ax1.set_ylabel('cyclic stress/ultimate stress / -')
    ax1.axvline(0.1, color='green', label='goal')
    ax1.axhline(0.3, color='green',)
    ax1.legend()
    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Inf-Cyc-slvl.pdf')
    plt.savefig(out_full+'-Inf-Cyc-slvl.png')
    plt.show()
    
    df=pd.concat([cs_cfl,cs_E_lsq_m_pR_rise],axis=1)
    fig, ax1 = plt.subplots()
    kws = {"marker":"d", "s": 40, "facecolor": "none", "edgecolor":"black",
           "linewidth": 0.5}
    sns.scatterplot(data=df.loc[cs_E_lsq_m_pR_rise_so],x='cyc_f_hi',y='1',
                    label='statisitcal outliers', ax=ax1, **kws)
    sns.scatterplot(data=df.loc[cs_E_lsq_m_pR_rise_so],x='cyc_f_hi',y='2',
                    legend=None,ax=ax1,**kws)
    sns.scatterplot(data=df.loc[cs_E_lsq_m_pR_rise_so],x='cyc_f_hi',y='5',
                    legend=None,ax=ax1,**kws)
    sns.scatterplot(data=df.loc[cs_E_lsq_m_pR_rise_so],x='cyc_f_hi',y='10',
                    legend=None,ax=ax1,**kws)
    sns.scatterplot(data=df,x='cyc_f_hi',y='1',label='cycle 1',ax=ax1)
    sns.scatterplot(data=df,x='cyc_f_hi',y='2',label='cycle 2',ax=ax1)
    sns.scatterplot(data=df,x='cyc_f_hi',y='5',label='cycle 5',ax=ax1)
    sns.scatterplot(data=df,x='cyc_f_hi',y='10',label='cycle 10',ax=ax1)
    ax1.set_title('%s\nDeviation of Youngs Modulus vs. cyclic stress level'%name_Head)
    ax1.set_xlabel('cyclic stress/ultimate stress  [-]')
    ax1.set_ylabel('cyclic related to final Youngs Modulus / -')
    ax1.axvline(0.3, color='green', label='goal')
    ax1.axhline(1.0, color='green',)
    ax1.legend()
    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Inf-Cyc-E_slvl.pdf')
    plt.savefig(out_full+'-Inf-Cyc-E_slvl.png')
    plt.show()

    
    fig, ax1 = plt.subplots()
    sns.lineplot(data=cs_E_lsq_m_pR_rise.loc[cs_E_lsq_m_pR_rise_si].T,palette=('cyan',),
                 linestyle='dashed',linewidth=0.25,legend=False,ax=ax1)
    sns.lineplot(data=cs_E_lsq_m_pR_fall.loc[cs_E_lsq_m_pR_fall_si].T,palette=('orange',),
                 linestyle='dashed',linewidth=0.25,legend=False,ax=ax1)
    lns = sns.lineplot(data=cs_E_lsq_m_pR_rise.loc[cs_E_lsq_m_pR_rise_si].mean(),color='blue',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='load',ax=ax1)
    lns = sns.lineplot(data=cs_E_lsq_m_pR_fall.loc[cs_E_lsq_m_pR_fall_si].mean(),color='red',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='unload',ax=ax1)
    ax1.legend_.remove()
    ax1.xaxis.grid()
    ln = lns.get_legend_handles_labels()[0]
    la = lns.get_legend_handles_labels()[1]
    ax2=ax1.twinx()
    sns.lineplot(data=cs_epl_m.loc[cs_epl_m_si].T,palette=('limegreen',),
                 linestyle='dashed',linewidth=0.25,legend=False,ax=ax2)
    lns = sns.lineplot(data=cs_epl_m.loc[cs_epl_m_si].mean(),color='green',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='plastic strain',ax=ax2)
    ln += lns.get_legend_handles_labels()[0]
    la += lns.get_legend_handles_labels()[1]
    ax2.legend_.remove()
    ax1.set_xlabel('cycle / -')
    ax1.set_ylabel('cyclic related to final Youngs Modulus / -')
    ax2.set_ylabel('plastic strain / -')
    ax1.legend(ln,la,loc='center right')
    plt.title('%s\nCyclic loading (without statistical outliers)'%name_Head)
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Inf-Cyc-all.pdf')
    plt.savefig(out_full+'-Inf-Cyc-all.png')
    plt.show()
    
    fig, ax1 = plt.subplots()
    lns = sns.lineplot(data=cs_E_lsq_m_pR_rise.loc[cs_E_lsq_m_pR_rise_si].mean(),color='blue',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='load',ax=ax1)
    lns = sns.lineplot(data=cs_E_lsq_m_pR_fall.loc[cs_E_lsq_m_pR_fall_si].mean(),color='red',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='unload',ax=ax1)
    ax1.legend_.remove()
    ax1.xaxis.grid()
    ln = lns.get_legend_handles_labels()[0]
    la = lns.get_legend_handles_labels()[1]
    ax2=ax1.twinx()
    lns = sns.lineplot(data=cs_epl_m.loc[cs_epl_m_si].mean(),color='green',
                 linestyle='-',linewidth=1.0,legend=True,
                 label='plastic strain',ax=ax2)
    ln += lns.get_legend_handles_labels()[0]
    la += lns.get_legend_handles_labels()[1]
    ax2.legend_.remove()
    ax1.set_xlabel('cycle / -')
    ax1.set_ylabel('cyclic related to final Youngs Modulus / -')
    ax2.set_ylabel('plastic strain / -')
    ax1.legend(ln,la,loc='center right')
    plt.title('%s\nCyclic loading (without statistical outliers)'%name_Head)
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Inf-Cyc-mean.pdf')
    plt.savefig(out_full+'-Inf-Cyc-mean.png')
    plt.show()

#%%% Lineare Regression
txt="\n "+"="*100
txt+=("\n Linear regression:")
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
if ptype=="TBT":
    Lin_reg_df['Eoptvsfu'] = MG_linRegStats(cs[YM_opt_str], cs['fu'],
                                            "Youngs Modulus (optical)",
                                            "ultimate strength")
Lin_reg_df['EconvsRho'] = MG_linRegStats(cs[YM_con_str], cs['Density_app'],
                                         "Youngs Modulus (conventional)",
                                         "apparent density")
if ptype=="TBT":
    Lin_reg_df['EoptvsRho'] = MG_linRegStats(cs[YM_opt_str], cs['Density_app'],
                                             "Youngs Modulus (optical)",
                                             "apparent density")
Lin_reg_df['fuvsRho'] = MG_linRegStats(cs['fu'], cs['Density_app'],
                                       "ultimate strength",
                                       "apparent density")

# des_txt = ("\n  - Linear relation between Youngs Modulus (conventional) and ultimate strength:\n   ")
# Y, X = cs[YM_con_str], cs['fu']
# tmp  = Evac.YM_sigeps_lin(Y, X)
# smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
# Lin_reg_df['Econvsfu']=pd.Series([*tmp,smst,des_txt],index=['s','c','Rquad',
#                                                             'fit','smstat',
#                                                             'Description'])
# if ptype=="TBT":
#     des_txt = ("\n  - Linear relation between Youngs Modulus (optical) and ultimate strength:\n   ")
#     Y, X = cs[YM_opt_str], cs['fu']
#     tmp  = Evac.YM_sigeps_lin(Y, X)
#     smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
#     Lin_reg_df['Eoptvsfu']=pd.Series([*tmp,smst,des_txt],index=['s','c','Rquad',
#                                                                 'fit','smstat',
#                                                                 'Description'])

# des_txt = ("\n  - Linear relation between Youngs Modulus (conventional) and apparent density:\n   ")
# Y, X = cs[YM_con_str], cs['Density_app']
# tmp  = Evac.YM_sigeps_lin(Y, X)
# smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
# Lin_reg_df['EconvsRho']=pd.Series([*tmp,smst,des_txt],index=['s','c','Rquad',
#                                                              'fit','smstat',
#                                                              'Description'])
# if ptype=="TBT":
#     des_txt = ("\n  - Linear relation between Youngs Modulus (optical) and apparent density:\n   ")
#     Y, X = cs[YM_opt_str], cs['Density_app']
#     tmp  = Evac.YM_sigeps_lin(Y, X)
#     smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
#     Lin_reg_df['EoptvsRho']=pd.Series([*tmp,smst,des_txt],index=['s','c','Rquad',
#                                                                  'fit','smstat',
#                                                                  'Description'])

# des_txt = ("\n  - Linear relation between ultimate strength and apparent density:\n   ")
# Y, X = cs['fu'], cs['Density_app']
# tmp  = Evac.YM_sigeps_lin(Y, X)
# smst = sm.OLS(Y, sm.add_constant(X)).fit().summary()
# Lin_reg_df['fuvsRho']=pd.Series([*tmp,smst,des_txt],index=['s','c','Rquad',
#                                                            'fit','smstat',
#                                                            'Description'])

for i in Lin_reg_df.columns:
    txt += Lin_reg_df.loc['Description',i]
    txt += Evac.str_indent(Evac.fit_report_adder(*Lin_reg_df.loc[['fit','Rquad'],i]))
    txt += Evac.str_indent(Lin_reg_df.loc['smstat',i])

Evac.MG_strlog(txt,log_mg,1,printopt=False)

# # from sklearn.model_selection import train_test_split
# Y = cs[YM_opt_str]
# X = cs['Density_app']
# # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 465)
# # X_train = sm.add_constant(X_train)
# # results = sm.OLS(y_train, X_train).fit()
# X = sm.add_constant(X)
# results = sm.OLS(Y, X).fit()
# results.summary()

#%%% Herkunft
#%%%% Populationsanalyse + Multicomparison
Evac.MG_strlog("\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog("\n %s-Origin: (Groups are significantly different for p < %.3f)"%(mpop,alpha),
               log_mg,1,printopt=False)
Origin_T_df=pd.DataFrame([],dtype='string')

txt,_,T = Evac.group_ANOVA_MComp(df=cs, groupby='Origin_short', ano_Var=YM_con_str,
                                 group_str='Origin', ano_str='Youngs Modulus (conventional)',
                                 mpop=mpop, alpha=alpha, group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Origin_T_df[YM_con_str]=pd.Series(d)
if ptype=="TBT":
    txt,_,T = Evac.group_ANOVA_MComp(df=cs, groupby='Origin_short', ano_Var=YM_opt_str,
                                     group_str='Origin', ano_str='Youngs Modulus (optical)',
                                     mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
    Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
    d,txt2 = Evac.MComp_interpreter(T)
    Origin_T_df[YM_opt_str]=pd.Series(d)
txt,_,T = Evac.group_ANOVA_MComp(df=cs, groupby='Origin_short', ano_Var='fu',
                                 group_str='Origin', ano_str='ultimate strength',
                                 mpop=mpop, alpha=alpha, group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Origin_T_df['fu']=pd.Series(d)
txt,_,T = Evac.group_ANOVA_MComp(df=cs, groupby='Origin_short', ano_Var='Density_app',
                                 group_str='Origin', ano_str='apparent density',
                                 mpop=mpop, alpha=alpha, group_ren={}, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Origin_T_df['Density_app']=pd.Series(d)
Origin_T_df.rename(columns=VIParams_rename,inplace=True)
Evac.MG_strlog("\n\n   -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
               log_mg,1,printopt=False)
Evac.MG_strlog(Evac.str_indent(Origin_T_df,6), log_mg,1,printopt=False)


#%%% Spender
#%%%% ANOVA + Tukey HSD
Evac.MG_strlog("\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog("\n %s-Donor: (Groups are significantly different for p < %.3f)"%(mpop,alpha),
               log_mg,1,printopt=False)
Donor_T_df=pd.DataFrame([],dtype='string')

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# anova_df = cs.loc[:,[YM_con_str, YM_opt_str,'fy','fu']]
# # # ANOVA results with combinations of 2 groups:
# # formula = '%s ~ C(%s) + C(%s) + C(fy) + C(fu)'%(YM_con_str,YM_con_str,YM_opt_str)
# # lm = ols(formula, anova_df).fit()
# # print(lm.summary())
# # table = sm.stats.anova_lm(lm, typ=1)
# # print(table)
# formula = '%s ~ %s'%(YM_con_str,YM_con_str)
# lm = ols(formula, anova_df).fit()
# print(lm.summary())
# table = sm.stats.anova_lm(lm, typ=1)
# print(table)

txt,_,T = Evac.group_ANOVA_MComp(df=cs, groupby='Donor', ano_Var=YM_con_str,
                                 group_str='Donor', ano_str='Youngs Modulus (conventional)',
                                 mpop=mpop, alpha=alpha, group_ren=Donor_dict, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Donor_T_df[YM_con_str]=pd.Series(d)
if ptype=="TBT":
    txt,_,T = Evac.group_ANOVA_MComp(df=cs, groupby='Donor', ano_Var=YM_opt_str,
                                     group_str='Donor', ano_str='Youngs Modulus (optical)',
                                     mpop=mpop, alpha=alpha, group_ren=Donor_dict, **MComp_kws)
    Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
    d,txt2 = Evac.MComp_interpreter(T)
    Donor_T_df[YM_opt_str]=pd.Series(d)
txt,_,T = Evac.group_ANOVA_MComp(df=cs, groupby='Donor', ano_Var='fu',
                                 group_str='Donor', ano_str='ultimate strength',
                                 mpop=mpop, alpha=alpha, group_ren=Donor_dict, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Donor_T_df['fu']=pd.Series(d)
txt,_,T = Evac.group_ANOVA_MComp(df=cs, groupby='Donor', ano_Var='Density_app',
                                 group_str='Donor', ano_str='apparent density',
                                 mpop=mpop, alpha=alpha, group_ren=Donor_dict, **MComp_kws)
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
d,txt2 = Evac.MComp_interpreter(T)
Donor_T_df['Density_app']=pd.Series(d)
Donor_T_df.rename(columns=VIParams_rename,inplace=True)
Evac.MG_strlog("\n\n   -> Multicomparison relationship interpretation (groups that share similar letter have an equal mean):",
               log_mg,1,printopt=False)
Evac.MG_strlog(Evac.str_indent(Donor_T_df,6), log_mg,1,printopt=False)


#%%%% Correlation
# doda_corr_c=cs_short.groupby('Donor').mean()
# doda_corr_d=doda.loc[doda_corr_c.index,VIParams_don]
# doda_corr_d['Sex'] = doda_corr_d['Sex'].map({'f':1,'m':-1})
# doda_corr_d_num_cols=doda_corr_d.select_dtypes(include=['int','float']).columns
# doda_corr_i= Evac.ICD_bool_df(doda.ICDCodes,**{'level':0})
# # doda_corr_i= Evac.ICD_bool_df(doda.ICDCodes,**{'level':1})
# doda_corr_i['SP'] = np.invert(doda['Special_Anamnesis'].isna()).astype(int)
# doda_corr=pd.concat([doda_corr_c,doda_corr_d,doda_corr_i],axis=1)
# doda_corr=doda_corr.corr(method=mcorr)

doda_corr_d=doda.loc(axis=1)[VIParams_don]
doda_corr_d['Sex'] = doda_corr_d['Sex'].map({'f':1,'m':-1})
doda_corr_d_num_cols=doda_corr_d.select_dtypes(include=['int','float']).columns
doda_corr_i= Evac.ICD_bool_df(doda.ICDCodes,**{'level':0})
doda_corr_i['SP'] = np.invert(doda['Special_Anamnesis'].isna()).astype(int)
doda_di = pd.concat([doda_corr_d.loc(axis=1)[doda_corr_d_num_cols],doda_corr_i],axis=1)
cs_doda_short=pd.merge(left=cs_short,right=doda_di,left_on='Donor',right_index=True)
doda_corr=cs_doda_short.corr(method=mcorr)

fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, 
                              sharex=True, sharey=False, figsize = (6.3,2*3.54))
fig.suptitle('%s\nInfluence of donor data on material parameters'%name_Head)
sns.heatmap(doda_corr.loc[doda_corr_d_num_cols,
                          cs_short.select_dtypes(include=['int','float']).columns].round(2),
                 center=0,annot=True, annot_kws={"size":5, 'rotation':90}, ax=ax1)
Evac.tick_label_renamer(ax=ax1, renamer=VIPar_plt_renamer, axis='both')
# ax1.set_xlabel('Material parameters')
ax1.set_ylabel('Donor data')
ax1.tick_params(axis='y', labelrotation=90)
plt.setp(ax1.get_yticklabels(), va="center")
sns.heatmap(doda_corr.loc[doda_corr_i.columns,
                          cs_short.select_dtypes(include=['int','float']).columns].round(2),
                 center=0,annot=True, annot_kws={"size":5, 'rotation':90}, ax=ax2)
Evac.tick_label_renamer(ax=ax2, renamer=VIPar_plt_renamer, axis='both')
ax2.set_xlabel('Material parameters')
ax2.set_ylabel('Donor ICD-codes')
ax2.tick_params(axis='x', labelrotation=90)
ax2.tick_params(axis='y', labelrotation=90)
plt.setp(ax2.get_yticklabels(), va="center")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Inf-Mat_Don.pdf')
plt.savefig(out_full+'-Inf-Mat_Don.png')
plt.show()

#%%%% Lineare Regression

doda_t = doda
doda_t.columns = doda_t.columns.str.replace('Naming','Donor')
cs_doda=cs.join(doda_t.set_index('Donor'),on='Donor',how='inner',rsuffix='Don_')

txt="\n "+"="*100
txt+=("\n Linear regression:")
Lin_reg_doda_df =pd.DataFrame([],dtype='O')


Lin_reg_doda_df['RhovsAge'] = MG_linRegStats(cs_doda['Density_app'], cs_doda['Age'],
                                              "apparent density",
                                              "donor age")
Lin_reg_doda_df['fuvsAge'] = MG_linRegStats(cs_doda['fu'], cs_doda['Age'],
                                              "ultimate strength",
                                              "donor age")
Lin_reg_doda_df['EconvsAge'] = MG_linRegStats(cs_doda[YM_con_str], cs_doda['Age'],
                                              "Youngs Modulus (conventional)",
                                              "donor age")
if ptype=="TBT":
    Lin_reg_doda_df['EoptvsAge'] = MG_linRegStats(cs_doda[YM_opt_str], cs_doda['Age'],
                                                  "Youngs Modulus (optical)",
                                                  "donor age")

# des_txt = ("\n  - Linear relation between apparent density and donor age:\n   ")
# tmp  = Evac.YM_sigeps_lin(cs_doda['Density_app'],cs_doda['Age'])
# Lin_reg_doda_df['RhovsAge']=pd.Series([*tmp,des_txt],
#                                       index=['s','c','Rquad','fit','Description'])
# des_txt = ("\n  - Linear relation between ultimate strength and donor age:\n   ")
# tmp  = Evac.YM_sigeps_lin(cs_doda['fu'],cs_doda['Age'])
# Lin_reg_doda_df['fuvsAge']=pd.Series([*tmp,des_txt],
#                                        index=['s','c','Rquad','fit','Description'])

# des_txt = ("\n  - Linear relation between Youngs Modulus (conventional) and donor age:\n   ")
# tmp  = Evac.YM_sigeps_lin(cs_doda[YM_con_str],cs_doda['Age'])
# Lin_reg_doda_df['EconvsAge']=pd.Series([*tmp,des_txt],
#                                        index=['s','c','Rquad','fit','Description'])
# if ptype=="TBT":
#     des_txt = ("\n  - Linear relation between Youngs Modulus (optical) and donor age:\n   ")
#     tmp  = Evac.YM_sigeps_lin(cs_doda[YM_opt_str],cs_doda['Age'])
#     Lin_reg_doda_df['EoptvsAge']=pd.Series([*tmp,des_txt],
#                                            index=['s','c','Rquad','fit','Description'])

for i in Lin_reg_doda_df.columns:
    txt += Lin_reg_doda_df.loc['Description',i]
    txt += Evac.str_indent(Evac.fit_report_adder(*Lin_reg_doda_df.loc[['fit','Rquad'],i]))
    txt += Evac.str_indent(Lin_reg_doda_df.loc['smstat',i])
Evac.MG_strlog(txt,log_mg,1,printopt=False)

#%%% Failure Codes
Evac.MG_strlog("\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog("\n Failure Codes Analysies:",
               log_mg,1,printopt=False)
fco=-1

# complete Codes
fc_b_df, fc_unique = Evac.Failure_code_bool_df(dfa.Failure_code, 
                                      sep=',',level=2, strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_all_sum = fc_b_df.sum().sort_values(ascending=False)
fc_b_df_fail_sum = fc_b_df[dfa.statistics==False].sum().sort_values(ascending=False)
fc_b_df_nofail_sum = fc_b_df[dfa.statistics==True].sum().sort_values(ascending=False)

txt='Failure codes frequency: (first %d)'%fco
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
txt='- all:\n   %s'%fc_b_df_all_sum.iloc[0:fco][fc_b_df_all_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),log_mg,1,printopt=False)
txt='- fail:\n   %s'%fc_b_df_fail_sum.iloc[0:fco][fc_b_df_fail_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),log_mg,1,printopt=False)
txt='- no fail:\n   %s'%fc_b_df_nofail_sum.iloc[0:fco][fc_b_df_nofail_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),log_mg,1,printopt=False)

fc_b_df_mat = np.dot(fc_b_df.T, fc_b_df)
fc_b_df_freq = pd.DataFrame(fc_b_df_mat, columns = fc_unique, index = fc_unique)

fig, ax = plt.subplots()
ax.set_title('%s\nFailure codes frequence'%name_Head)
sns.heatmap(fc_b_df_freq, cmap = "Reds",ax=ax)
ax.set_xlabel('Failure code')
ax.set_ylabel('Failure code')
plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig(out_full+'-FCodes-all.pdf')
plt.savefig(out_full+'-FCodes-all.png')
plt.show()
plt.close(fig)

ind = [(True if x in no_stats_fc else False) for x in fc_unique]
# fc_b_df_hm = fc_b_df_freq.loc[ind,np.invert(ind)]
fc_b_df_hm = fc_b_df_freq.loc[ind]

fig, ax = plt.subplots()
ax.set_title('%s\nFailure codes frequence (only excluding)'%name_Head)
sns.heatmap(fc_b_df_hm.loc(axis=1)[np.invert((fc_b_df_hm==0).all())],
           cmap = "Reds",annot=True,ax=ax)
ax.set_xlabel('Failure code')
ax.set_ylabel('Failure code (excluding)')
plt.yticks(rotation=0)
fig.tight_layout()
plt.savefig(out_full+'-FCodes-excl.pdf')
plt.savefig(out_full+'-FCodes-excl.png')
plt.show()
plt.close(fig)


# combined Procedure and type - strength 2,3
fc_b_df, fc_unique = Evac.Failure_code_bool_df(dfa.Failure_code, 
                                      sep=',',level=1, strength=[2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_all_sum = fc_b_df.sum().sort_values(ascending=False)
fc_b_df_fail_sum = fc_b_df[dfa.statistics==False].sum().sort_values(ascending=False)
fc_b_df_nofail_sum = fc_b_df[dfa.statistics==True].sum().sort_values(ascending=False)

txt='Failure codes frequency: (combined - only strength 2 and 3, first %d)'%fco
Evac.MG_strlog(Evac.str_indent(txt),log_mg,1,printopt=False)
txt='- all:\n   %s'%fc_b_df_all_sum.iloc[0:fco][fc_b_df_all_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),log_mg,1,printopt=False)
txt='- fail:\n   %s'%fc_b_df_fail_sum.iloc[0:fco][fc_b_df_fail_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),log_mg,1,printopt=False)
txt='- no fail:\n   %s'%fc_b_df_nofail_sum.iloc[0:fco][fc_b_df_nofail_sum!=0].to_dict()
Evac.MG_strlog(Evac.str_indent(txt,5),log_mg,1,printopt=False)

fc_b_df_mat = np.dot(fc_b_df.T, fc_b_df)
fc_b_df_freq = pd.DataFrame(fc_b_df_mat, columns = fc_unique, index = fc_unique)

fig, ax = plt.subplots()
ax.set_title('%s\nFailure codes frequence (combined, only medium and strong)'%name_Head)
sns.heatmap(fc_b_df_freq, cmap = "Reds",ax=ax)
ax.set_xlabel('Failure code')
ax.set_ylabel('Failure code')
plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig(out_full+'-FCodes-all-ProTyp.pdf')
plt.savefig(out_full+'-FCodes-all-ProTyp.png')
plt.show()
plt.close(fig)

ind = [(True if x.split('.')[0] in Evac.Failure_code_lister(no_stats_fc) else False) for x in fc_unique]
# fc_b_df_hm = fc_b_df_freq.loc[ind,np.invert(ind)]
fc_b_df_hm = fc_b_df_freq.loc[ind]

fig, ax = plt.subplots()
ax.set_title('%s\nFailure codes frequence (only excluding, combined, only medium and strong)'%name_Head)
sns.heatmap(fc_b_df_hm.loc(axis=1)[np.invert((fc_b_df_hm==0).all())],
           cmap = "Reds",annot=True,ax=ax)
ax.set_xlabel('Failure code')
ax.set_ylabel('Failure code (excluding)')
plt.yticks(rotation=0)
fig.tight_layout()
plt.savefig(out_full+'-FCodes-excl-ProTyp.pdf')
plt.savefig(out_full+'-FCodes-excl-ProTyp.png')
plt.show()
plt.close(fig)


#%% Plots
#%%% Donor and Location
# Boxplot: Median, Q1 25% Quartil, Q3 75% Quartil, Ausreißer 1,5*(Q3-Q1), extreme Ausreißer 3,0*(Q3-Q1), Min+Max innerhalb 1,5*(Q3-Q1)
#%%%% Herkunft
fig, ax1 = plt.subplots()
ax1.grid(True)
# sns.set_theme(style="whitegrid")
if ptype=="TBT":
    df=pd.melt(cs, id_vars=['Origin_sshort'], value_vars=[YM_con_str,YM_opt_str])
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", hue="variable", data=df, ax=ax1, 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value", hue="variable",
                        data=df, ax=ax1, dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles[0:2], ['$E_{con}$','$E_{opt}$'], loc="best")
    # ax = sns.swarmplot(x="H_sshort", y="value", hue="variable", data=df, ax=ax1, color=".25")
else:
    df=pd.melt(cs, id_vars=['Origin_sshort'], value_vars=[YM_con_str])
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax1, dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYoungs-modulus vs. location'%name_Head)
ax1.set_xlabel('Location / -')
ax1.set_ylabel('E / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-E_Loc-box.pdf')
plt.savefig(out_full+'-E_Loc-box.png')
plt.show()

if ptype=="ACT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs, id_vars=['Origin_sshort','Direction_test'], value_vars=[YM_con_str])
    df.sort_values(['Origin_sshort','Direction_test'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", hue="Direction_test",
                     data=df, ax=ax1, palette={'x':'r','y':'g','z':'b'}, saturation=.5,
                     showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                "markeredgecolor":"black",
                                                "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value", hue="Direction_test",
                       data=df, ax=ax1, palette={'x':'r','y':'g','z':'b'},
                       dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s\nYoungs-modulus vs. location and direction'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('E / MPa')
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles[0:3], ['$x$','$y$','$z$'], title="Direction", loc="best")
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-E_Loc-Dir.pdf')
    plt.savefig(out_full+'-E_Loc-Dir.png')
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs, id_vars=['Origin_sshort','Direction_test'], value_vars=[YM_con_str])
    df.sort_values(['Origin_sshort','Direction_test'],inplace=True)
    df.Direction_test.replace({'y':'yz','z':'yz'},inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", hue="Direction_test",
                     data=df, ax=ax1, palette={'x':'r','yz':'cyan'}, saturation=.5,
                     showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                "markeredgecolor":"black",
                                                "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value", hue="Direction_test",
                       data=df, ax=ax1, palette={'x':'r','yz':'cyan'},
                       dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s\nYoungs-modulus vs. location and direction'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('E / MPa')
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles[0:2], ['$x$','$yz$'], title="Direction", loc="best")
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-E_Loc-Dir_comp.pdf')
    plt.savefig(out_full+'-E_Loc-Dir_comp.png')
    plt.show()
    
if ptype == "ATT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs[cs.Type=='Fascia'], id_vars=['Origin_sshort'],
               value_vars=[YM_con_str])
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax1, dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s - Fascia\nYoungs-modulus vs. location'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('E / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Fas-E_Loc-box.pdf')
    plt.savefig(out_full+'-Fas-E_Loc-box.png')
    plt.show()  
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs[cs.Type=='Ligament'], id_vars=['Origin_sshort'],
               value_vars=[YM_con_str])
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax1, dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s - Ligament\nYoungs-modulus vs. location'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('E / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-E_Loc-box.pdf')
    plt.savefig(out_full+'-Lig-E_Loc-box.png')
    plt.show()  
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs[cs.Type=='Ligament'], id_vars=['Origin_sshort'],
               value_vars=[D_con_str])
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax1, dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s - Ligament\nSpring stiffness vs. location'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('D / (N/mm)')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-D_Loc-box.pdf')
    plt.savefig(out_full+'-Lig-D_Loc-box.png')
    plt.show()  
    
fig, ax1 = plt.subplots()
ax1.grid(True)
df=pd.melt(cs, id_vars=['Origin_sshort'], value_vars=['fu'])
df.sort_values(['Origin_sshort','variable'],inplace=True)
ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                             "markeredgecolor":"black",
                                             "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Origin_sshort", y="value", data=df, ax=ax1,
                   dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nUltimate strength vs. location'%name_Head)
ax1.set_xlabel('Location / -')
ax1.set_ylabel('Stress / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-fu_Loc-box.pdf')
plt.savefig(out_full+'-fu_Loc-box.png')
plt.show()

if ptype == "ATT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs[cs.Type=='Fascia'], id_vars=['Origin_sshort'],
               value_vars=['fu'])
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax1, dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s - Fascia\nUltimate strength vs. location'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('Stress / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Fas-fu_Loc-box.pdf')
    plt.savefig(out_full+'-Fas-fu_Loc-box.png')
    plt.show()  
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs[cs.Type=='Ligament'], id_vars=['Origin_sshort'],
               value_vars=['fu'])
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax1, dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s - Ligament\nUltimate strength vs. location'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('Stress / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-fu_Loc-box.pdf')
    plt.savefig(out_full+'-Lig-fu_Loc-box.png')
    plt.show()  
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs[cs.Type=='Ligament'], id_vars=['Origin_sshort'],
               value_vars=['Fu'])
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                      showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value",
                        data=df, ax=ax1, dodge=True, edgecolor="black",
                        linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s - Ligament\nUltimate force vs. location'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('F / N')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-Fou_Loc-box.pdf')
    plt.savefig(out_full+'-Lig-Fou_Loc-box.png')
    plt.show()  


fig, ax1 = plt.subplots()
ax1.grid(True)
df=pd.melt(cs, id_vars=['Origin_sshort'], value_vars=['Wu_con'])
df.sort_values(['Origin_sshort','variable'],inplace=True)
ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                             "markeredgecolor":"black",
                                             "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Origin_sshort", y="value", data=df, ax=ax1,
                   dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nUltimate work vs. location'%name_Head)
ax1.set_xlabel('Location / -')
ax1.set_ylabel('Work / Nmm')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Wu_Loc-box.pdf')
plt.savefig(out_full+'-Wu_Loc-box.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
df=pd.melt(cs, id_vars=['Origin_sshort'], value_vars=['Density_app'])
df.sort_values(['Origin_sshort','variable'],inplace=True)
ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                             "markeredgecolor":"black",
                                             "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Origin_sshort", y="value", data=df, ax=ax1,
                   dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nApparent density vs. location'%name_Head)
ax1.set_xlabel('Location / -')
ax1.set_ylabel('Density / (g/cm³)')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Rho_Loc-box.pdf')
plt.savefig(out_full+'-Rho_Loc-box.png')
plt.show()

if ptype=="ACT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs, id_vars=['Origin_sshort','Direction_test'], value_vars=['fu'])
    df.sort_values(['Origin_sshort','Direction_test'],inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", hue="Direction_test",
                     data=df, ax=ax1, palette={'x':'r','y':'g','z':'b'}, saturation=.5,
                     showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                "markeredgecolor":"black",
                                                "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value", hue="Direction_test",
                       data=df, ax=ax1, palette={'x':'r','y':'g','z':'b'},
                       dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s\nUltimate strength vs. location and direction'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('Stress / MPa')
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles[0:3], ['$x$','$y$','$z$'], title="Direction", loc="best")
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-fu_Loc-Dir.pdf')
    plt.savefig(out_full+'-fu_Loc-Dir.png')
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs, id_vars=['Origin_sshort','Direction_test'], value_vars=['fu'])
    df.sort_values(['Origin_sshort','Direction_test'],inplace=True)
    df.Direction_test.replace({'y':'yz','z':'yz'},inplace=True)
    ax = sns.boxplot(x="Origin_sshort", y="value", hue="Direction_test",
                     data=df, ax=ax1, palette={'x':'r','yz':'cyan'}, saturation=.5,
                     showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                                "markeredgecolor":"black",
                                                "markersize":"12","alpha":0.75})
    ax = sns.swarmplot(x="Origin_sshort", y="value", hue="Direction_test",
                       data=df, ax=ax1, palette={'x':'r','yz':'cyan'},
                       dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
    ax1.set_title('%s\nUltimate strength vs. location and direction'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('Stress / MPa')
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles[0:2], ['$x$','$yz$'], title="Direction", loc="best")
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-fu_Loc-Dir_comp.pdf')
    plt.savefig(out_full+'-fu_Loc-Dir_comp.png')
    plt.show()

#%%%% Spender
fig, ax1 = plt.subplots()
ax1.grid(True)
if ptype=="TBT":
    df=pd.melt(cs, id_vars=['Donor','Origin_sshort'], value_vars=[YM_opt_str])
    df.Donor.replace(Donor_dict,inplace=True)
    df.sort_values(['Origin_sshort','variable'],inplace=True)
if ptype=="ACT" or "ATT":
    df=pd.melt(cs, id_vars=['Donor','Origin_sshort'], value_vars=[YM_con_str])
    df.Donor.replace(Donor_dict,inplace=True)
    df.sort_values(['Origin_sshort','variable'],inplace=True)
df.sort_values(['Donor','Origin_sshort'],inplace=True)
ax = sns.barplot(x="Origin_sshort", y="value", hue="Donor", data=df, ax=ax1, errwidth=1, capsize=.1)
ax1.legend(title='Donor',loc="best", ncol=2)
ax1.set_title('%s\nYoungs-modulus vs. location and donor'%name_Head)
ax1.set_xlabel('Location / -')
# ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('E / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-E_Loc-Donor.pdf')
plt.savefig(out_full+'-E_Loc-Donor.png')
plt.show()


if ptype == "ATT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs[cs.Type=='Fascia'], id_vars=['Donor','Origin_sshort'],
               value_vars=[YM_con_str])
    df.Donor.replace(Donor_dict,inplace=True)
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    df.sort_values(['Donor','Origin_sshort'],inplace=True)
    ax = sns.barplot(x="Origin_sshort", y="value", hue="Donor", data=df, ax=ax1, errwidth=1, capsize=.1)
    ax1.legend(title='Donor',loc="best", ncol=2)
    ax1.set_title('%s - Fascia\nYoungs-modulus vs. location and donor'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('E / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Fas-E_Loc-Donor.pdf')
    plt.savefig(out_full+'-Fas-E_Loc-Donor.png')
    plt.show()  
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    df=pd.melt(cs[cs.Type=='Ligament'], id_vars=['Donor','Origin_sshort'],
               value_vars=[YM_con_str])
    df.Donor.replace(Donor_dict,inplace=True)
    df.sort_values(['Origin_sshort','variable'],inplace=True)
    df.sort_values(['Donor','Origin_sshort'],inplace=True)
    ax = sns.barplot(x="Origin_sshort", y="value", hue="Donor", data=df, ax=ax1, errwidth=1, capsize=.1)
    ax1.legend(title='Donor',loc="best", ncol=2)
    ax1.set_title('%s - Ligament\nYoungs-modulus vs. location and donor'%name_Head)
    ax1.set_xlabel('Location / -')
    ax1.set_ylabel('E / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-E_Loc-Donor.pdf')
    plt.savefig(out_full+'-Lig-E_Loc-Donor.png')
    plt.show()  
    

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.barplot(x="Donor", y="value", hue="Origin_sshort",
                 data=df, ax=ax1, errwidth=1, capsize=.1)
ax1.legend(title='Location',loc="best", ncol=2)
ax1.set_title('%s\nYoungs-modulus vs. donor and location'%name_Head)
ax1.set_xlabel('Donor / -')
# ax1.tick_params(axis='x', labelrotation=90)
ax1.set_ylabel('E / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-E_Donor-Loc.pdf')
plt.savefig(out_full+'-E_Donor-Loc.png')
plt.show()


fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Donor", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Donor", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nYoungs-modulus vs. donor'%name_Head)
ax1.set_xlabel('Donor / -')
ax1.set_ylabel('E / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-E_Donor.pdf')
plt.savefig(out_full+'-E_Donor.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
df=pd.melt(cs, id_vars=['Donor','Origin_sshort'], value_vars=["fu"])
df.Donor.replace(Donor_dict,inplace=True)
# df.sort_values(['H_sshort','variable'],inplace=True)
ax = sns.boxplot(x="Donor", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Donor", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nUltimate strength vs. donor'%name_Head)
ax1.set_xlabel('Donor / -')
ax1.set_ylabel('Stress / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-fu_Donor.pdf')
plt.savefig(out_full+'-fu_Donor.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
df=pd.melt(cs, id_vars=['Donor','Origin_sshort'], value_vars=["Density_app"])
df.Donor.replace(Donor_dict,inplace=True)
# df.sort_values(['H_sshort','variable'],inplace=True)
ax = sns.boxplot(x="Donor", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Donor", y="value", data=df, ax=ax1, dodge=True, edgecolor="black", linewidth=.5, alpha=.5, size=2)
ax1.set_title('%s\nApparent density vs. donor'%name_Head)
ax1.set_xlabel('Donor / -')
ax1.set_ylabel('Density / (g/cm³)')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Rho_Donor.pdf')
plt.savefig(out_full+'-Rho_Donor.png')
plt.show()

#%%% Regression to parameters
formula_in_dia = False

fig, ax1 = plt.subplots()
ax1.grid(True)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['EconvsRho'].iloc[:3]).split(',')
    txt = r'$E_{con}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt='$E_{con}$'
ax = sns.regplot(x="Density_app", y=YM_con_str, data=cs,
                 label=txt, ax=ax1)
if ptype=="TBT":
    if formula_in_dia:
        txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['EoptvsRho'].iloc[:3]).split(',')
        txt = r'$E_{opt}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
    else:
        txt='$E_{opt}$'
    ax = sns.regplot(x="Density_app", y=YM_opt_str, data=cs,
                     label=txt, ax=ax1)
ax1.legend()
ax1.set_title('%s\nYoungs-modulus vs. density'%name_Head)
ax1.set_xlabel('Density / (g/cm³)')
ax1.set_ylabel('E / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Reg-E_Rho.pdf')
plt.savefig(out_full+'-Reg-E_Rho.png')
plt.show()

if ptype == "ATT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.regplot(x="Density_app", y=YM_con_str, data=cs[cs.Type=='Fascia'],
                     label='$E_{con}$', ax=ax1)
    ax1.set_title('%s - Fascia\nYoungs-modulus vs. density'%name_Head)
    ax1.set_xlabel('Density / (g/cm³)')
    ax1.set_ylabel('E / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Fas-Reg-E_Rho.pdf')
    plt.savefig(out_full+'-Fas-Reg-E_Rho.png')
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.regplot(x="Density_app", y=YM_con_str, data=cs[cs.Type=='Ligament'],
                     label='$E_{con}$', ax=ax1)
    ax1.set_title('%s - Ligament\nYoungs-modulus vs. density'%name_Head)
    ax1.set_xlabel('Density / (g/cm³)')
    ax1.set_ylabel('E / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-Reg-E_Rho.pdf')
    plt.savefig(out_full+'-Lig-Reg-E_Rho.png')
    plt.show()
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.regplot(x="Mass", y=D_con_str, data=cs[cs.Type=='Ligament'],
                     label='$E_{con}$', ax=ax1)
    ax1.set_title('%s - Ligament\nSpring stiffness v. mass'%name_Head)
    ax1.set_xlabel('m / g')
    ax1.set_ylabel('D / (N/mm)')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-Reg-D_Mass.pdf')
    plt.savefig(out_full+'-Lig-Reg-D_Mass.png')
    plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['fuvsRho'].iloc[:3]).split(',')
    txt = r'$f_{u}$ = %s $\rho_{app}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$f_{u}$'
ax = sns.regplot(x="Density_app", y='fu', data=cs, label=txt, ax=ax1)
ax1.set_title('%s\nUltimate strength vs. density'%name_Head)
ax1.set_xlabel('Density / (g/cm³)')
ax1.set_ylabel('Strength / MPa')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Reg-fu_Rho.pdf')
plt.savefig(out_full+'-Reg-fu_Rho.pdf')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
if formula_in_dia:
    txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Econvsfu'].iloc[:3]).split(',')
    txt = r'$E_{con}$ = %s $f_{u}$ + %s ($R²$ = %s)'%(*txt,)
else:
    txt = '$E_{con}$'
ax = sns.regplot(x="fu", y=YM_con_str, data=cs, label=txt, ax=ax1)
if ptype=="TBT":
    if formula_in_dia:
        txt = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Eoptvsfu'].iloc[:3]).split(',')
        txt = r'$E_{opt}$ = %s $f_{u}$ + %s ($R²$ = %s)'%(*txt,)
    else:
        txt = '$E_{opt}$'        
    ax = sns.regplot(x="fu", y=YM_opt_str, data=cs, label=txt, ax=ax1)
ax1.legend()
ax1.set_title('%s\nYoungs-modulus vs. ultimate strength'%name_Head)
ax1.set_xlabel('Strength / MPa')
ax1.set_ylabel('E / MPa')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Reg-E_fu.pdf')
plt.savefig(out_full+'-Reg-E_fu.png')
plt.show()

if ptype == "ATT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.regplot(x="fu", y=YM_con_str, data=cs[cs.Type=='Fascia'],
                     label='$E_{con}$', ax=ax1)
    ax1.set_title('%s - Fascia\nYoungs-modulus vs. ultimate strength'%name_Head)
    ax1.set_xlabel('Strength / MPa')
    ax1.set_ylabel('E / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Fas-Reg-E_fu.pdf')
    plt.savefig(out_full+'-Fas-Reg-E_fu.png')
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.regplot(x="fu", y=YM_con_str, data=cs[cs.Type=='Ligament'],
                     label='$E_{con}$', ax=ax1)
    ax1.set_title('%s - Ligament\nYoungs-modulus vs. ultimate strength'%name_Head)
    ax1.set_xlabel('Strength / MPa')
    ax1.set_ylabel('E / MPa')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-Reg-E_fu.pdf')
    plt.savefig(out_full+'-Lig-Reg-E_fu.png')
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.regplot(x="Fu", y=D_con_str, data=cs[cs.Type=='Ligament'],
                     label='$D_{con}$', ax=ax1)
    ax1.set_title('%s - Ligament\nSpring stiffness vs. ultimate force'%name_Head)
    ax1.set_xlabel('Force / N')
    ax1.set_ylabel('D / (N/mm)')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig-Reg-D_Fu.pdf')
    plt.savefig(out_full+'-Lig-Reg-D_Fu.png')
    plt.show()
    
#%%%% Donor:
    
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.regplot(x="Age", y=YM_con_str, data=cs_doda, label=r'$E_{con}$', ax=ax1)
ax = sns.scatterplot(x=d_eva_short['Age'], y=d_eva_short['E_con','mean'],
                     label=r'$E_{con,mean}$', s=250, marker='_', ax=ax1)
if ptype=="TBT":
    ax = sns.regplot(x="Age", y=YM_opt_str, data=cs_doda, label=r'$E_{opt}$', ax=ax1)
    ax = sns.scatterplot(x=d_eva_short['Age'], y=d_eva_short['E_opt','mean'],
                         label=r'$E_{opt,mean}$', s=250, marker='_', ax=ax1)
ax1.set_title('%s\nYoungs-modulus vs. donor age'%name_Head)
ax1.set_xlabel('Age / Year')
ax1.set_ylabel('E / MPa')
fig.suptitle('')
ax1.set_xlim(left=math.floor((ax1.get_xlim()[0]-1)/10)*10,
             right=math.ceil((ax1.get_xlim()[1]+1)/10)*10)
ax1.legend(loc='upper center',bbox_to_anchor=(0.25, 1.0))
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Reg-Don-E_Age.pdf')
plt.savefig(out_full+'-Reg-Don-E_Age.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.regplot(x="Age", y='fu', data=cs_doda, label=r'$f_{u}$', ax=ax1)
ax = sns.scatterplot(x=d_eva_short['Age'], y=d_eva_short['fu','mean'],
                     label=r'$f_{u,mean}$', s=250, marker='_', ax=ax1)
ax1.set_title('%s\nUltimate strength vs. donor age'%name_Head)
ax1.set_xlabel('Age / Year')
ax1.set_ylabel('Stress / MPa')
fig.suptitle('')
ax1.set_xlim(left=math.floor((ax1.get_xlim()[0]-1)/10)*10,
             right=math.ceil((ax1.get_xlim()[1]+1)/10)*10)
ax1.legend(loc='upper center',bbox_to_anchor=(0.25, 1.0))
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Reg-Don-fu_Age.pdf')
plt.savefig(out_full+'-Reg-Don-fu_Age.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.regplot(x="Age", y='Density_app', data=cs_doda, label=r'$\rho_{app}$', ax=ax1)
ax = sns.scatterplot(x=d_eva_short['Age'], y=d_eva_short['Density_app','mean'],
                     label=r'$\rho_{app,mean}$', s=250, marker='_', ax=ax1)
ax1.set_title('%s\nDensity vs. donor age'%name_Head)
ax1.set_xlabel('Age / Year')
ax1.set_ylabel('Density / (g/cm³)')
fig.suptitle('')
ax1.set_xlim(left=math.floor((ax1.get_xlim()[0]-1)/10)*10,
             right=math.ceil((ax1.get_xlim()[1]+1)/10)*10)
ax1.legend(loc='upper center',bbox_to_anchor=(0.25, 1.0))
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Reg-Don-Rho_Age.pdf')
plt.savefig(out_full+'-Reg-Don-Rho_Age.png')
plt.show()

#%%% Histograms
cs_hist = cs[VIParams_mat+["Density_app"]]/cs[VIParams_mat+["Density_app"]].max(axis=0)
cs_hist.rename(columns=VIParams_rename,inplace=True)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.histplot(cs_hist, stat='count', bins=20,
                  ax=ax1, kde=True, color=sns.color_palette()[0],
                  edgecolor=None, legend = True, alpha=0.25)
old_legend = ax1.legend_
handles = old_legend.legendHandles
labels = pd.Series([t.get_text() for t in old_legend.get_texts()])
labels.replace(VIPar_plt_renamer,inplace=True)
#title = old_legend.get_title().get_text()
for i in range(len(handles)):
    t=list(handles[i].get_facecolor())
    t[-1]=0.75
    handles[i].set_facecolor(tuple(t))
ax1.legend(handles, labels, loc='best', ncol=3)
ax1.set_title('%s\nHistogram of material parameters'%name_Head)
ax1.set_xlabel('Normed parameter / -')
ax1.set_ylabel('Count / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Hist_AllMat.pdf')
plt.savefig(out_full+'-Hist_AllMat.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.histplot(cs[YM_con_str], stat='count', bins=20, label='$E_{con}$',
                  ax=ax1, kde=True, color=sns.color_palette()[0], edgecolor=None)
if ptype=="TBT":
    ax = sns.histplot(cs[YM_opt_str], stat='count', bins=20, label='$E_{opt}$',
                      ax=ax1, kde=True, color=sns.color_palette()[1], edgecolor=None)
    ax1.legend()
ax1.set_title('%s\nYoungs-modulus histogram'%name_Head)
ax1.set_xlabel('E / MPa')
ax1.set_ylabel('Count / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Hist_E.pdf')
plt.savefig(out_full+'-Hist_E.png')
plt.show()

if ptype == "ATT":    
    cs_hist = cs[cs.Type=='Fascia'][VIParams_mat+["Density_app"]]/cs[cs.Type=='Fascia'][VIParams_mat+["Density_app"]].max(axis=0)
    cs_hist.rename(columns=VIParams_rename,inplace=True)
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.histplot(cs_hist, stat='count', bins=20,
                      ax=ax1, kde=True, color=sns.color_palette()[0],
                      edgecolor=None, legend = True, alpha=0.25)
    old_legend = ax1.legend_
    handles = old_legend.legendHandles
    labels = pd.Series([t.get_text() for t in old_legend.get_texts()])
    labels.replace(VIPar_plt_renamer,inplace=True)
    for i in range(len(handles)):
        t=list(handles[i].get_facecolor())
        t[-1]=0.75
        handles[i].set_facecolor(tuple(t))
    #title = old_legend.get_title().get_text()
    ax1.legend(handles, labels, loc='best', ncol=3)
    ax1.set_title('%s - Fascia\nHistogram of material parameters'%name_Head)
    ax1.set_xlabel('Normed parameter / -')
    ax1.set_ylabel('Count / -')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Fas_AllMat.pdf')
    plt.savefig(out_full+'-Fas_AllMat.png')
    plt.show()
    
    cs_hist = cs[cs.Type=='Ligament'][VIParams_mat+["Density_app"]]/cs[cs.Type=='Ligament'][VIParams_mat+["Density_app"]].max(axis=0)
    cs_hist.rename(columns=VIParams_rename,inplace=True)
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.histplot(cs_hist, stat='count', bins=20,
                      ax=ax1, kde=True, color=sns.color_palette()[0],
                      edgecolor=None, legend = True, alpha=0.25)
    old_legend = ax1.legend_
    handles = old_legend.legendHandles
    labels = pd.Series([t.get_text() for t in old_legend.get_texts()])
    labels.replace(VIPar_plt_renamer,inplace=True)
    for i in range(len(handles)):
        t=list(handles[i].get_facecolor())
        t[-1]=0.75
        handles[i].set_facecolor(tuple(t))
    #title = old_legend.get_title().get_text()
    ax1.legend(handles, labels, loc='best', ncol=3)
    ax1.set_title('%s - Ligament\nHistogram of material parameters'%name_Head)
    ax1.set_xlabel('Normed parameter / -')
    ax1.set_ylabel('Count / -')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Lig_AllMat.pdf')
    plt.savefig(out_full+'-Lig_AllMat.png')
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.histplot(cs[cs.Type=='Fascia'][YM_con_str],
                      stat='count', bins=20, label='$E_{con,Fascia}$',
                      ax=ax1, kde=True, color=sns.color_palette()[0], edgecolor=None)
    ax = sns.histplot(cs[cs.Type=='Ligament'][YM_con_str],
                      stat='count', bins=20, label='$E_{con,Ligament}$',
                      ax=ax1, kde=True, color=sns.color_palette()[1], edgecolor=None)
    ax1.legend()
    ax1.set_title('%s\nYoungs-modulus histogram'%name_Head)
    ax1.set_xlabel('E / MPa')
    ax1.set_ylabel('Count / -')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-FasLig-Hist_E.pdf')
    plt.savefig(out_full+'-FasLig-Hist_E.png')
    plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
x = sns.histplot(cs.fu, stat='count', bins=20, label='$f_{u}$',
                 ax=ax1, kde=True, color=sns.color_palette()[0], edgecolor=None)
# ax1.legend()
ax1.set_title('%s\nUltimate stress histogram'%name_Head)
ax1.set_xlabel('Stress / MPa')
ax1.set_ylabel('Count / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Hist_fu.pdf')
plt.savefig(out_full+'-Hist_fu.png')
plt.show()
if ptype == "ATT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.histplot(cs[cs.Type=='Fascia']['fu'],
                      stat='count', bins=20, label='$f_{u,Fascia}$',
                      ax=ax1, kde=True, color=sns.color_palette()[0], edgecolor=None)
    ax = sns.histplot(cs[cs.Type=='Ligament']['fu'],
                      stat='count', bins=20, label='$f_{u,Ligament}$',
                      ax=ax1, kde=True, color=sns.color_palette()[1], edgecolor=None)
    ax1.legend()
    ax1.set_title('%s\nUltimate stress histogram'%name_Head)
    ax1.set_xlabel('Stress / MPa')
    ax1.set_ylabel('Count / -')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-FasLig-Hist_fu.pdf')
    plt.savefig(out_full+'-FasLig-Hist_fu.png')
    plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
x = sns.histplot(cs.Density_app, stat='count', bins=20, label=r"$\rho_{app}$",
                 ax=ax1, kde=True, color=sns.color_palette()[0], edgecolor=None)
# ax1.legend()
ax1.set_title('%s\nDensity histogram'%name_Head)
ax1.set_xlabel('Density / (g/cm³)')
ax1.set_ylabel('Count / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Hist_Rho.pdf')
plt.savefig(out_full+'-Hist_Rho.png')
plt.show()
if ptype == "ATT":
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax = sns.histplot(cs[cs.Type=='Fascia']['Density_app'],
                      stat='count', bins=20, label=r'$\rho_{app,Fascia}$',
                      ax=ax1, kde=True, color=sns.color_palette()[0], edgecolor=None)
    ax = sns.histplot(cs[cs.Type=='Ligament']['Density_app'],
                      stat='count', bins=20, label=r'$\rho_{app,Ligament}$',
                      ax=ax1, kde=True, color=sns.color_palette()[1], edgecolor=None)
    ax1.legend()
    ax1.set_title('%s\nDensity histogram'%name_Head)
    ax1.set_xlabel('Density / (g/cm³)')
    ax1.set_ylabel('Count / -')
    fig.suptitle('')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-FasLig-Hist_Rho.pdf')
    plt.savefig(out_full+'-FasLig-Hist_Rho.png')
    plt.show()





log_mg.close()

