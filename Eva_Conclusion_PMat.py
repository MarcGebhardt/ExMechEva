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
#%% Functions

#%% Einlesen und auswählen
#%%% Main
Version="230920"
ptype="TBT"
ptype="ACT"
ptype="ATT"

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
    VIParams_mat=["fy","ey_opt","Uy_opt","Wy_opt",
                  "fu","eu_opt","Uu_opt","Wu_opt",
                  "fb","eb_opt","Ub_opt","Wb_opt",
                  YM_con_str,YM_opt_str]
    VIParams_rename = {'geo_MoI_mid':'MoI_mid',
                       YM_con_str:'E_con',YM_opt_str:'E_opt'}
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
    VIParams_mat=["fy","ey_con","Uy_con","Wy_con",
                  "fu","eu_con","Uu_con","Wu_con",
                  "fb","eb_con","Ub_con","Wb_con",
                  YM_con_str]
    VIParams_rename = {YM_con_str:'E_con'}
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
    VIParams_mat=["fy","ey_con","Uy_con","Wy_con",
                  "fu","eu_con","Uu_con","Wu_con",
                  "fb","eb_con","Ub_con","Wb_con",
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

log_mg=open(out_full+'.log','w')
Evac.MG_strlog(name_out, log_mg, printopt=False)
Evac.MG_strlog("\n   Paths:", log_mg, printopt=False)
Evac.MG_strlog("\n   - in:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(os.path.abspath(path+name_in+'.h5')), log_mg, printopt=False)
Evac.MG_strlog("\n   - out:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(out_full), log_mg, printopt=False)
Evac.MG_strlog("\n   - donor:", log_mg, printopt=False)
Evac.MG_strlog("\n         {}".format(path_doda), log_mg, printopt=False)
# Evac.MG_strlog("\n   Donors:"+Evac.str_indent('\n{}'.format(pd.Series(Donor_dict).to_string()),5), log_mg, printopt=False)

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
VIParams_short=[VIParams_rename.get(item,item)  for item in VIParams]

pd.DataFrame().mean()
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

#%% Data-Export
writer = pd.ExcelWriter(out_full+'.xlsx', engine = 'xlsxwriter')
# tmp=dft_comb_rel.rename(VIPar_plt_renamer,axis=1)
dfa.to_excel(writer, sheet_name='Data')
# tmp=dfa.append(cs_eva,sort=False)
# if ptype=="ATT": tmp=tmp.append(c_short_Type_eva,sort=False)
cs_short_eva.to_excel(writer, sheet_name='Summary')
if ptype=="ATT":
    c_short_Type_eva.loc['Fascia'].unstack().to_excel(writer, sheet_name='Summary-Fascia')
    c_short_Type_eva.loc['Ligament'].unstack().to_excel(writer, sheet_name='Summary-Ligament')
h_short_eva.to_excel(writer, sheet_name='Location')

writer.close()





#%% Close Log
log_mg.close()