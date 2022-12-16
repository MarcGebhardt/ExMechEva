# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 12:09:22 2021

@author: mgebhard
"""
import numpy as np
import pandas as pd
from scipy import stats as spstat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sns.set_style("whitegrid")
sns.set_theme(font="segoe ui")

import Eva_common as Evac

# plt.rcParams['figure.figsize'] = [6.3,3.54]
# plt.rcParams['figure.dpi'] = 150
# plt.rcParams['font.size']= 8.0
#plt.rcParams['lines.markersize','lines.linewidth']= [6.0, 1.5]
# plt.rcParams['lines.linewidth']= 1.0
# plt.rcParams['lines.markersize']= 5.0
# plt.rcParams['markers.fillstyle']= 'none'

# sns.set_theme(style="whitegrid", font="segoe ui",
#               rc={'xtick.bottom': True,'xtick.top': False,
#                   'ytick.left': True,'ytick.right': False,
#                   'xtick.color': '.10','ytick.color': '.10'})

# sns.set_theme(style="whitegrid", font="segoe ui")
sns.set_theme(context="paper",style="whitegrid",
              font="segoe ui",palette="tab10",
              rc={'figure.dpi': 300.0,'figure.figsize': [16/2.54, 8/2.54],
                  'font.size':9, 'ytick.alignment': 'center',
                  'axes.titlesize':9, 'axes.titleweight': 'bold',
                  'axes.labelsize':9,
                  'xtick.labelsize': 9,'ytick.labelsize': 9,
                  'legend.title_fontsize': 9,'legend.fontsize': 9,
                  'lines.linewidth': 1.0,'markers.fillstyle': 'none'})
# ,rc={'figure.figsize':[6.3,3.54]}
# ,rc={'font.size':9}
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



# =============================================================================
#%% 1 Inputs
path_out = 'D:/Gebhardt/Veröffentlichungen/2021-X-X_ Präp/ADD/'
path_ausw="Auswertung/"
Donor_dict={"LEIULANA_52-17":"PT2","LEIULANA_67-17":"PT3","LEIULANA_57-17":"PT4",
            "LEIULANA_48-17":"PT5","LEIULANA_22-17":"PT6","LEIULANA_60-17":"PT7"}

no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
               'B01.1','B01.2','B01.3', 'B02.3',
               'C01.1','C01.2','C01.3', 'C02.3'] # 2.21,3.11
# no_stats_fc = ['1.11','1.12','1.21','1.22','1.31'] # 2.21,3.11
# no_stats_fc = ['1.11','1.12','1.21','1.22','1.31','2.21','3.11','3.21']
fco = -1


#Statistik
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

log_mg=open(path_out+'Evaluation_Preparation.log','w')
#%%% 1.1 Cortical
protpaths_C=[]
# path_main="F:/Messung/004-200515-Becken2-DBV/"
# name_protokoll="200515_Becken2_DBV_Protokoll_new.xlsx"
# protpaths_C.append(path_main+path_ausw+name_protokoll)
# no_stats=()
path_main="F:/Messung/005-200724_Becken3-DBV/"
name_protokoll="200724_Becken3-DBV_Protokoll_new.xlsx"
protpaths_C.append(path_main+path_ausw+name_protokoll)
# no_stats=(29)
path_main="F:/Messung/006-200917_Becken4-DBV/"
name_protokoll="200917_Becken4-DBV_Protokoll_new.xlsx"
protpaths_C.append(path_main+path_ausw+name_protokoll)
# no_stats=(24,28,31)
path_main="F:/Messung/007-201014_Becken5-DBV/"
name_protokoll="201014_Becken5-DBV_Protokoll_new.xlsx"
protpaths_C.append(path_main+path_ausw+name_protokoll)
# no_stats=(7,14,19,20,21,24)
path_main="F:/Messung/008-201125_Becken6-DBV/"
name_protokoll="201125_Becken6-DBV_Protokoll_new.xlsx"
protpaths_C.append(path_main+path_ausw+name_protokoll)
# no_stats=(24,26,32,1,14,30)
path_main="F:/Messung/009-210120_Becken7-DBV/"
name_protokoll="210120_Becken7-DBV_Protokoll_new.xlsx"
protpaths_C.append(path_main+path_ausw+name_protokoll)
# no_stats=(32)

# prot_dtyp_C={'lE': np.float64,'t1': np.float64,'t2': np.float64,'t3': np.float64,'b1': np.float64,'b2': np.float64,'b3': np.float64,'l': np.float64,'m': np.float64,'tm': np.float64,'bm': np.float64,'QA': np.float64,'V': np.float64,'Überstand': np.float64,'Rohdichte': np.float64,'DIC_End': np.float64}
prot_dtyp_C={'thickness_1': np.float64,'thickness_2': np.float64,'thickness_3': np.float64,
             'width_1': np.float64,'width_2': np.float64,'width_3': np.float64,
             'Length': np.float64,'thickness_mean': np.float64,'width_mean': np.float64,
             'Area_CS': np.float64,'Volume': np.float64,
             'Mass': np.float64,'Density_app': np.float64,
             'Length_test': np.float64,'length_overhang': np.float64,
             'Failure_code': 'O'}

dfa_C=pd.DataFrame([])
for i in protpaths_C:
    df1 = pd.read_excel(i, header=11, skiprows=range(12,13),
                        index_col=0, dtype=prot_dtyp_C)
    dfa_C = dfa_C.append(df1)
    del df1

# dfa_C.Failure_code = dfa_C.Failure_code.astype(str)
# dfa_C.Failure_code = dfa_C.Failure_code.agg(lambda x: x.split(','))
# stats_C = dfa_C.Failure_code.agg(lambda x: False if len(set(x).intersection(set(no_stats_fc)))>0 else True)


dfa_C.Failure_code  = Evac.list_cell_compiler(dfa_C.Failure_code)
stats_C = Evac.list_interpreter(dfa_C.Failure_code, no_stats_fc)


#%%% 1.2 Trabecular Bone
protpaths_T=[]
# path_main="F:/Messung/004-200514-Becken2-ADV/"
# name_protokoll="200514_Becken2_ADV_Protokoll_new.xlsx"
# no_stats=()
# protpaths_T.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/005-200723_Becken3-ADV/"
name_protokoll="200723_Becken3-ADV_Protokoll_new.xlsx"
# no_stats=(9)
protpaths_T.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/006-200916_Becken4-ADV/"
name_protokoll="200916_Becken4-ADV_Protokoll_new.xlsx"
# no_stats=()
protpaths_T.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/007-201013_Becken5-ADV/"
name_protokoll="201013_Becken5-ADV_Protokoll_new.xlsx"
# no_stats=()
protpaths_T.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/008-201124_Becken6-ADV/"
name_protokoll="201124_Becken6-ADV_Protokoll_new.xlsx"
# no_stats=()
protpaths_T.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/009-210119_Becken7-ADV/"
name_protokoll="210119_Becken7-ADV_Protokoll_new.xlsx"
# no_stats=(17,18,23)
protpaths_T.append(path_main+path_ausw+name_protokoll)


# prot_dtyp_T={'la': np.float64,'lt': np.float64,'lm': np.float64,'d': np.float64,'h': np.float64,'m': np.float64,'hp': np.float64,'QA': np.float64,'V': np.float64,'Rohdichte': np.float64}
prot_dtyp_T={'Length_x': np.float64,'Length_y': np.float64,'Length_z': np.float64,
             'Diameter': np.float64,'Height': np.float64,
             'Area_CS': np.float64,'Volume': np.float64,
             'Mass': np.float64,'Density_app': np.float64,
             'Length_test': np.float64,
             'Failure_code': 'O'}


dfa_T=pd.DataFrame([])
for i in protpaths_T:
    df2 = pd.read_excel(i, header=11, skiprows=range(12,13), 
                        index_col=0, dtype=prot_dtyp_T)
    dfa_T = dfa_T.append(df2)
    del df2

# dfa_T.Failure_code = dfa_T.Failure_code.astype(str)
# dfa_T.Failure_code = dfa_T.Failure_code.agg(lambda x: x.split(','))

# stats_T = dfa_T.Failure_code.agg(lambda x: False if len(set(x).intersection(set(no_stats_fc)))>0 else True)

dfa_T.Failure_code  = Evac.list_cell_compiler(dfa_T.Failure_code)
stats_T = Evac.list_interpreter(dfa_T.Failure_code, no_stats_fc)


#%%% 1.2 Soft Tissue
protpaths_S=[]
# path_main="F:/Messung/004-200512-Becken2-AZV/"
# name_protokoll="200512_Becken2_AZV_Protokoll_new.xlsx"
# VA=False #Versuchsart unterteielt?
# no_stats=(1,2,3,4,5,18,19)
# protpaths_S.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/005-200721_Becken3-AZV/"
name_protokoll="200721_Becken3-AZV_Protokoll_new.xlsx"
# VA=False #Versuchsart unterteielt?
# no_stats=()
protpaths_S.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/006-200910_Becken4-AZV/"
name_protokoll="200910_Becken4-AZV_Protokoll_new.xlsx"
# VA=False #Versuchsart unterteielt?
# no_stats=(1,7,8) #evtl auch 2,5,20,11
protpaths_S.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/007-201009_Becken5-AZV/"
name_protokoll="201009_Becken5-AZV_Protokoll_new.xlsx"
# VA=False #Versuchsart unterteielt?
# no_stats=(1,2,5,11,12,16,17,18,19,24,25) #evtl auch 10,13,14
protpaths_S.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/008-201117_Becken6-AZV/"
name_protokoll="201117_Becken6-AZV_Protokoll_new.xlsx"
# VA=False #Versuchsart unterteielt?
# no_stats=(2,3,4,8,10,11,12,14,15,23,25) #evtl auch 9,17
protpaths_S.append(path_main+path_ausw+name_protokoll)
path_main="F:/Messung/009-210114_Becken7-AZV/"
name_protokoll="210114_Becken7-AZV_Protokoll_new.xlsx"
# VA=False #Versuchsart unterteielt?
# no_stats=(15,16,18,23,29,17) #evtl auch 9,10,11,17,19,24
protpaths_S.append(path_main+path_ausw+name_protokoll)

prot_dtyp_S={'thickness_1': np.float64,'thickness_2': np.float64,
             'width_1': np.float64,'width_2': np.float64,
             'thickness_mean': np.float64,'width_mean': np.float64,
             'Area_CS': np.float64,'Volume': np.float64,
             'Mass': np.float64,'Density_app': np.float64,
             'Length_test': np.float64,'Lenght_clamp': np.float64,
             'Force_max_det': np.float64, 'Torque_clamp': np.float64,
             'Failure_code': 'O'}
dfa_S=pd.DataFrame([])
for i in protpaths_S:
    # df1=pd.read_excel(i, header=11, skiprows=range(12,13), index_col=1, dtype=prot_dtyp_S)
    # del df1['Unnamed: 0']
    # df1=pd.read_excel(i, header=[10,11], skiprows=range(12,13), index_col=0)
    df3=pd.read_excel(i, header=11, skiprows=range(12,13),
                      index_col=0, dtype=prot_dtyp_S)
    dfa_S=dfa_S.append(df3)
    del df3

# dfa_S.Failure_code = dfa_S.Failure_code.astype(str)
# dfa_S.Failure_code = dfa_S.Failure_code.agg(lambda x: x.split(','))

# stats_S = dfa_S.Failure_code.agg(lambda x: False if len(set(x).intersection(set(no_stats_fc)))>0 else True)


dfa_S.Failure_code  = Evac.list_cell_compiler(dfa_S.Failure_code)
stats_S = Evac.list_interpreter(dfa_S.Failure_code, no_stats_fc)


#%% 2 Analysis
#%%% 2.1 Cortical Bone
#%%%% Geometry and Count
# print(dfa_C[stats_C==False].loc(axis=1)['No','Failure_code'])
C_w_mean = dfa_C[stats_C].loc(axis=1)['width_1','width_2','width_3'].to_numpy().mean()
C_w_std  = dfa_C[stats_C].loc(axis=1)['width_1','width_2','width_3'].to_numpy().std()

Evac.MG_strlog("\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog('\nCortical Bone:', log_mg, 1, printopt=False)
Evac.MG_strlog('\n   Width: %.2f \u00B1 %.2f'%(C_w_mean,C_w_std),
               log_mg, 1, printopt=False)

ist_sum_CB=0
soll_sum=0
for group in dfa_C[stats_C].groupby('Donor'):
    ist = dfa_C[stats_C].groupby('Donor').count().iloc(axis=1)[0][group[0]]
    soll = dfa_C.groupby('Donor').count().iloc(axis=1)[0][group[0]]
    txt=('\n   ',group[0],':  %d / %d'%(ist,soll))
    Evac.MG_strlog(''.join(txt),log_mg, 1, printopt=False)
    ist_sum_CB+=ist
    soll_sum+=soll
Evac.MG_strlog('\n   Total           :  %d / %d'%(ist_sum_CB,soll_sum),
               log_mg, 1, printopt=False)

#%%%% Histograms
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.histplot(dfa_C[stats_C].loc(axis=1)['width_1','width_2','width_3'],
                 stat='count', bins=20,
                  ax=ax1, kde=True, color=sns.color_palette()[0],
                  edgecolor=None, legend = True, alpha=0.25)
ax1.set_title('Histogram of cortical bone specimen widths (N=%d)'%ist_sum_CB)
ax1.set_xlabel('Width in mm')
ax1.set_ylabel('Count')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'CB-Hist-Width.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
# ax = sns.histplot(dfa_C[stats_C].loc(axis=1)['thickness_1','thickness_2','thickness_3','thickness_mean'],
ax = sns.histplot(dfa_C[stats_C].loc(axis=1)['thickness_mean'],
                 stat='count', bins=20,
                  ax=ax1, kde=True, color=sns.color_palette()[0],
                  edgecolor=None, legend = True, alpha=0.25)
ax1.set_title('Histogram of cortical bone specimen mean thickness (N=%d)'%ist_sum_CB)
ax1.set_xlabel('Thickness in mm')
ax1.set_ylabel('Count')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'CB-Hist-Thick.png')
plt.show()

#%%%% Additional (Thickness to Donor and Harvesting-Location)
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
h=dfa_C.Origin
for i in relist:
    h=h.str.replace(i,'')
for i in h.index:
    if h[i].endswith(' '): h[i]=h[i][:-1]
h2=h.map(Locdict)
if (h2.isna()).any(): print('Locdict have missing/wrong values! \n   (Lnr: %s)'%['{:s}'.format(i) for i in h2.loc[h2.isna()].index])
dfa_C.insert(3,'Origin_short',h)
dfa_C.insert(4,'Origin_sshort',h2)
del h, h2

# df=pd.melt(dfa_C[stats_C], id_vars=['Origin_short'], value_vars=['thickness_mean'])
# df.sort_values(['Origin_short'],inplace=True)
# fig, ax1 = plt.subplots()
# ax1.grid(True)
# # ax = sns.histplot(dfa_C[stats_C].loc(axis=1)['thickness_1','thickness_2','thickness_3','thickness_mean'],
# ax = sns.histplot(df, x='value', hue='Origin_short',
#                  stat='count', bins=20,
#                   ax=ax1, kde=True, color=sns.color_palette()[9],
#                   edgecolor=None, legend = True, alpha=0.25)
# ax1.set_title('Histogram of cortical bone specimen mean thickness (N=%d)'%ist_sum_CB)
# ax1.set_xlabel('Thickness in mm')
# ax1.set_ylabel('Count')
# fig.suptitle('')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

df=pd.melt(dfa_C[stats_C], id_vars=['Origin_sshort'], value_vars=['thickness_mean'])
# df.sort_values(['Origin_sshort'],inplace=True)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Origin_sshort", y="value",
                    data=df, ax=ax1, dodge=True, edgecolor="black",
                    linewidth=.5, alpha=.5, size=2)
ax1.set_title('Mean thickness of cortical bone specimen by harvesting location (N=%d)'%ist_sum_CB)
ax1.set_xlabel('Location')
ax1.set_ylabel('Thickness in mm')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'CB-Box-Thick_Loc.png')
plt.show()

df=pd.melt(dfa_C[stats_C], id_vars=['Donor'], value_vars=['thickness_mean'])
df.Donor.replace(Donor_dict,inplace=True)
# df.sort_values(['Origin_sshort'],inplace=True)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(x="Donor", y="value", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Donor", y="value",
                    data=df, ax=ax1, dodge=True, edgecolor="black",
                    linewidth=.5, alpha=.5, size=2)
ax1.set_title('Mean thickness of cortical bone specimen by harvesting location (N=%d)'%ist_sum_CB)
ax1.set_xlabel('Cadaver')
ax1.set_ylabel('Thickness in mm')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'CB-Box-Thick_Don.png')
plt.show()

df=pd.melt(dfa_C[stats_C], id_vars=['Donor','Origin_sshort'], value_vars=['thickness_mean'])
df.Donor.replace(Donor_dict,inplace=True)
fig, ax1 = plt.subplots()
ax1.grid(True)
# ax = sns.barplot(x="Origin_sshort", y="value", hue="Donor",
#                   data=df, ax=ax1, errwidth=1, capsize=.1)
ax = sns.boxplot(x="Origin_sshort", y="value", hue="Donor", data=df, ax=ax1, 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"12","alpha":0.75})
ax = sns.swarmplot(x="Origin_sshort", y="value", hue="Donor",
                    data=df, ax=ax1, dodge=True, edgecolor="black",
                    linewidth=.5, alpha=.5, size=2)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[0:5], labels[0:5], title='Donor', loc="best")
# ax1.legend(title='Donor',loc="best", ncol=2)
ax1.set_title('Mean thickness of cortical bone specimen by harvesting location and donor (N=%d)'%ist_sum_CB)
ax1.set_xlabel('Location')
ax1.set_ylabel('Thickness in mm')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'CB-Box-Thick_Loc_Don.png')
plt.show()


#%%%% Statistics
Evac.MG_strlog("\n "+"-"*100,log_mg,1,printopt=False)
Evac.MG_strlog("\n Distribution tests:",
               log_mg,1,printopt=False)
txt=("\n  - Statistical test (Shapiro-Wilks):")
stat, p = spstat.shapiro(dfa_C[stats_C].loc[:,'thickness_mean'].dropna())
txt+=("\n    Statistics of %s = %.3e, p= %.3e" % ('thickness_mean',stat, p))
if p > alpha:
    txt+=("    -> Sample looks Gaussian (fail to reject H0)")
else: 
    txt+=("    -> Sample does not look Gaussian (reject H0)")
Evac.MG_strlog(txt,log_mg,1,printopt=False)

txt=("\n  - Statistical test (D’Agostino’s K^2):")
stat, p = spstat.normaltest(dfa_C[stats_C].loc[:,'thickness_mean'].dropna())
txt+=("\n    Statistics of %s = %.3e, p= %.3e" % ('thickness_mean',stat, p))
if p > alpha:
    txt+=("    -> Sample looks Gaussian (fail to reject H0)")
else: 
    txt+=("    -> Sample does not look Gaussian (reject H0)")
Evac.MG_strlog(txt,log_mg,1,printopt=False)



txt,_,T = Evac.group_ANOVA_MComp(df=dfa_C[stats_C], groupby='Origin_short', ano_Var='thickness_mean',
                                 group_str='Origin', ano_str='Thickness',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
d,txt2 = Evac.MComp_interpreter(T)

Evac.MG_strlog("\n "+"-"*100, log_mg, 1, printopt=False)
Evac.MG_strlog("\n  %s thickness-harvesting location:"%mpop,
               log_mg, 1, printopt=False)
Evac.MG_strlog(Evac.str_indent(txt), log_mg, 1, printopt=False)
Evac.MG_strlog(Evac.str_indent(pd.Series(d)), log_mg, 1, printopt=False)
txt,_,T = Evac.group_ANOVA_MComp(df=dfa_C[stats_C], groupby='Donor', ano_Var='thickness_mean',
                                 group_str='Donor', ano_str='Thickness',
                                 mpop=mpop, alpha=alpha,  group_ren={}, **MComp_kws)
d,txt2 = Evac.MComp_interpreter(T)
Evac.MG_strlog("\n\n  %s thickness-Donor:"%mpop,
               log_mg, 1, printopt=False)
Evac.MG_strlog(Evac.str_indent(txt), log_mg, 1, printopt=False)
Evac.MG_strlog(Evac.str_indent(pd.Series(d)), log_mg, 1, printopt=False)

#%%%% Assessment Codes
# fc_unique_C = to_1D(dfa_C.Failure_code).drop_duplicates().sort_values().values
# fc_unique_C = np.delete(fc_unique_C,fc_unique_C=='nan')
# fc_b_df_C = boolean_df(dfa_C.Failure_code, fc_unique_C).astype(int)

fc_b_df_C, fc_unique_C = Evac.Failure_code_bool_df(dfa_C.Failure_code, 
                                      sep=',',level=2, strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)

fc_b_df_C_all_sum = fc_b_df_C.sum().sort_values(ascending=False)
fc_b_df_C_fail_sum = fc_b_df_C[stats_C==False].sum().sort_values(ascending=False)
fc_b_df_C_nofail_sum = fc_b_df_C[stats_C==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n "+"-"*100, log_mg, 1, printopt=False)
Evac.MG_strlog("\n  Failure codes frequency: (first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_C_all_sum.iloc[0:fco][fc_b_df_C_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_C_fail_sum.iloc[0:fco][fc_b_df_C_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_C_nofail_sum.iloc[0:fco][fc_b_df_C_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)

fc_b_df_C_mat = np.dot(fc_b_df_C.T, fc_b_df_C)
fc_b_df_C_freq = pd.DataFrame(fc_b_df_C_mat, columns = fc_unique_C, index = fc_unique_C)

# fig, ax = plt.subplots()
# ax.set_title('Cortical Bone - failure codes frequence')
# sns.heatmap(fc_b_df_C_freq, cmap = "Reds",ax=ax)
# ax.set_xlabel('Failure code')
# ax.set_ylabel('Failure code')
# plt.xticks(rotation=90)
# fig.tight_layout()
# plt.show()
# plt.close(fig)

ind = [(True if x in no_stats_fc else False) for x in fc_unique_C]
fc_b_df_C_hm = fc_b_df_C_freq.loc[ind,np.invert(ind)]

fig, ax = plt.subplots()
ax.set_title('Occurrence of assessment codes for cortical bone specimens\n(only exclusion combination)')
sns.heatmap(fc_b_df_C_hm.loc(axis=1)[np.invert((fc_b_df_C_hm==0).all())],
           cmap = "Reds",annot=True,ax=ax)
ax.set_xlabel('Assessment code')
ax.set_ylabel('Assessment code (excluding)')
ax1.tick_params(axis='y', labelrotation=90)
fig.tight_layout()
plt.savefig(path_out+'CB-Failure_codes.png')
plt.show()
plt.close(fig)

fc_b_df_C, fc_unique_C = Evac.Failure_code_bool_df(dfa_C.Failure_code, 
                                      sep=',',level=1, strength=[2,3],
                                      drop_duplicates=False, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)

fc_b_df_C_all_sum = fc_b_df_C.sum().sort_values(ascending=False)
fc_b_df_C_fail_sum = fc_b_df_C[stats_C==False].sum().sort_values(ascending=False)
fc_b_df_C_nofail_sum = fc_b_df_C[stats_C==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  Failure codes frequency: (only strength 2 and 3, first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_C_all_sum.iloc[0:fco][fc_b_df_C_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_C_fail_sum.iloc[0:fco][fc_b_df_C_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_C_nofail_sum.iloc[0:fco][fc_b_df_C_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)

fc_b_df_C, fc_unique_C = Evac.Failure_code_bool_df(dfa_C.Failure_code, 
                                      sep=',',level='F', strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)

fc_b_df_C_all_sum = fc_b_df_C.sum().sort_values(ascending=False)
fc_b_df_C_fail_sum = fc_b_df_C[stats_C==False].sum().sort_values(ascending=False)
fc_b_df_C_nofail_sum = fc_b_df_C[stats_C==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  Failure codes frequency: (only Failure Type, first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_C_all_sum.iloc[0:fco][fc_b_df_C_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_C_fail_sum.iloc[0:fco][fc_b_df_C_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_C_nofail_sum.iloc[0:fco][fc_b_df_C_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)


#%%% 2.2 Trabecular Bone
#%%%% Geometry and Count
# print(dfa_T[stats_T==False].loc(axis=1)['No','Failure_code'])
dfa_T[stats_T].loc(axis=1)['Length_x','Length_y','Length_z'].agg(['mean','std'])
dfa_T[stats_T].iloc(axis=1)[0].count()

T_w_mean = dfa_T[stats_T].loc(axis=1)['Length_x','Length_y','Length_z'].to_numpy().mean()
T_w_std  = dfa_T[stats_T].loc(axis=1)['Length_x','Length_y','Length_z'].to_numpy().std()

Evac.MG_strlog("\n\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog('\nTrabecular Bone:', log_mg, 1, printopt=False)
Evac.MG_strlog('\n   Width: %.2f \u00B1 %.2f'%(T_w_mean,T_w_std),
               log_mg, 1, printopt=False)
Wtest=dfa_T[stats_T].loc(axis=1)['Length_x','Length_y','Length_z']

Wtest['mean']=Wtest.loc(axis=1)['Length_x','Length_y','Length_z'].mean(axis=1)
Wtest['std']=Wtest.loc(axis=1)['Length_x','Length_y','Length_z'].std(axis=1)

Wtest['dx']=Wtest.eval('(Length_x - mean) / mean')
Wtest['dy']=Wtest.eval('(Length_y - mean) / mean')
Wtest['dz']=Wtest.eval('(Length_z - mean) / mean')

Wtest['dcube_1'] = Wtest.eval('(dx**2 + dy**2 + dz**2)**0.5') # Vektorielle Abweichung
Wtest['dcube_2'] = Wtest.eval('(abs(dx) + abs(dy) + abs(dz)) / 3') # Mittlere absolute Abweichung zum Mittelwert
Wtest['dcube_3'] = Wtest.eval('std / 3**0.5 / mean') # Standardabweichung des Mittelwerts zu Mittelwert
Evac.MG_strlog('\n   cubic deviation 1:  mean %.2f %% \u00B1 %.2f %% (%.2f - %.2f %%)'%(*Wtest['dcube_1'].agg(['mean','std','min','max'])*100,),
               log_mg, 1, printopt=False)
Evac.MG_strlog('\n   cubic deviation 2:  mean %.2f %% \u00B1 %.2f %% (%.2f - %.2f %%)'%(*Wtest['dcube_2'].agg(['mean','std','min','max'])*100,),
               log_mg, 1, printopt=False)
Evac.MG_strlog('\n   cubic deviation 3:  mean %.2f %% \u00B1 %.2f %% (%.2f - %.2f %%)'%(*Wtest['dcube_3'].agg(['mean','std','min','max'])*100,),
               log_mg, 1, printopt=False)

Wtest['dcube_O'] = Wtest.eval('(Length_x*Length_y + Length_x*Length_z + Length_y*Length_z)/(3*mean**2)')
Wtest['dcube_V'] = Wtest.eval('Length_x*Length_y*Length_z / mean**3')
Wtest['dcube_dR'] = Wtest.eval('(Length_x**2+Length_y**2+Length_z**2)**0.5 / (mean*3**0.5)')
Evac.MG_strlog('\n   cubic deviation O:  mean %.2f %% \u00B1 %.2f %% (%.2f - %.2f %%)'%(*Wtest['dcube_O'].agg(['mean','std','min','max'])*100,),
               log_mg, 1, printopt=False)
Evac.MG_strlog('\n   cubic deviation V:  mean %.2f %% \u00B1 %.2f %% (%.2f - %.2f %%) -> used in Paper!'%(*Wtest['dcube_V'].agg(['mean','std','min','max'])*100,),
               log_mg, 1, printopt=False)
Evac.MG_strlog('\n   cubic deviation dR: mean %.2f %% \u00B1 %.2f %% (%.2f - %.2f %%)'%(*Wtest['dcube_dR'].agg(['mean','std','min','max'])*100,),
               log_mg, 1, printopt=False)


ist_sum_TB=0
soll_sum=0
for group in dfa_T[stats_T].groupby('Donor'):
    ist = dfa_T[stats_T].groupby('Donor').count().iloc(axis=1)[0][group[0]]
    soll = dfa_T.groupby('Donor').count().iloc(axis=1)[0][group[0]]
    txt=('\n   ',group[0],':  %d / %d'%(ist,soll))
    Evac.MG_strlog(''.join(txt),log_mg, 1, printopt=False)
    ist_sum_TB+=ist
    soll_sum+=soll
Evac.MG_strlog('\n   Total           :  %d / %d'%(ist_sum_TB,soll_sum),
               log_mg, 1, printopt=False)

#%%%% Histograms
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.histplot(Wtest['dcube_V']*100,
                 stat='count', bins=20,
                  ax=ax1, kde=True, color=sns.color_palette()[0],
                  edgecolor=None, legend = True, alpha=0.25)
ax1.set_title('Histogram of trabecular bone specimen cubicity (N=%d)'%ist_sum_TB)
ax1.set_xlabel('Cubicity in %')
ax1.set_ylabel('Count')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'TB-Hist_Cube.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
# ax = sns.histplot(Wtest[['Length_x','Length_y','Length_z']],
ax = sns.histplot(Wtest[['Length_x','Length_y','Length_z','mean']],
                 stat='count', bins=20,
                  ax=ax1, kde=True, color=sns.color_palette()[0],
                  edgecolor=None, legend = True, alpha=0.25)
ax1.set_title('Histogram of trabecular bone specimen lengths (N=%d)'%ist_sum_TB)
ax1.set_xlabel('Length in mm')
ax1.set_ylabel('Count')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'TB-Hist_Length.png')
plt.show()

# Wtest['dmean']=Wtest.loc(axis=1)['dx','dy','dz'].mean(axis=1)
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.histplot(Wtest[['dx','dy','dz']],
# ax = sns.histplot(Wtest[['dx','dy','dz','dmean']],
                 stat='count', bins=20,
                  ax=ax1, kde=True, color=sns.color_palette()[0],
                  edgecolor=None, legend = True, alpha=0.25)
ax1.set_title('Histogram of trabecular bone specimen length deviation from mean length (N=%d)'%ist_sum_TB)
ax1.set_xlabel('Deviation')
ax1.set_ylabel('Count')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'TB-Hist_Length-Deviation.png')
plt.show()



#%%%% Assessment Codes
# fc_unique_T = to_1D(dfa_T.Failure_code).drop_duplicates().sort_values().values
# fc_unique_T = np.delete(fc_unique_T,fc_unique_T=='nan')
# fc_b_df_T = boolean_df(dfa_T.Failure_code, fc_unique_T).astype(int)

fc_b_df_T, fc_unique_T = Evac.Failure_code_bool_df(dfa_T.Failure_code, 
                                      sep=',',level=2, strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_T_all_sum = fc_b_df_T.sum().sort_values(ascending=False)
fc_b_df_T_fail_sum = fc_b_df_T[stats_T==False].sum().sort_values(ascending=False)
fc_b_df_T_nofail_sum = fc_b_df_T[stats_T==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n "+"-"*100, log_mg, 1, printopt=False)
Evac.MG_strlog("\n  Failure codes frequency: (first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_T_all_sum.iloc[0:fco][fc_b_df_T_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_T_fail_sum.iloc[0:fco][fc_b_df_T_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_T_nofail_sum.iloc[0:fco][fc_b_df_T_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)

fc_b_df_T_mat = np.dot(fc_b_df_T.T, fc_b_df_T)
fc_b_df_T_freq = pd.DataFrame(fc_b_df_T_mat, columns = fc_unique_T, index = fc_unique_T)

# fig, ax = plt.subplots()
# ax.set_title('Trabecular Bone - failure codes frequence')
# sns.heatmap(fc_b_df_T_freq, cmap = "Reds",ax=ax)
# ax.set_xlabel('Failure code')
# ax.set_ylabel('Failure code')
# plt.xticks(rotation=90)
# fig.tight_layout()
# plt.show()
# plt.close(fig)

ind = [(True if x in no_stats_fc else False) for x in fc_unique_T]
fc_b_df_T_hm = fc_b_df_T_freq.loc[ind,np.invert(ind)]

fig, ax = plt.subplots()
ax.set_title('Occurrence of assessment codes for trabecular bone specimens\n(only exclusion combination)')
sns.heatmap(fc_b_df_T_hm.loc(axis=1)[np.invert((fc_b_df_T_hm==0).all())],
           cmap = "Reds",annot=True,ax=ax)
ax.set_xlabel('Assessment code')
ax.set_ylabel('Assessment code (excluding)')
ax1.tick_params(axis='y', labelrotation=90)
fig.tight_layout()
plt.savefig(path_out+'TB-Failure_codes.png')
plt.show()
plt.close(fig)

fc_b_df_T, fc_unique_T = Evac.Failure_code_bool_df(dfa_T.Failure_code, 
                                      sep=',',level=1, strength=[2,3],
                                      drop_duplicates=False, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_T_all_sum = fc_b_df_T.sum().sort_values(ascending=False)
fc_b_df_T_fail_sum = fc_b_df_T[stats_T==False].sum().sort_values(ascending=False)
fc_b_df_T_nofail_sum = fc_b_df_T[stats_T==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  Failure codes frequency: (only strength 2 and 3, first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_T_all_sum.iloc[0:fco][fc_b_df_T_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_T_fail_sum.iloc[0:fco][fc_b_df_T_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_T_nofail_sum.iloc[0:fco][fc_b_df_T_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)

fc_b_df_T, fc_unique_T = Evac.Failure_code_bool_df(dfa_T.Failure_code, 
                                      sep=',',level='F', strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_T_all_sum = fc_b_df_T.sum().sort_values(ascending=False)
fc_b_df_T_fail_sum = fc_b_df_T[stats_T==False].sum().sort_values(ascending=False)
fc_b_df_T_nofail_sum = fc_b_df_T[stats_T==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  Failure codes frequency: (only Failure Type, first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_T_all_sum.iloc[0:fco][fc_b_df_T_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_T_fail_sum.iloc[0:fco][fc_b_df_T_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_T_nofail_sum.iloc[0:fco][fc_b_df_T_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)

#%%% 2.3 Soft Tissue
#%%%% Geometry and Count
S_w_mean = dfa_S[stats_S].loc(axis=1)['width_1','width_2'].to_numpy().mean()
S_w_std  = dfa_S[stats_S].loc(axis=1)['width_1','width_2'].to_numpy().std()
Evac.MG_strlog("\n\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog('\nSoft Tissue:', log_mg, 1, printopt=False)
Evac.MG_strlog('\n   Width: %.2f \u00B1 %.2f'%(S_w_mean,S_w_std),
               log_mg, 1, printopt=False)

S_w_type = dfa_S[stats_S].groupby('Type')['width_mean'].agg(['mean','std'])
for i in S_w_type.index:
    Evac.MG_strlog('\n   %s: %.2f \u00B1 %.2f'%(i,S_w_type.loc[i,'mean'],S_w_type.loc[i,'std']),
                   log_mg, 1, printopt=False)

ist_sum_ST=0
soll_sum=0
for group in dfa_S[stats_S].groupby('Donor'):
    ist = dfa_S[stats_S].groupby('Donor').count().iloc(axis=1)[0][group[0]]
    soll = dfa_S.groupby('Donor').count().iloc(axis=1)[0][group[0]]
    txt=('\n   ',group[0],':  %d / %d'%(ist,soll))
    Evac.MG_strlog(''.join(txt),log_mg, 1, printopt=False)
    ist_sum_ST+=ist
    soll_sum+=soll
Evac.MG_strlog('\n   Total           :  %d / %d'%(ist_sum_ST,soll_sum),
               log_mg, 1, printopt=False)

#%%%% Histograms
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.histplot(dfa_S[stats_S][dfa_S[stats_S]['Type']=='Fascia'].loc(axis=1)['width_1','width_2'],
                 stat='count', bins=20,
                 ax=ax1, kde=True, color=sns.color_palette()[0],
                 edgecolor=None, legend = True, alpha=0.25)
ax1.set_title('Histogram of fascia specimen widths (N=%d)'%ist_sum_ST)
ax1.set_xlabel('Width in mm')
ax1.set_ylabel('Count')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'ST-Fascia-Hist_Width.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.histplot(dfa_S[stats_S][dfa_S[stats_S]['Type']=='Fascia'].loc(axis=1)['width_mean']/20,
                 stat='count', bins=20,
                 ax=ax1, kde=True, color=sns.color_palette()[0],
                 edgecolor=None, legend = True, alpha=0.25)
ax1.set_title('Histogram of fascia specimen widths test length ratio (N=%d)'%ist_sum_ST)
ax1.set_xlabel('Relation in -')
ax1.set_ylabel('Count')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'ST-Fascia-Hist_Testratio.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.histplot((dfa_S[stats_S][dfa_S[stats_S]['Type']=='Fascia'].loc(axis=1)['width_mean']/20)-0.5,
                 stat='count', bins=20,
                 ax=ax1, kde=True, color=sns.color_palette()[0],
                 edgecolor=None, legend = True, alpha=0.25)
ax1.set_title('Histogram of fascia specimen deviation to optimal width-length-ratio of 0.5 (N=%d)'%ist_sum_ST)
ax1.set_xlabel('Deviation')
ax1.set_ylabel('Count')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'ST-Fascia-Hist_Testratio-Deviation.png')
plt.show()


#%%%% Assessment Codes
# fc_unique_S = to_1D(dfa_S.Failure_code).drop_duplicates().sort_values().values
# fc_unique_S = np.delete(fc_unique_S,fc_unique_S=='nan')
# fc_b_df_S = boolean_df(dfa_S.Failure_code, fc_unique_S).astype(int)

fc_b_df_S, fc_unique_S = Evac.Failure_code_bool_df(dfa_S.Failure_code, 
                                      sep=',',level=2, strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_S_all_sum = fc_b_df_S.sum().sort_values(ascending=False)
fc_b_df_S_fail_sum = fc_b_df_S[stats_S==False].sum().sort_values(ascending=False)
fc_b_df_S_nofail_sum = fc_b_df_S[stats_S==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n "+"-"*100, log_mg, 1, printopt=False)
Evac.MG_strlog("\n  Failure codes frequency: (first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_S_all_sum.iloc[0:fco][fc_b_df_S_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_S_fail_sum.iloc[0:fco][fc_b_df_S_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_S_nofail_sum.iloc[0:fco][fc_b_df_S_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)


fc_b_df_S_mat = np.dot(fc_b_df_S.T, fc_b_df_S)
fc_b_df_S_freq = pd.DataFrame(fc_b_df_S_mat, columns = fc_unique_S, index = fc_unique_S)

# fig, ax = plt.subplots()
# ax.set_title('Soft Tissue - failure codes frequence')
# sns.heatmap(fc_b_df_S_freq, cmap = "Reds",ax=ax)
# ax.set_xlabel('Failure code')
# ax.set_ylabel('Failure code')
# plt.xticks(rotation=90)
# fig.tight_layout()
# plt.show()
# plt.close(fig)

ind = [(True if x in no_stats_fc else False) for x in fc_unique_S]
fc_b_df_S_hm = fc_b_df_S_freq.loc[ind,np.invert(ind)]

fig, ax = plt.subplots()
ax.set_title('Occurrence of assessment codes for soft tissue specimens\n(only exclusion combination)')
sns.heatmap(fc_b_df_S_hm.loc(axis=1)[np.invert((fc_b_df_S_hm==0).all())],
           cmap = "Reds",annot=True,ax=ax)
ax.set_xlabel('Assessment code')
ax.set_ylabel('Assessment code (excluding)')
ax1.tick_params(axis='y', labelrotation=90)
fig.tight_layout()
plt.savefig(path_out+'ST-Failure_codes.png')
plt.show()
plt.close(fig)

fc_b_df_S, fc_unique_S = Evac.Failure_code_bool_df(dfa_S.Failure_code, 
                                      sep=',',level=1, strength=[2,3],
                                      drop_duplicates=False, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_S_all_sum = fc_b_df_S.sum().sort_values(ascending=False)
fc_b_df_S_fail_sum = fc_b_df_S[stats_S==False].sum().sort_values(ascending=False)
fc_b_df_S_nofail_sum = fc_b_df_S[stats_S==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  Failure codes frequency: (only strength 2 and 3, first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_S_all_sum.iloc[0:fco][fc_b_df_S_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_S_fail_sum.iloc[0:fco][fc_b_df_S_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_S_nofail_sum.iloc[0:fco][fc_b_df_S_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)

fc_b_df_S, fc_unique_S = Evac.Failure_code_bool_df(dfa_S.Failure_code, 
                                      sep=',',level='F', strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_S_all_sum = fc_b_df_S.sum().sort_values(ascending=False)
fc_b_df_S_fail_sum = fc_b_df_S[stats_S==False].sum().sort_values(ascending=False)
fc_b_df_S_nofail_sum = fc_b_df_S[stats_S==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  Failure codes frequency: (only Failure Type, first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_S_all_sum.iloc[0:fco][fc_b_df_S_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_S_fail_sum.iloc[0:fco][fc_b_df_S_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_S_nofail_sum.iloc[0:fco][fc_b_df_S_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)

#%%% 2.4 All
#%%%% Assessment Codes
dfa_All=dfa_C.append(dfa_T).append(dfa_S).loc(axis=1)['Failure_code','Note']
# stats_All = dfa_All.Failure_code.agg(lambda x: False if len(set(x).intersection(set(no_stats_fc)))>0 else True)

stats_All = Evac.list_interpreter(dfa_All.Failure_code, no_stats_fc)

fc_unique_All = to_1D(dfa_All.Failure_code).drop_duplicates().sort_values().values
fc_unique_All = np.delete(fc_unique_All,fc_unique_All=='nan')
fc_b_df_All = boolean_df(dfa_All.Failure_code, fc_unique_All).astype(int)

fc_b_df_All, fc_unique_All = Evac.Failure_code_bool_df(dfa_All.Failure_code, 
                                      sep=',',level=2, strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_All_all_sum = fc_b_df_All.sum().sort_values(ascending=False)
fc_b_df_All_fail_sum = fc_b_df_All[stats_All==False].sum().sort_values(ascending=False)
fc_b_df_All_nofail_sum = fc_b_df_All[stats_All==True].sum().sort_values(ascending=False)

Evac.MG_strlog("\n\n "+"="*100,log_mg,1,printopt=False)
Evac.MG_strlog('\nAll:', log_mg, 1, printopt=False)
Evac.MG_strlog("\n  Failure codes frequency: (first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_All_all_sum.iloc[0:fco][fc_b_df_All_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_All_fail_sum.iloc[0:fco][fc_b_df_All_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_All_nofail_sum.iloc[0:fco][fc_b_df_All_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)


fc_b_df_All_mat = np.dot(fc_b_df_All.T, fc_b_df_All)
fc_b_df_All_freq = pd.DataFrame(fc_b_df_All_mat, columns = fc_unique_All, index = fc_unique_All)
ind = [(True if x in no_stats_fc else False) for x in fc_unique_All]
fc_b_df_All_hm = fc_b_df_All_freq.loc[ind,np.invert(ind)]
fc_b_df_All_ht = fc_b_df_All_freq.loc[ind,:]

fig, ax = plt.subplots()
ax.set_title('Occurrence of assessment codes for all tissue type specimens\n(only exclusion combination)')
sns.heatmap(fc_b_df_All_hm.loc(axis=1)[np.invert((fc_b_df_All_hm==0).all())],
           cmap = "Reds",annot=True,ax=ax)
ax.set_xlabel('Assessment code')
ax.set_ylabel('Assessment code (excluding)')
ax.tick_params(axis='y', labelrotation=90)
fig.tight_layout()
plt.savefig(path_out+'ALL-Failure_codes.png')
plt.show()
plt.close(fig)

fc_b_df_All, fc_unique_All = Evac.Failure_code_bool_df(dfa_All.Failure_code, 
                                      sep=',',level=1, strength=[2,3],
                                      drop_duplicates=False, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_All_all_sum = fc_b_df_All.sum().sort_values(ascending=False)
fc_b_df_All_fail_sum = fc_b_df_All[stats_All==False].sum().sort_values(ascending=False)
fc_b_df_All_nofail_sum = fc_b_df_All[stats_All==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  Failure codes frequency: (only strength 2 and 3, first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_All_all_sum.iloc[0:fco][fc_b_df_All_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_All_fail_sum.iloc[0:fco][fc_b_df_All_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_All_nofail_sum.iloc[0:fco][fc_b_df_All_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)


fc_b_df_All, fc_unique_All = Evac.Failure_code_bool_df(dfa_All.Failure_code, 
                                      sep=',',level='F', strength=[1,2,3],
                                      drop_duplicates=True, sort_values=True,
                                      exclude =['nan'],
                                      replace_whitespaces=True, as_int=True)
fc_b_df_All_all_sum = fc_b_df_All.sum().sort_values(ascending=False)
fc_b_df_All_fail_sum = fc_b_df_All[stats_All==False].sum().sort_values(ascending=False)
fc_b_df_All_nofail_sum = fc_b_df_All[stats_All==True].sum().sort_values(ascending=False)
Evac.MG_strlog("\n\n  Failure codes frequency: (only Failure Type, first %d)"%fco,
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    all:     %s"%fc_b_df_All_all_sum.iloc[0:fco][fc_b_df_All_all_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    fail:    %s"%fc_b_df_All_fail_sum.iloc[0:fco][fc_b_df_All_fail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)
Evac.MG_strlog("\n    no fail: %s"%fc_b_df_All_nofail_sum.iloc[0:fco][fc_b_df_All_nofail_sum!=0].to_dict(),
               log_mg, 1, printopt=False)


fc_wrongid,fc_wrong=Evac.Failure_code_checker(dfa_All.Failure_code)
Evac.MG_strlog('\n\n  Wrong codes: %s'%fc_wrong, log_mg, 1, printopt=False)
Evac.MG_strlog(Evac.str_indent(fc_wrongid,5), log_mg, 1, printopt=False)


# # test = ['1.21']
# test = ['A003']

# test_All = Evac.list_interpreter(dfa_All.Failure_code, test, option = 'include')

# # test_All = dfa_All.Failure_code.agg(lambda x: False if len(set(x).intersection(set(test)))>0 else True)
# dfa_All[test_All==False].loc(axis=1)['Failure_code','Note'].to_csv('D:/Gebhardt/Veröffentlichungen/2021-X-X_ Präp/ADD/'+'Failure_codes_121.csv',sep=';') 

dfa_All.loc(axis=1)['Failure_code','Note'].to_csv(path_out+'Failure_codes.csv',sep=';') 
dfa_All[stats_All==False].loc(axis=1)['Failure_code','Note'].to_csv(path_out+'Failure_codes_fail.csv',sep=';') 


#%%% 3 Plot für Paper

#%%%% Assessment Codes Overview (exclusion)
# fig, ax = plt.subplots(figsize=(8/2.54, 6/2.54))
# # sns.heatmap(fc_b_df_All_hm.loc(axis=1)[np.invert((fc_b_df_All_hm==0).all())],
# #             cmap = "Reds",annot=True,ax=ax)
# sns.heatmap(fc_b_df_All_ht.loc(axis=1)[np.invert((fc_b_df_All_ht==0).all())],
#             cmap = "Reds",annot=True,ax=ax)
# # ax.set_title('Occurrence of assessment codes for all tissue type specimens\n(only exclusion combination)')
# ax.set_title('Occurrence of assessment codes')
# ax.set_xlabel('Assessment code')
# ax.set_ylabel('Assessment code (excluding)')
# ax.tick_params(axis='y', labelrotation=90)
# fig.tight_layout()
# plt.savefig(path_out+'Paper-ALL-Assessment_codes.png')
# plt.savefig(path_out+'Paper-ALL-Assessment_codes.pdf')
# plt.show()
# plt.close(fig)

gs_kw = dict(width_ratios=[1, 1], height_ratios=[.0001,.9999])
fig, ax = plt.subplot_mosaic([['left', 'space'],['left','right']],
                              gridspec_kw=gs_kw,
                              # empty_sentinel='lower mid',
                              figsize=(16/2.54, 6/2.54),
                              constrained_layout=True)
# sns.heatmap(fc_b_df_All_hm.loc(axis=1)[np.invert((fc_b_df_All_hm==0).all())],
#            cmap = "Reds",annot=True,ax=ax['left'])
sns.heatmap(fc_b_df_All_ht.loc(axis=1)[np.invert((fc_b_df_All_ht==0).all())],
           cmap = "Reds",annot=True,ax=ax['left'])
# ax.set_title('Occurrence of assessment codes for all tissue type specimens\n(only exclusion combination)')
ax['left'].set_title('Occurrence')
ax['left'].set_xlabel('Assessment code')
ax['left'].set_ylabel('Assessment code (excluding)')
ax['left'].tick_params(axis='y', labelrotation=0)
xl=ax['left'].get_xticklabels()[2]
# xl.set_color("grey")
xl.set_fontstyle("italic")
xl=ax['left'].get_xticklabels()[3]
# xl.set_color("grey")
xl.set_fontstyle("italic")
xl=ax['left'].get_xticklabels()[5]
# xl.set_color("grey")
xl.set_fontstyle("italic")
xl=ax['left'].get_xticklabels()[6]
# xl.set_color("grey")
xl.set_fontstyle("italic")

content=[['A00 - Anatomical abnormality'],
         ['A01 - Pre-dissection or anatomical abscence'],
         ['A02 - Pre-dissection or anatomical damage'],
         ['A07 - Anatomical geometric abnormality'],
         ['B01 - Not dissected'],
         ['B02 - Damage during dissection'],
         ['B10 - Localization error during dissection'],
         ['B16 - Dissection scheme error during'],
         [' '],
         ['XYY.2 - Medium strength'],
         ['XYY.3 - Complete strength']]
axt= ax['right'].table(content, loc='center', cellLoc='left', rowLoc='center',
                        colWidths=[1],edges='open')
axt.auto_set_font_size(False)
axt.set_fontsize(9)
def set_pad_for_column(col, pad=0.1):
    cells = axt.get_celld()
    column = [cell for cell in axt.get_celld() if cell[1] == col]
    for cell in column:
        cells[cell].PAD = pad
set_pad_for_column(col=0, pad=0.01)
axt.scale(1, 1.2)
ax['right'].grid(False)
ax['right'].axis('off')
ax['space'].grid(False)
ax['space'].axis('off')
ax['space'].set_title('Description')
fig.suptitle('Evaluation of exluding assessment code combinations',fontweight="bold")
plt.savefig(path_out+'Paper-Fig07-ALL-Assessment_codes-wt.png')
plt.savefig(path_out+'Paper-Fig07-ALL-Assessment_codes-wt.pdf')
plt.show()
plt.close(fig)

#%%%% Geometrical quality control
lss = [':', '--', '-.','-']
fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(16/2.54, 12/2.54))
axt = sns.histplot(Wtest['dcube_V']*100,
                   stat='count', bins=20, ax=ax[0,0], kde=True)
ax[0,0].set_title('Trabecular bone (N=%d)\nCubicity'%ist_sum_TB)
ax[0,0].set_xlabel('Cubicity in %')
ax[0,0].set_ylabel('Count')
xl=ax[0,0].get_xticklabels()[4]
xl.set_color("green")
xl.set_fontweight("bold")

axt = sns.histplot(Wtest[['dx','dy','dz']]*100,
                   stat='count', bins=20, ax=ax[0,1], kde=True)
handles, labels = ax[0,1].get_legend_handles_labels()
Evac.tick_legend_renamer(ax=ax[0,1],
                         renamer={'dx':'x','dy':'y','dz':'z'},
                         title='Local\ndirection')
handles = ax[0,1].legend_.legendHandles[::-1]
for line, ls, handle in zip(ax[0,1].lines, lss, handles):
    line.set_linestyle(ls)
    handle.set_ls(ls)
legend_elements=[Line2D([0], [0], color=sns.color_palette("tab10")[0],
                        ls=lss[-2], label='x'),
                 Line2D([0], [0], color=sns.color_palette("tab10")[1],
                        ls=lss[-3], label='y'),
                 Line2D([0], [0], color=sns.color_palette("tab10")[2],
                        ls=lss[-4], label='z')]
ax[0,1].legend(handles=legend_elements, title='Local\ndirection', loc='best')
ax[0,1].set_title('Trabecular bone (N=%d)\nLength deviation from mean'%ist_sum_TB)
ax[0,1].set_xlabel('Relative deviation in %')
ax[0,1].set_ylabel('Count')
xl=ax[0,1].get_xticklabels()[2]
xl.set_color("green")
xl.set_fontweight("bold")

axt = sns.histplot(dfa_C[stats_C].loc(axis=1)['width_1','width_2','width_3'],
                   stat='count', bins=20, ax=ax[1,0], kde=True)
handles = ax[1,0].legend_.legendHandles[::-1]
for line, ls, handle in zip(ax[1,0].lines, lss, handles):
    line.set_linestyle(ls)
    handle.set_ls(ls)
legend_elements=[Line2D([0], [0], color=sns.color_palette("tab10")[0],
                        ls=lss[-2], label='left'),
                 Line2D([0], [0], color=sns.color_palette("tab10")[1],
                        ls=lss[-3], label='mid'),
                 Line2D([0], [0], color=sns.color_palette("tab10")[2],
                        ls=lss[-4], label='right')]
ax[1,0].legend(handles=legend_elements, title='Measurement\nlocation', loc='best')
ax[1,0].set_title('Cortical bone (N=%d)\nWidth'%ist_sum_CB)
ax[1,0].set_xlabel('Width in mm')
ax[1,0].set_ylabel('Count')
# xl=ax[1,0].get_xticklabels()[3]
# xl.set_color("green")
# xl.set_fontweight("bold")

axt = sns.histplot(((dfa_S[stats_S][dfa_S[stats_S]['Type']=='Fascia'].loc(axis=1)['width_mean']/20)-0.5)*100,
                   stat='count', bins=20, ax=ax[1,1], kde=True)
ax[1,1].set_title('Fascia (N=%d)\nDeviation to width-length-ratio of 0.5'%ist_sum_ST)
ax[1,1].set_xlabel('Relative deviation in  %')
ax[1,1].set_ylabel('Count')
xl=ax[1,1].get_xticklabels()[1]
xl.set_color("green")
xl.set_fontweight("bold")

fig.suptitle('Examplary results of the specimens geometrical quality control',fontweight="bold")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(path_out+'Paper-Fig08-Results-Geo.png')
plt.savefig(path_out+'Paper-Fig08-Results-Geo.pdf')
plt.show()
plt.close(fig)

#%%%% Cortical Bone observation
gs_kw = dict(width_ratios=[0.715, 1.0, 0.285], height_ratios=[1, 2])
fig, ax = plt.subplot_mosaic([['upper', 'upper', 'upper'],
                               ['Donor','Location','Pelvis']],
                              gridspec_kw=gs_kw,
                              # empty_sentinel='lower mid',
                              figsize=(16/2.54, 12/2.54),
                              constrained_layout=True)
# ax['lower left']['sharey']=ax['lower right']
axt = sns.histplot(dfa_C[stats_C].loc(axis=1)['thickness_mean'],
                   stat='count', bins=20, ax=ax['upper'], kde=True)
ax['upper'].set_title('Distribution')
ax['upper'].set_xlabel('Thickness in mm')
ax['upper'].set_ylabel('Count')

from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
with get_sample_data("D:/Gebhardt/Veröffentlichungen/2021-X-X_ Präp/IMG/05-Results/Loc-Kort_wb.png") as file:
    arr_img = plt.imread(file)
imagebox = OffsetImage(arr_img, zoom=0.105)
imagebox.image.axes = ax['Pelvis']
ab = AnnotationBbox(imagebox, (0,0),
                    xybox=(0.43, 0.5),
                    xycoords='axes fraction',frameon=False)
ax['Pelvis'].add_artist(ab)
#import matplotlib.image as mpimg
#img = mpimg.imread('D:/Gebhardt/Veröffentlichungen/2021-X-X_ Präp/IMG/05-Results/Loc-Kort.png')
ax['Pelvis'].grid(False)
ax['Pelvis'].axis('off')
#ax['Pelvis'].imshow(img,aspect='equal')

df=pd.melt(dfa_C[stats_C], id_vars=['Origin_sshort'], value_vars=['thickness_mean'])
axt = sns.boxplot(x="Origin_sshort", y="value", data=df, ax=ax['Location'], 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                             "markeredgecolor":"black", "markersize":"12",
                                             "alpha":0.75})
axt = sns.swarmplot(x="Origin_sshort", y="value",
                    data=df, ax=ax['Location'], dodge=True, edgecolor="black",
                    linewidth=.5, alpha=.5, size=2)
ax['Location'].set_title('By harvesting region')
ax['Location'].set_xlabel('Region')
# ax['lower left'].set_ylabel('Thickness in mm')
ax['Location'].set_ylabel('')
ax['Location'].tick_params(axis='y',which='both',left=False,labelleft=False)

df=pd.melt(dfa_C[stats_C], id_vars=['Donor'], value_vars=['thickness_mean'])
df.Donor.replace(Donor_dict,inplace=True)
axt = sns.boxplot(x="Donor", y="value", data=df, ax=ax['Donor'], 
                  showmeans=True, meanprops={"marker":"_", "markerfacecolor":"white",
                                             "markeredgecolor":"black", "markersize":"12",
                                             "alpha":0.75})
axt = sns.swarmplot(x="Donor", y="value",
                    data=df, ax=ax['Donor'], dodge=True, edgecolor="black",
                    linewidth=.5, alpha=.5, size=2)
ax['Donor'].set_title('By cadaver')
ax['Donor'].set_xlabel('Cadaver')
ax['Donor'].set_ylabel('Thickness in mm')
# ax['lower right'].tick_params(axis='y',which='both',left=False,labelleft=False)

fig.suptitle('Observations on the mean thickness of the cortical bone specimens (N=%d)'%ist_sum_CB,fontweight="bold")
plt.savefig(path_out+'Paper-Fig09-CB-Thick.png')
plt.savefig(path_out+'Paper-Fig09-CB-Thick.pdf')
plt.show()
plt.close(fig)

#%%% 4 Excel-out für Paper
Cols_C_out=['Designation','Type','Donor','Region','Origin',
            'thickness_1','thickness_2','thickness_3',
            'width_1','width_2','width_3',
            'Length','Mass','Date_test','Failure_code','Note']
Cols_T_out=['Designation','Type','Donor','Region','Origin',
            'Length_x','Length_y','Length_z',
            'Mass','Date_test','Failure_code','Note']
Cols_S_out=['Designation','Type','Donor','Region','Origin',
            'thickness_1','thickness_2',
            'width_1','width_2',
            'Length','Mass','Date_test','Failure_code','Note']
Donor_dict_out={"52-17":"PT2","67-17":"PT3","57-17":"PT4",
                "48-17":"PT5","22-17":"PT6","60-17":"PT7"}

dfC_o=dfa_C[Cols_C_out].copy(deep=True)
dfC_o['Statistics']=stats_C
dfC_o.rename({'thickness_1':'Thickness_1','thickness_2':'Thickness_2','thickness_3':'Thickness_3',
              'width_1':'Width_1','width_2':'Width_2','width_3':'Width_3',
              'Failure_code':'Assessment_code'},axis=1,inplace=True)
dfC_o.Date_test=pd.to_datetime(dfC_o.Date_test).dt.strftime('%Y-%m-%d')
dfT_o=dfa_T[Cols_T_out].copy(deep=True)
dfT_o['Statistics']=stats_T
dfT_o.rename({'Failure_code':'Assessment_code'},axis=1,inplace=True)
dfT_o.Date_test=pd.to_datetime(dfT_o.Date_test).dt.strftime('%Y-%m-%d')
dfS_o=dfa_S[Cols_S_out].copy(deep=True)
dfS_o['Statistics']=stats_S
dfS_o.rename({'thickness_1':'Thickness_1','thickness_2':'Thickness_2',
              'width_1':'Width_1','width_2':'Width_2',
              'Failure_code':'Assessment_code'},axis=1,inplace=True)
dfS_o.Date_test=pd.to_datetime(dfS_o.Date_test).dt.strftime('%Y-%m-%d')

# Spender-Anonymisierung
dfC_o.Donor.replace(Donor_dict_out,regex=True,inplace=True)
ind=dfC_o.index.to_series().replace(Donor_dict_out,regex=True)
dfC_o.index=ind
dfT_o.Donor.replace(Donor_dict_out,regex=True,inplace=True)
ind=dfT_o.index.to_series().replace(Donor_dict_out,regex=True)
dfT_o.index=ind
dfS_o.Donor.replace(Donor_dict_out,regex=True,inplace=True)
ind=dfS_o.index.to_series().replace(Donor_dict_out,regex=True)
dfS_o.index=ind

with pd.ExcelWriter(path_out+'Data_Preparation.xlsx') as writer:  
    dfC_o.to_excel(writer, sheet_name='Cortical Bone')
    dfT_o.to_excel(writer, sheet_name='Trabecular Bone')
    dfS_o.to_excel(writer, sheet_name='Soft Tissue')

#%%% X Close Log
log_mg.close()