# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:43:41 2022

@author: mgebhard
"""

import os
import copy
from datetime import date
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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
ptype="TBT"
# data="S1"
data="S2"
data="Complete"

no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
               'B01.1','B01.2','B01.3', 'B02.3',
               'C01.1','C01.2','C01.3', 'C02.3',
               'D01.1','D01.2','D01.3', 'D02.3',
               'F01.1','F01.2','F01.3', 'F02.3',
               'G01.1','G01.2','G01.3', 'G02.3']
var_suffix = [""] #Suffix of variants of measurements (p.E. diffferent moistures)


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
protpaths.loc['Complete','fname_in']  = "CBMM_TBT-Summary"
protpaths.loc['Complete','fname_out'] = "CBMM_TBT-Conclusion"
protpaths.loc['Complete','name'] = "two series"

protpaths.loc[:,'path_main']    = "F:/Mess_FU_Kort/"
protpaths.loc[:,'path_eva1']    = "Auswertung/"

combpaths['prot'] = protpaths['path_main']+protpaths['path_eva1']+protpaths['name_prot']
combpaths['in']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']+protpaths['fname_in']
combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']+protpaths['fname_out']

# path = "F:/Mess_FU_Kort/Auswertung/Test_py/"
# name_in   = "S1_TBT-Summary"
# name_out  = "S1_TBT-Conclusion"
# name_Head = "Moisture Manipulation Compact Bone (%s)"%date.today().strftime("%d.%m.%Y")
# path      = "F:/Mess_FU_Kort/Auswertung/Test_py/"
# name_in   = "S1_TBT-Summary"
# name_out  = "S1_TBT-Conclusion"
name_Head = "Moisture Manipulation Compact Bone (%s, %s)"%(protpaths.loc[data,'name'],
                                                           date.today().strftime("%d.%m.%Y"))

# out_full= os.path.abspath(path+name_out)
out_full= combpaths.loc[data,'out']
h5_conc = 'Summary'
h5_data = 'Test_Data'



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
dfa.Failure_code  = Evac.list_cell_compiler(dfa.Failure_code)
dfa['statistics'] = Evac.list_interpreter(dfa.Failure_code, no_stats_fc)


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
protv[['Temperature_test','Humidity_test','Mass']]=protv[['Temperature_test','Humidity_test','Mass']].astype('float')


i1 = dfa.index.to_series().apply(lambda x: '{0}'.format(x[:-1]))
i2 = dfa.index.to_series().apply(lambda x: '{0}'.format(x[-1]))
dfa.index = pd.MultiIndex.from_arrays([i1,i2], names=['Key','Variant'])

dft=pd.concat([dfa,protv],axis=1)
t = dft.Failure_code.iloc(axis=1)[0]+dft.Failure_code.iloc(axis=1)[1]
dft=dft.loc[:, ~dft.columns.duplicated()]
t2=t.apply(lambda x: x.remove('nan') if ((len(x)>1)&('nan' in x)) else x)
dft.Failure_code = t
dft['statistics'] = Evac.list_interpreter(dft.Failure_code, no_stats_fc)

#Wassergehalt zu darrtrocken
t=dft.Mass.loc(axis=0)[:,'G'].droplevel(1)
dft['DMWtoG']=(dft.Mass-t)/t
t=dft.Mass_org.loc(axis=0)[:,'G'].droplevel(1)
dft['DMWtoA']=(dft.Mass-t)/t


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
cs_num_cols = cs.select_dtypes(include=['int','float']).columns

Evac.MG_strlog("\n\n   - Max-/Minimum Deviation:",
               log_mg, printopt=False)
tmp=cs.loc(axis=1)[['DMWtoG','DEFtoB','DEFtoG','DERtoB','DERtoG']]
Evac.MG_strlog(Evac.str_indent("\n"+tmp.agg(['max','idxmax']).T.to_string(),5),
               log_mg, printopt=False)
Evac.MG_strlog(Evac.str_indent("\n"+tmp.agg(['min','idxmin']).T.to_string(),5),
               log_mg, printopt=False)

t=cs.loc(axis=1)[['Designation','DMWtoA','DMWtoG','DEFtoB','DEFtoG','DERtoB','DERtoG']].reset_index()
t['Numsch']=t.Designation.apply(lambda x: '{0}'.format(x[-3:]))
t=t.sort_values(['Designation','Variant'])
t_des=t[t.Variant<='G']
t_ads=t[t.Variant>='G']

t_ona=t.dropna().sort_values(['Designation','Variant'])
t_ona_ads=t_ads.dropna().sort_values(['Designation','Variant'])
t_ona_des=t_des.dropna().sort_values(['Designation','Variant'])


#%% Prepare
# E-Methoden zu Multiindex
c_E = cs.loc(axis=1)[cs.columns.str.startswith('E_')]
c_E.columns = c_E.columns.str.split('_', expand=True)
c_E = c_E.droplevel(0,axis=1)
c_E.columns.names=['Determination','Range','Method','Parameter']
# E-Methoden relevant
c_E_lsq = c_E.loc(axis=1)['lsq']
c_E_inc = c_E.loc(axis=1)['inc']

idx = pd.IndexSlice
c_E_lsq_m=c_E.loc(axis=1)[idx['lsq',:,:,'E']].droplevel([0],axis=1)
c_E_inc_m=c_E.loc(axis=1)[idx['inc',:,:,'meanwoso']].droplevel([0],axis=1)

c_E_lsq_Rquad  = c_E_lsq.loc(axis=1)[idx[:,:,'Rquad']].droplevel([2],axis=1)
c_E_inc_stnorm = c_E_inc.loc(axis=1)[idx[:,:,'stdnwoso']].droplevel([2],axis=1)

#%% Prepare load and unload (09.12.22)
def VIP_searcher(Vdf, col=0, sel='l', ser='F', rstart='A', rend='B'):
    sen=sel+'_'+ser
    seV=ser+sel
    if Vdf.index.str.contains('{0}{1}|{0}{2}'.format(*[seV,rstart,rend])).sum()==2:
        val=Vdf.loc[[seV+rstart,seV+rend],col].values
    else:
        val=[np.nan,np.nan]
    out={sen:tuple(val)}
    return out

data_read = pd.HDFStore(combpaths.loc[data,'in']+'_all.h5','r')
dfEt=data_read['Add_/E_inc_df']
dfVIP=data_read['Add_/VIP']
data_read.close()

cEt=pd.DataFrame([])
for i in dfEt.index:
    tmpE=dfEt.loc[i]
    tmpV=dfVIP.loc[i]
    Evar=pd.DataFrame([],columns=['l_F','l_R','u_F','u_R'])
    Evar.loc['con']=pd.Series({**VIP_searcher(tmpV,0,'l','F'),
                               **VIP_searcher(tmpV,0,'l','R'),
                               **VIP_searcher(tmpV,0,'u','F'),
                               **VIP_searcher(tmpV,0,'u','R')})
    Evar.loc['opt']=pd.Series({**VIP_searcher(tmpV,1,'l','F'),
                               **VIP_searcher(tmpV,1,'l','R'),
                               **VIP_searcher(tmpV,1,'u','F'),
                               **VIP_searcher(tmpV,1,'u','R')})
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

cEt.columns = cEt.columns.str.split('_', expand=True)
cEt.columns.names=['Load','Range','Method']
cEt=cEt.reorder_levels(['Method','Load','Range'],axis=1).sort_index(axis=1,level=0)
# cEt.to_csv(out_full+'-E_lu_FR.csv',sep=';')
cEE=cEt.loc(axis=1)['D2Mgwt',:,:]
#%% 1st Eva
#%%% Plots

# cs.boxplot(column='DMWtoG',by='Variant')
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=dft.DMWtoG.unstack(),ax=ax1)
ax1.set_title('%s\nWater content based on dry mass of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{mass,Water}$ / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Box-DMWtoG.pdf')
plt.savefig(out_full+'-Box-DMWtoG.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=dft.DMWtoA.unstack(),ax=ax1)
ax1.set_title('%s\nWater content based on original mass of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{mass,Water}$ / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Box-DMWtoA.pdf')
plt.savefig(out_full+'-Box-DMWtoA.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=cs['DEFtoB'].unstack(),ax=ax1)
ax1.set_title('%s\nDeviation of Youngs Modulus (fixed range) based on water storaged\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Box-DEFtoB.pdf')
plt.savefig(out_full+'-Box-DEFtoB.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=cs['DEFtoG'].unstack(),ax=ax1)
ax1.set_title('%s\nDeviation of Youngs Modulus (fixed range) based on dry mass\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Box-DEFtoG.pdf')
plt.savefig(out_full+'-Box-DEFtoG.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=cs['DERtoB'].unstack(),ax=ax1)
ax1.set_title('%s\nDeviation of Youngs Modulus (refined range) based on water storaged\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Box-DERtoB.pdf')
plt.savefig(out_full+'-Box-DERtoB.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.boxplot(data=cs['DERtoG'].unstack(),ax=ax1)
ax1.set_title('%s\nDeviation of Youngs Modulus (refined range) based on dry mass\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Box-DERtoG.pdf')
plt.savefig(out_full+'-Box-DERtoG.png')
plt.show()




fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(y='DMWtoA',x='Variant',hue='Numsch',data=t,ax=ax1)
ax = sns.lineplot(y='DMWtoA',x='Variant',hue='Numsch',data=t,ax=ax1, legend=False)
ax1.set_title('%s\nWater content based on original mass of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{mass,Water}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-SL-DMWtoA-Var.pdf')
plt.savefig(out_full+'-SL-DMWtoA-Var.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(y='DMWtoG',x='Variant',hue='Numsch',data=t,ax=ax1)
ax = sns.lineplot(y='DMWtoG',x='Variant',hue='Numsch',data=t,ax=ax1, legend=False)
ax1.set_title('%s\nWater content based on dry mass of the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{mass,Water}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-SL-DMWtoG-Var.pdf')
plt.savefig(out_full+'-SL-DMWtoG-Var.png')
plt.show()


fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(y='DEFtoB',x='Variant',hue='Numsch',data=t,ax=ax1)
ax = sns.lineplot(y='DEFtoB',x='Variant',hue='Numsch',data=t,ax=ax1, legend=False)
ax1.set_title('%s\nDeviation of Youngs Modulus (fixed range) based on water storaged\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-SL-DEFtoB-Var.pdf')
plt.savefig(out_full+'-SL-DEFtoB-Var.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(y='DEFtoG',x='Variant',hue='Numsch',data=t,ax=ax1)
ax = sns.lineplot(y='DEFtoG',x='Variant',hue='Numsch',data=t,ax=ax1, legend=False)
ax1.set_title('%s\nDeviation of Youngs Modulus (fixed range) based on dry mass\nof the different manipulation variants'%name_Head)
ax1.set_xlabel('Variant / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-SL-DEFtoG-Var.pdf')
plt.savefig(out_full+'-SL-DEFtoG-Var.png')
plt.show()




fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(x='DMWtoG',y='DEFtoB',hue='Numsch',data=t,ax=ax1)
ax1.set_title('%s\nYoungs Modulus (fixed range) deviation versus water content'%name_Head)
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-SL-DMWtoG-DEFtoB.pdf')
plt.savefig(out_full+'-SL-DMWtoG-DEFtoB.png')
plt.show()
fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(x='DMWtoG',y='DERtoB',hue='Numsch',data=t,ax=ax1)
ax1.set_title('%s\nYoungs Modulus (refined range) deviation versus water content'%name_Head)
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-SL-DMWtoG-DERtoB.pdf')
plt.savefig(out_full+'-SL-DMWtoG-DERtoB.png')
plt.show()



fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(x='DMWtoG',y='DEFtoG',hue='Numsch',data=t,ax=ax1)
ax1.set_title('%s\nYoungs Modulus (fixed range) deviation versus water content'%name_Head)
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-SL-DMWtoG-DEFtoG.pdf')
plt.savefig(out_full+'-SL-DMWtoG-DEFtoG.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax = sns.scatterplot(x='DMWtoG',y='DERtoG',hue='Numsch',data=t,ax=ax1)
ax1.set_title('%s\nYoungs Modulus (refined range) deviation versus water content'%name_Head)
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
h,l = ax.axes.get_legend_handles_labels()
ax.axes.legend_.remove()
ax.legend(h,l, ncol=4,title='Specimen-No.',fontsize=6.0)
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-SL-DMWtoG-DERtoG.pdf')
plt.savefig(out_full+'-SL-DMWtoG-DERtoG.png')
plt.show()





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
Lin_reg_df['Ads-DMWtoG-DEFtoG'] = MG_linRegStats(t_ona_ads['DEFtoG'], t_ona_ads['DMWtoG'],
                                                "Adsorption Deviation YM (fixed) to dry", 
                                                "Adsorption water content")

Lin_reg_df['Com-DMWtoG-DEFtoB'] = MG_linRegStats(t_ona['DEFtoB'], t_ona['DMWtoG'],
                                                "Complete Deviation YM (fixed) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DEFtoB'] = MG_linRegStats(t_ona_des['DEFtoB'], t_ona_des['DMWtoG'],
                                                "Desorption Deviation YM (fixed) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Ads-DMWtoG-DEFtoB'] = MG_linRegStats(t_ona_ads['DEFtoB'], t_ona_ads['DMWtoG'],
                                                "Adsorption Deviation YM (fixed) to saturated", 
                                                "Adsorption water content")

Lin_reg_df['Com-DMWtoA-DEFtoB'] = MG_linRegStats(t_ona['DEFtoB'], t_ona['DMWtoA'],
                                                "Complete Deviation YM (fixed) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoA-DEFtoB'] = MG_linRegStats(t_ona_des['DEFtoB'], t_ona_des['DMWtoA'],
                                                "Desorption Deviation YM (fixed) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Ads-DMWtoA-DEFtoB'] = MG_linRegStats(t_ona_ads['DEFtoB'], t_ona_ads['DMWtoA'],
                                                "Adsorption Deviation YM (fixed) to saturated", 
                                                "Adsorption water content")


Lin_reg_df['Com-DMWtoG-DERtoG'] = MG_linRegStats(t_ona['DERtoG'], t_ona['DMWtoG'],
                                                "Complete Deviation YM (refined) to dry", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DERtoG'] = MG_linRegStats(t_ona_des['DERtoG'], t_ona_des['DMWtoG'],
                                                "Desorption Deviation YM (refined) to dry", 
                                                "Desorption water content")
Lin_reg_df['Ads-DMWtoG-DERtoG'] = MG_linRegStats(t_ona_ads['DERtoG'], t_ona_ads['DMWtoG'],
                                                "Adsorption Deviation YM (refined) to dry", 
                                                "Adsorption water content")

Lin_reg_df['Com-DMWtoG-DERtoB'] = MG_linRegStats(t_ona['DERtoB'], t_ona['DMWtoG'],
                                                "Complete Deviation YM (refined) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoG-DERtoB'] = MG_linRegStats(t_ona_des['DERtoB'], t_ona_des['DMWtoG'],
                                                "Desorption Deviation YM (refined) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Ads-DMWtoG-DERtoB'] = MG_linRegStats(t_ona_ads['DERtoB'], t_ona_ads['DMWtoG'],
                                                "Adsorption Deviation YM (refined) to saturated", 
                                                "Adsorption water content")

Lin_reg_df['Com-DMWtoA-DERtoB'] = MG_linRegStats(t_ona['DERtoB'], t_ona['DMWtoA'],
                                                "Complete Deviation YM (refined) to saturated", 
                                                "Complete Deviation water content")
Lin_reg_df['Des-DMWtoA-DERtoB'] = MG_linRegStats(t_ona_des['DERtoB'], t_ona_des['DMWtoA'],
                                                "Desorption Deviation YM (refined) to saturated", 
                                                "Desorption water content")
Lin_reg_df['Ads-DMWtoA-DERtoB'] = MG_linRegStats(t_ona_ads['DERtoB'], t_ona_ads['DMWtoA'],
                                                "Adsorption Deviation YM (refined) to saturated", 
                                                "Adsorption water content")

for i in Lin_reg_df.columns:
    txt += Lin_reg_df.loc['Description',i]
    txt += Evac.str_indent(Evac.fit_report_adder(*Lin_reg_df.loc[['fit','Rquad'],i]))
    txt += Evac.str_indent(Lin_reg_df.loc['smstat',i])

Evac.MG_strlog(txt,log_mg,1,printopt=False)
#%%%% Plots
formula_in_dia = True

if formula_in_dia:
    txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoG-DEFtoG'].iloc[:3]).split(',')
    txtc = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtc,)
    txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoG-DEFtoG'].iloc[:3]).split(',')
    txtd = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtd,)
    txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Ads-DMWtoG-DEFtoG'].iloc[:3]).split(',')
    txta = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txta,)
else:
    txtc = r'$D_{E}$'
    
    
fig, axa = plt.subplots(nrows=3, ncols=1, 
                              sharex=True, sharey=False, figsize = (6.3,3*3.54))
fig.suptitle('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
axa[0].grid(True)
axa[0].set_title('Desorption and Adsorption (all Variants)')
ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t,ax=axa[0],label=txtc)
axa[0].set_ylabel(r'$D_{E,dry}$ / -')
axa[0].set_xlabel(None)
axa[0].legend()
axa[1].grid(True)
axa[1].set_title('Desorption')
ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
axa[1].set_ylabel(r'$D_{E,dry}$ / -')
axa[1].set_xlabel(None)
axa[1].legend()
axa[2].grid(True)
axa[2].set_title('Adsorption')
ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t[t.Variant>='G'],ax=axa[2],label=txta)
axa[2].set_xlabel(r'$D_{mass,Water}$ / -')
axa[2].set_ylabel(r'$D_{E,dry}$ / -')
axa[2].legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Regd-DMWtoG-DEFtoG.pdf')
plt.savefig(out_full+'-Regd-DMWtoG-DEFtoG.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t,ax=ax1, label='Complete')
ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
ax = sns.regplot(x='DMWtoG',y='DEFtoG',data=t[t.Variant>='G'],ax=ax1, label='Adsorption')
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Rega-DMWtoG-DEFtoG.pdf')
plt.savefig(out_full+'-Rega-DMWtoG-DEFtoG.png')
plt.show()

#------------
if formula_in_dia:
    txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoG-DERtoG'].iloc[:3]).split(',')
    txtc = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtc,)
    txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoG-DERtoG'].iloc[:3]).split(',')
    txtd = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtd,)
    txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Ads-DMWtoG-DERtoG'].iloc[:3]).split(',')
    txta = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txta,)
else:
    txtc = r'$D_{E}$'
    
    
fig, axa = plt.subplots(nrows=3, ncols=1, 
                              sharex=True, sharey=False, figsize = (6.3,3*3.54))
fig.suptitle('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
axa[0].grid(True)
axa[0].set_title('Desorption and Adsorption (all Variants)')
ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t,ax=axa[0],label=txtc)
axa[0].set_ylabel(r'$D_{E,dry}$ / -')
axa[0].set_xlabel(None)
axa[0].legend()
axa[1].grid(True)
axa[1].set_title('Desorption')
ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
axa[1].set_ylabel(r'$D_{E,dry}$ / -')
axa[1].set_xlabel(None)
axa[1].legend()
axa[2].grid(True)
axa[2].set_title('Adsorption')
ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t[t.Variant>='G'],ax=axa[2],label=txta)
axa[2].set_xlabel(r'$D_{mass,Water}$ / -')
axa[2].set_ylabel(r'$D_{E,dry}$ / -')
axa[2].legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Regd-DMWtoG-DERtoG.pdf')
plt.savefig(out_full+'-Regd-DMWtoG-DERtoG.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t,ax=ax1, label='Complete')
ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
ax = sns.regplot(x='DMWtoG',y='DERtoG',data=t[t.Variant>='G'],ax=ax1, label='Adsorption')
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,dry}$ / -')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Rega-DMWtoG-DERtoG.pdf')
plt.savefig(out_full+'-Rega-DMWtoG-DERtoG.png')
plt.show()

#==================
if formula_in_dia:
    txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoG-DEFtoB'].iloc[:3]).split(',')
    txtc = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtc,)
    txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoG-DEFtoB'].iloc[:3]).split(',')
    txtd = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtd,)
    txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Ads-DMWtoG-DEFtoB'].iloc[:3]).split(',')
    txta = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txta,)
else:
    txtc = r'$D_{E}$'
    
fig, axa = plt.subplots(nrows=3, ncols=1, 
                              sharex=True, sharey=False, figsize = (6.3,3*3.54))
fig.suptitle('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
axa[0].grid(True)
axa[0].set_title('Desorption and Adsorption (all Variants)')
ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t,ax=axa[0],label=txtc)
axa[0].set_ylabel(r'$D_{E,saturated}$ / -')
axa[0].set_xlabel(None)
axa[0].legend()
axa[1].grid(True)
axa[1].set_title('Desorption')
ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
axa[1].set_ylabel(r'$D_{E,saturated}$ / -')
axa[1].set_xlabel(None)
axa[1].legend()
axa[2].grid(True)
axa[2].set_title('Adsorption')
ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t[t.Variant>='G'],ax=axa[2],label=txta)
axa[2].set_xlabel(r'$D_{mass,Water}$ / -')
axa[2].set_ylabel(r'$D_{E,saturated}$ / -')
axa[2].legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Regd-DMWtoG-DEFtoB.pdf')
plt.savefig(out_full+'-Regd-DMWtoG-DEFtoB.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t,ax=ax1, label='Complete')
ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
ax = sns.regplot(x='DMWtoG',y='DEFtoB',data=t[t.Variant>='G'],ax=ax1, label='Adsorption')
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Rega-DMWtoG-DEFtoB.pdf')
plt.savefig(out_full+'-Rega-DMWtoG-DEFtoB.png')
plt.show()

#------------
if formula_in_dia:
    txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoG-DERtoB'].iloc[:3]).split(',')
    txtc = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtc,)
    txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoG-DERtoB'].iloc[:3]).split(',')
    txtd = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txtd,)
    txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Ads-DMWtoG-DERtoB'].iloc[:3]).split(',')
    txta = r'$D_{E}$ = %s $D_{W,dry}$ + %s ($R²$ = %s)'%(*txta,)
else:
    txtc = r'$D_{E}$'
    
fig, axa = plt.subplots(nrows=3, ncols=1, 
                              sharex=True, sharey=False, figsize = (6.3,3*3.54))
fig.suptitle('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
axa[0].grid(True)
axa[0].set_title('Desorption and Adsorption (all Variants)')
ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t,ax=axa[0],label=txtc)
axa[0].set_ylabel(r'$D_{E,saturated}$ / -')
axa[0].set_xlabel(None)
axa[0].legend()
axa[1].grid(True)
axa[1].set_title('Desorption')
ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
axa[1].set_ylabel(r'$D_{E,saturated}$ / -')
axa[1].set_xlabel(None)
axa[1].legend()
axa[2].grid(True)
axa[2].set_title('Adsorption')
ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t[t.Variant>='G'],ax=axa[2],label=txta)
axa[2].set_xlabel(r'$D_{mass,Water}$ / -')
axa[2].set_ylabel(r'$D_{E,saturated}$ / -')
axa[2].legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Regd-DMWtoG-DERtoB.pdf')
plt.savefig(out_full+'-Regd-DMWtoG-DERtoB.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t,ax=ax1, label='Complete')
ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
ax = sns.regplot(x='DMWtoG',y='DERtoB',data=t[t.Variant>='G'],ax=ax1, label='Adsorption')
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Rega-DMWtoG-DERtoB.pdf')
plt.savefig(out_full+'-Rega-DMWtoG-DERtoB.png')
plt.show()


#==================
if formula_in_dia:
    txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoA-DEFtoB'].iloc[:3]).split(',')
    txtc = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txtc,)
    txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoA-DEFtoB'].iloc[:3]).split(',')
    txtd = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txtd,)
    txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Ads-DMWtoA-DEFtoB'].iloc[:3]).split(',')
    txta = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txta,)
else:
    txtc = r'$D_{E}$'
    
fig, axa = plt.subplots(nrows=3, ncols=1, 
                              sharex=True, sharey=False, figsize = (6.3,3*3.54))
fig.suptitle('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
axa[0].grid(True)
axa[0].set_title('Desorption and Adsorption (all Variants)')
ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t,ax=axa[0],label=txtc)
axa[0].set_ylabel(r'$D_{E,saturated}$ / -')
axa[0].set_xlabel(None)
axa[0].legend()
axa[1].grid(True)
axa[1].set_title('Desorption')
ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
axa[1].set_ylabel(r'$D_{E,saturated}$ / -')
axa[1].set_xlabel(None)
axa[1].legend()
axa[2].grid(True)
axa[2].set_title('Adsorption')
ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t[t.Variant>='G'],ax=axa[2],label=txta)
axa[2].set_xlabel(r'$D_{mass,org}$ / -')
axa[2].set_ylabel(r'$D_{E,saturated}$ / -')
axa[2].legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Regd-DMWtoA-DEFtoB.pdf')
plt.savefig(out_full+'-Regd-DMWtoA-DEFtoB.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('%s\nRegression of Youngs Modulus (fixed range) deviation versus water content'%name_Head)
ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t,ax=ax1, label='Complete')
ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
ax = sns.regplot(x='DMWtoA',y='DEFtoB',data=t[t.Variant>='G'],ax=ax1, label='Adsorption')
ax1.set_xlabel(r'$D_{mass,Water}$ / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Rega-DMWtoA-DEFtoB.pdf')
plt.savefig(out_full+'-Rega-DMWtoA-DEFtoB.png')
plt.show()

#------------
if formula_in_dia:
    txtc = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Com-DMWtoA-DERtoB'].iloc[:3]).split(',')
    txtc = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txtc,)
    txtd = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Des-DMWtoA-DERtoB'].iloc[:3]).split(',')
    txtd = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txtd,)
    txta = '{0:5.3e},{1:5.3e},{2:.3f}'.format(*Lin_reg_df['Ads-DMWtoA-DERtoB'].iloc[:3]).split(',')
    txta = r'$D_{E}$ = %s $D_{W,org}$ + %s ($R²$ = %s)'%(*txta,)
else:
    txtc = r'$D_{E}$'
    
fig, axa = plt.subplots(nrows=3, ncols=1, 
                              sharex=True, sharey=False, figsize = (6.3,3*3.54))
fig.suptitle('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
axa[0].grid(True)
axa[0].set_title('Desorption and Adsorption (all Variants)')
ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t,ax=axa[0],label=txtc)
axa[0].set_ylabel(r'$D_{E,saturated}$ / -')
axa[0].set_xlabel(None)
axa[0].legend()
axa[1].grid(True)
axa[1].set_title('Desorption')
ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t[t.Variant<='G'],ax=axa[1],label=txtd)
axa[1].set_ylabel(r'$D_{E,saturated}$ / -')
axa[1].set_xlabel(None)
axa[1].legend()
axa[2].grid(True)
axa[2].set_title('Adsorption')
ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t[t.Variant>='G'],ax=axa[2],label=txta)
axa[2].set_xlabel(r'$D_{mass,org}$ / -')
axa[2].set_ylabel(r'$D_{E,saturated}$ / -')
axa[2].legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Regd-DMWtoA-DERtoB.pdf')
plt.savefig(out_full+'-Regd-DMWtoA-DERtoB.png')
plt.show()

fig, ax1 = plt.subplots()
ax1.grid(True)
ax1.set_title('%s\nRegression of Youngs Modulus (refined range) deviation versus water content'%name_Head)
ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t,ax=ax1, label='Complete')
ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t[t.Variant<='G'],ax=ax1, label='Desorption')
ax = sns.regplot(x='DMWtoA',y='DERtoB',data=t[t.Variant>='G'],ax=ax1, label='Adsorption')
ax1.set_xlabel(r'$D_{mass,org}$ / -')
ax1.set_ylabel(r'$D_{E,saturated}$ / -')
ax1.legend()
fig.suptitle('')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(out_full+'-Rega-DMWtoA-DERtoB.pdf')
plt.savefig(out_full+'-Rega-DMWtoA-DERtoB.png')
plt.show()

log_mg.close()