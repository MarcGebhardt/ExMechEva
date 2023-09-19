# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:29:49 2021

@author: mgebhard
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:40:20 2021

@author: mgebhard
ToDo:
    - opt-DIC Funktionalität
    - Implementierung genutze Dehnung aus Methoden (range, plot, output)
    - Dehnung für Ausgabe auf E-Modul beziehen (relevant, wenn Auflagereindrückung nicht eliminiert)
    - Automatik Compression / Anstiegswechsel (Be-Entlasten)
        - VIP-Bestimmung
        - E-Methoden (Spannungs-/Dehnungsmaximum)
    
Changelog:
    - 21-09-16: Anpassung 6.3.2 (try except bei F4 und Ende FM-1)
    - 21-10-21: Ende Auswertung erst nach maximaler Kraft
    - 21-10-29: plastic strain shift -1 (zugehörig nun zu abgelaufenem Zyklus)
"""

#%% 0 Imports
import sys
import traceback
import pandas as pd
import numpy as np
import scipy.integrate   as scint
import scipy.interpolate as scipo
import scipy.optimize    as scopt
import lmfit
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import vg
from datetime import datetime, timedelta
import time
import warnings
import json

import Bending as Bend
import Eva_common as Evac

warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore',category=FutureWarning)

plt.rcParams['figure.figsize'] = [6.3,3.54]
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size']= 8.0
#plt.rcParams['lines.markersize','lines.linewidth']= [6.0, 1.5]
plt.rcParams['lines.linewidth']= 1.0
plt.rcParams['lines.markersize']= 4.0
plt.rcParams['markers.fillstyle']= 'none'
#pd.set_option('display.expand_frame_repr', False)


output_lvl= 1 # 0=none, 1=only text, 2=diagramms


#%% 0.1 Add. modules
def ATT_Option_reader(options):
    for o in options.index:
        if (o=='OPT_File_Meas') or (o=='OPT_File_Meas'):
            if Evac.check_empty(options[o]):
                options[o]='Test'
            else:
                options[o]=str(options[o])
        elif (o=='OPT_Measurement_file'):
            if Evac.check_empty(options[o]):
                options[o]='{"header":48,"head_names":["Time","Trigger","F_WZ","L_PM","F_PM","L_IWA1","L_IWA2"], "used_names_dict":{"Time":"Time","Force":"F_WZ","Way":["L_IWA1","L_IWA2"]}}'
            options[o]=json.loads(options[o])
            
        
        elif (o=='OPT_End'):
            if Evac.check_empty(options[o]):
                options[o]=np.nan
                
        elif (o=='OPT_pre_load_cycles'):
            if Evac.check_empty(options[o]):
                options[o]=10
            options[o]=int(options[o])
        
        elif (o=='OPT_Resampling'):
            if Evac.check_empty(options[o]):
                options[o]=True
            else:
                options[o]= Evac.str_to_bool(options[o])
        elif (o=='OPT_Resampling_moveave'):
            if Evac.check_empty(options[o]):
                options[o]=True
            else:
                options[o]= Evac.str_to_bool(options[o])
        elif (o=='OPT_Resampling_Frequency'):
            if Evac.check_empty(options[o]):
                options[o]=4.0
                
        
        elif (o=='OPT_Springreduction'):
            if Evac.check_empty(options[o]):
                options[o]=True
            else:
                options[o]= Evac.str_to_bool(options[o])
        elif (o=='OPT_Springreduction_K'):
            if Evac.check_empty(options[o]):
                options[o]=-0.116
        elif (o=='OPT_LVDT_failure'):
            if Evac.check_empty(options[o]):
                options[o]=[False,0,' ']
                # options[o]=[True,-1,'L_PM']
            else:
                options[o]=str(options[o]).replace('"','').split(',')
            options[o][0]= Evac.str_to_bool(options[o][0])
            options[o][1]=float(options[o][1])
            options[o][2]=  str(options[o][2])
                
        elif (o=='OPT_Compression'):
            if Evac.check_empty(options[o]):
                options[o]=False
            else:
                options[o]= Evac.str_to_bool(options[o])
                
        
        elif (o=='OPT_DIC'):
            if Evac.check_empty(options[o]):
                options[o]=False
            else:
                options[o]= Evac.str_to_bool(options[o])
         

        elif (o=='OPT_Determination_Distance'):
            if Evac.check_empty(options[o]):
                # options[o]=[100,100]
                options[o]=[10,5]
            else:
                options[o]=str(options[o]).replace('"','').split(',')
            options[o][0]=int(options[o][0])
            options[o][1]=int(options[o][1])
                
        elif (o=='OPT_YM_Determination_range'):
            if Evac.check_empty(options[o]):
                # options[o]=[0.25,0.75,'U']
                options[o]=[0.25,0.50,'U']
            else:
                options[o]=str(options[o]).replace('"','').split(',')
            options[o][0]=float(options[o][0])
            options[o][1]=float(options[o][1])
        elif(o=='OPT_YM_Determination_refinement'):
            if Evac.check_empty(options[o]):
                # options[o]=[0.15,0.75,'U','opt_LG',True,8]
                # options[o]=[0.05,0.75,'U','opt_LG',True,8]
                options[o]=[0.05,0.75,'Y','opt_LG',True,8]
                # options[o]=['S1',0.75,'U','opt_LG',True,8]
                # options[o]=[0.05,0.75,'U','opt_LG',True,4]
            else:
                options[o]=str(options[o]).replace('"','').split(',')
            if isinstance(options[o][0], str):
                if options[o][0].startswith('S'):
                    # options[o][0] = int(options[o][0][1:])
                    options[o][0] = options[o][0]
                else:
                    options[o][0]=float(options[o][0])
            options[o][1]=float(options[o][1])
            options[o][4]= Evac.str_to_bool(options[o][4])
            options[o][5]=  int(options[o][5])
    return options

#%% 1.0 Evaluation
def ATT_single(prot_ser, paths, mfile_add=''):
    # out_name = prot_ser['Donor']+'_'+prot_ser['Designation']
    out_name = prot_ser['Designation']+mfile_add
    # out_name = prot_ser.name
    out_full = paths['out']+out_name
    _opts = prot_ser[prot_ser.index.str.startswith('OPT_')]
    _opts = ATT_Option_reader(_opts)
    
    path_meas = paths['meas']+_opts['OPT_File_Meas']+mfile_add+".xlsx"
    path_dic = paths['dic']+_opts['OPT_File_DIC']+mfile_add+".csv"
    
    plt_name = prot_ser['Designation']+mfile_add
    timings=pd.Series([],dtype='float64')
    timings.loc[0.0]=time.perf_counter()
    
    
    rel_time_digs = Evac.sigdig(_opts['OPT_Resampling_Frequency'], 4)
    rel_time_digs = 2
    
           
    dic_used_Strain="Strain_"+_opts['OPT_YM_Determination_refinement'][3]
    tmp_in = _opts['OPT_YM_Determination_refinement'][3].split('_')[-1]
    
    if _opts['OPT_YM_Determination_refinement'][3].split('_')[-2] == 'con':
        tmp_md = '0'
    elif _opts['OPT_YM_Determination_refinement'][3].split('_')[-2] == 'opt':
        tmp_md = '2'
    else:
        raise ValueError('OPT_YM_Determination_refinement seams to be wrong')
    loc_Yd_tmp = 'E_lsq_R_A%s%sl'%(tmp_md,tmp_in)
        
    
    if output_lvl>=1: log_mg=open(out_full+'.log','w')
    ftxt=(("  Parameters of Evaluation:"),
          ("   Evaluation start time:     %s" %datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
          ("   Path protocol:             %s" %paths['prot']),
          ("   Path measurement:          %s" %path_meas),
          ("   Path optical- measurement: %s" %path_dic),
          ("   Resampling = %s (Frequency = %d Hz, moving-average = %s)" %(_opts['OPT_Resampling'],_opts['OPT_Resampling_Frequency'],_opts['OPT_Resampling_moveave'])),
          ("   Compression = %s" %(_opts['OPT_Compression'])),
          ("   LVDT failure = %s" %(_opts['OPT_LVDT_failure'])),
          ("   LVDT-spring-force-reduction = %s (K_Fed = %f N/mm)" %(_opts['OPT_Springreduction'],_opts['OPT_Springreduction_K'])),
          ("   Youngs Modulus determination between %f and %f of point %s" %(*_opts['OPT_YM_Determination_range'],)),
          ("   Distance between points: %d / %d steps " %(*_opts['OPT_Determination_Distance'],)),
          ("   Improvment of Youngs-Modulus-determination between %s*Stress_max and point %s," %(_opts['OPT_YM_Determination_refinement'][0],_opts['OPT_YM_Determination_refinement'][2])),
          ("    with smoothing on difference-quotient (%s, %d)" %(_opts['OPT_YM_Determination_refinement'][4],_opts['OPT_YM_Determination_refinement'][5])),
          ("    with allowable deviation of %f * difference-quotient_max in determination range" %(_opts['OPT_YM_Determination_refinement'][1])),
          ("   DIC-Measurement = %s" %(_opts['OPT_DIC'])),
          ("   DIC-Strain-suffix for range refinement and plots = %s" %(_opts['OPT_YM_Determination_refinement'][3])))
          # ("   DIC-minimal points (special / specimen) = %d / %d" %(_opts['OPT_DIC_Tester'][-2],_opts['OPT_DIC_Tester'][-1])),
          # ("   DIC-names of special points (l,r,head), = %s, %s, %s" %(*_opts['OPT_DIC_Points_TBT_device'],)),
          # ("   DIC-names of meas. points for fork (l,m,r), = %s, %s, %s" %(*_opts['OPT_DIC_Points_meas_fork'],)),
          # ("   DIC-maximal SD = %.3f mm and maximal displacement between steps %.1f mm" %(_opts['OPT_DIC_Tester'][0],_opts['OPT_DIC_Tester'][1])))
    Evac.MG_strlog('\n'.join(ftxt),log_mg,output_lvl,printopt=False)
    # =============================================================================
    
    #%% 2 Geometry
    Evac.MG_strlog("\n "+"="*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### 2 Geometry ###",log_mg,output_lvl,printopt=False)
    
    if prot_ser['Test_Shape'] == 'Belt':
        if prot_ser['CS_type'] == 'Rectangle':
            thickness_mean = prot_ser[['thickness_1','thickness_2']].mean()
            width_mean     = prot_ser[['width_1','width_2']].mean()
            Area   = thickness_mean * width_mean
            Volume = Area * prot_ser['Length']
        elif prot_ser['CS_type'] == 'Dogbone':
            raise NotImplementedError("Cross section type %s not implemented"%prot_ser['CS_type'])
        else:
            raise NotImplementedError("Cross section type %s not implemented"%prot_ser['CS_type'])
    else:
        raise NotImplementedError("Test shape %s not implemented"%prot_ser['Test_Shape'])
    
    Evac.MG_strlog("\n    Area   det=IN: %s (%.3f-%.3f)"%(Area==prot_ser['Area_CS'],
                                                          Area,prot_ser['Area_CS']),
                   log_mg,output_lvl,printopt=True)    
    Evac.MG_strlog("\n    Volume det=IN: %s (%.3f-%.3f)"%(Volume/1000==prot_ser['Volume'],
                                                          Volume/1000,prot_ser['Volume']),
                   log_mg,output_lvl,printopt=True)
    # reset
    Length = prot_ser['Length_test']
    Area   = prot_ser['Area_CS']
    Volume = prot_ser['Volume']
    
    # =============================================================================
    
    #%% 3 Read in measurements
    Evac.MG_strlog("\n "+"="*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### 3 Read in measurements ###",log_mg,output_lvl,printopt=False)
    timings.loc[3.0]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    # =============================================================================
    
    #%%% 3.1 Read in conventional measurement data
    timings.loc[3.1]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    mess=pd.read_excel(path_meas, header=_opts['OPT_Measurement_file']['header'],
                       names=_opts['OPT_Measurement_file']['head_names'])
    
    mess = mess-mess.iloc[0]
    
    if isinstance(_opts['OPT_Measurement_file']['used_names_dict']['Way'],type(list())):
        mess['L_IWA']=pd.Series(mess[_opts['OPT_Measurement_file']['used_names_dict']['Way']].mean(axis=1))
        _opts['OPT_Measurement_file']['used_names_dict']['Way'] ='L_IWA'
    if ('OPT_Force_measure_type' in _opts.index):
        if not Evac.check_empty(_opts['OPT_Force_measure_type']):
            _opts['OPT_Measurement_file']['used_names_dict']['Force'] ='F_'+_opts['OPT_Force_measure_type']
    if _opts['OPT_Springreduction']: 
        mess['F_IWA_red']=(mess.L_IWA)*_opts['OPT_Springreduction_K']
        
    # # =============================================================================
    # #%%% 3.2 Specify used conventional measured force and way
    timings.loc[3.2]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    messu = pd.DataFrame({'Time': mess[_opts['OPT_Measurement_file']['used_names_dict']['Time']],
                          'Force': mess[_opts['OPT_Measurement_file']['used_names_dict']['Force']],
                          'Way': mess[_opts['OPT_Measurement_file']['used_names_dict']['Way']]})

    if _opts['OPT_Springreduction']:     
        messu['Force'] = messu['Force'] - mess['F_IWA_red']
    if _opts['OPT_LVDT_failure'][0]:     
        messu['Way'] = _opts['OPT_LVDT_failure'][1]*mess[_opts['OPT_LVDT_failure'][2]]
        
    if _opts['OPT_Compression']==True:
        messu.Force=messu.Force*(-1)
    else:
        messu.Way=messu.Way*(-1)
    if np.invert(np.isnan(_opts['OPT_End'])):
        messu=messu.loc[messu.Time.round(rel_time_digs) <= _opts['OPT_End']]
    
    # =============================================================================
    #%%% 3.3 Read in optical measurement data
    timings.loc[3.3]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    if _opts['OPT_DIC']:
        dic=None
        dicu=None
        dic_dt=None
        step_range_dic=None
        raise NotImplementedError('DIC not implemented')
        
    # =============================================================================
    #%% 4 Merging measurements
    Evac.MG_strlog("\n "+"="*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### 4 Merging measurements ###",log_mg,output_lvl,printopt=False)
    timings.loc[4.0]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    # =============================================================================
    #%%% 4.1 Determine time offset between conventional and optical measurement
    timings.loc[4.1]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    if _opts['OPT_DIC']:
        if _opts['OPT_pre_load_cycles'] > 0:
            ri_Way,_,rich_Way=Evac.rise_curve(messu['Way'],True,4)
            ws = _opts['OPT_Determination_Distance'][0]
            we = rich_Way[(rich_Way == True) and (rich_Way.index >= _opts['OPT_Determination_Distance'][0])].index[0]
            mun_tmp = messu.loc[abs(messu.Way.loc[ws:we]-messu.Way.loc[ws:we].max()/8).idxmin():
                                abs(messu.Way.loc[ws:we]-messu.Way.loc[ws:we].max()/4).idxmin()]
            del ri_Way, rich_Way, ws, we
            print('Check pre load cycles implementation in time offset!')
        else:
            mun_tmp = messu.loc[abs(messu.Way.loc[:(messu.Way.idxmax())]-messu.Way.max()/8).idxmin():
                                abs(messu.Way.loc[:(messu.Way.idxmax())]-messu.Way.max()/4).idxmin()]
        linvm   = np.polyfit(mun_tmp.loc[:,'Time'],mun_tmp.loc[:,'Way'],1)
        tsm     = -linvm[1]/linvm[0]
        mun_tmp = dicu.loc[abs(dicu.Disp_opt_head.loc[:(dicu.Disp_opt_head.idxmax())]-dicu.Disp_opt_head.max()/8).idxmin():
                           abs(dicu.Disp_opt_head.loc[:(dicu.Disp_opt_head.idxmax())]-dicu.Disp_opt_head.max()/4).idxmin()]
        linvd   = np.polyfit(mun_tmp.loc[:,'Time'],mun_tmp.loc[:,'Disp_opt_head'],1)
        tsd     = -linvd[1]/linvd[0]
        toff    = tsm-tsd
        
        if True:
            maxt_tmp=max(messu.Time.loc[abs(messu.Way.loc[:(messu.Way.idxmax())]-messu.Way.max()/4).idxmin()],dicu.Time.loc[abs(dicu.Disp_opt_head.loc[:(dicu.Disp_opt_head.idxmax())]-dicu.Disp_opt_head.max()/4).idxmin()])
            xlin_tmp=np.linspace(min(tsm,tsd),maxt_tmp,11)
            
            fig, ax1 = plt.subplots()
            ax1.set_title('%s - Way-measuring time difference'%plt_name)
            ax1.set_xlabel('Time / s')
            ax1.set_ylabel('Way / mm')
            ax1.plot(messu.Time.loc[messu.Time<=maxt_tmp], messu.Way.loc[messu.Time<=maxt_tmp], 'r-', label='PM-way')
            ax1.plot(dicu.Time.loc[dicu.Time<=maxt_tmp],   dicu.Disp_opt_head.loc[dicu.Time<=maxt_tmp], 'b-', label='DIC-way')
            ax1.plot(xlin_tmp,np.polyval(linvm,xlin_tmp),'y:',label='PM-lin')
            ax1.plot(xlin_tmp,np.polyval(linvd,xlin_tmp),'g:',label='DIC-lin')
            ax1.axvline(x=tsm, color='red', linestyle=':', label='PM-way-start')
            ax1.axvline(x=tsd, color='blue', linestyle=':', label='DIC-way-start')
            ax1.grid()
            ax1.legend()
            ftxt=('$t_{S,PM}$  = % 2.4f s '%(tsm),
                  '$t_{S,DIC}$ = % 2.4f s '%(tsd))
            fig.text(0.95,0.15,'\n'.join(ftxt),
                     ha='right',va='bottom', bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(out_full+'-toff.pdf')
            plt.savefig(out_full+'-toff.png')
            plt.show(block=False)
            plt.close()
            del xlin_tmp
        
        Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
        Evac.MG_strlog("\n   Time offset between PM and DIC: %.3f s" %(toff),log_mg,output_lvl)
    else:
        toff=0.0
        
    messu.Time=(messu.Time-toff).round(rel_time_digs)
    
    # =============================================================================
    #%%% 4.2 Downsampling of conventional data
    timings.loc[4.2]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    mess_dt=messu.Time.diff().mean()
    mess_f=round((1/mess_dt),1)
    
    if _opts['OPT_Resampling']:
        if _opts['OPT_Resampling_moveave']:
            sampler=mess_f/_opts['OPT_Resampling_Frequency']
            m1=messu.transform(lambda x: np.convolve(x, np.ones(int(sampler))/sampler, mode='same'))
            m1.Time=messu.Time
            messu=m1
            
        a=messu.Time.loc[(np.mod(messu.Time,1/_opts['OPT_Resampling_Frequency'])==0)&(messu.Time>=0)].index[1]
        ti=pd.TimedeltaIndex(data=messu.Time[messu.Time>=messu.Time.loc[a]],
                             unit='s',name='dictimedelta')
        m2=messu[messu.Time>=messu.Time.loc[a]].set_index(ti,drop=True)
        m2=m2.resample('%.3fS'%(1/_opts['OPT_Resampling_Frequency']))
        messu=m2.interpolate(method='linear',limit_area='inside',limit=8)
        messu.Time=messu.Time.round(rel_time_digs)
        messu.index=pd.RangeIndex(0,messu.Time.count(),1)
        messu=messu.iloc[:-1] # letzen entfernen, da falsch gemittelt
    
    mess_dt=messu.Time.diff().mean()
    mess_f=round((1/mess_dt),1)
    
    # =============================================================================
    #%%% 4.3 Merge optical to conventional data
    timings.loc[4.3]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    if _opts['OPT_DIC']:
        dicu.Time=dicu.Time.round(rel_time_digs)
        ind=pd.RangeIndex(dicu.loc[dicu.Time>=messu.Time.min()].index[0],
                          messu.Time.count()+dicu.loc[dicu.Time>=messu.Time.min()].index[0],1)
        messu=messu.merge(dicu,how='left', on='Time').set_index(ind)
        
    else:
        # dic_f=round(1/dic_dt)
        dic_f=mess_f
        
    f_vdm=dic_f/mess_f
        
    # =============================================================================
    #%% 5 Start and End
    Evac.MG_strlog("\n "+"="*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### 5 Start and End ###",log_mg,output_lvl,printopt=False)
    timings.loc[5.0]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)                                                    
    # =============================================================================
    #%%% 5.1 Determine start and end of evaluation
    timings.loc[5.1]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    if np.isnan(_opts['OPT_End']):
        dic_to_mess_End=messu.iloc[-1].name
    else:
        # dic_to_mess_End=messu.loc[messu.Time<=prot.loc[prot_lnr]['DIC_End']*1/f_vdm].index[-1]
        dic_to_mess_End=messu.loc[messu.Time<=_opts['OPT_End']].index[-1]
        
    mun_tmp=messu.loc[:min(dic_to_mess_End,messu.Force.idxmax()),'Force']
    messu['driF'],messu['dcuF'],messu['driF_schg']=Evac.rise_curve(messu.loc[:dic_to_mess_End]['Force'],True,4)
    
    for i in messu.index: # Startpunkt über Vorzeichenwechsel im Anstieg
        if messu.loc[i,'driF_schg']:
            if not messu.loc[i+1:i+max(int(_opts['OPT_Determination_Distance'][0]/2),1),'driF_schg'].any():
                messu_iS=i
                break
    
    messu_iS,_=Evac.find_SandE(messu.loc[messu_iS:messu_iS+_opts['OPT_Determination_Distance'][0],
                                         'driF'],abs(messu['driF']).quantile(0.5),"pgm_other",0.1)

    # _,messu_iE=Evac.find_SandE(messu['driF'],0,"qua_self",0.5) # changed 211022 (B5-sr09)
    try: # search after maximum Force
        _,messu_iE=Evac.find_SandE(messu['driF'].loc[messu.Force.idxmax():],
                                    messu['driF'],"qua_other",0.5)
    except IndexError:
        messu_iE=dic_to_mess_End
        
    messu_iE=min(messu_iE,dic_to_mess_End)
    
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n   Start of evaluation after %.3f seconds, corresponds to %.5f %% of max. force."
                   %(messu.Time[messu_iS],100*abs(messu.Force[messu_iS])/abs(messu.Force).max()),log_mg,output_lvl)
    
    messu=messu.loc[messu_iS:messu_iE]
    if _opts['OPT_DIC']:
        step_range_dic = Evac.pd_combine_index(step_range_dic,messu.index)
        
    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Measuring'%plt_name)
    color1 = 'tab:red'
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('Force / N', color=color1)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=messu.Time.loc[messu_iS], color='gray', linestyle='-')
    ax1.axvline(x=messu.Time.loc[messu_iE], color='gray', linestyle='-')
    if np.invert(np.isnan(_opts['OPT_End'])):
        ax1.axvline(x=_opts['OPT_End'], color='gray', linestyle='-.')
    ax1.plot(mess.Time, mess[_opts['OPT_Measurement_file']['used_names_dict']['Force']],
             'r-', label='Force')
    # ax1.plot(mess.Time, mess.F_WZ, 'r-', label='Force-WZ')
    # ax1.plot(mess.Time, mess.F_PM, 'm-', label='Force-PM')
    if _opts['OPT_Springreduction']: 
        ax1.plot(mess.Time, -mess.F_IWA_red, 'b:', label='Force-IWA')
    ax1.grid()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Way / mm', color=color2)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.plot(mess.Time, (mess.L_PM), 'y--', label='Way-PM')
    ax2.plot(mess.Time, -(mess.L_IWA1-mess.L_IWA1[0]), 'b--', label='Way-IWA1')
    ax2.plot(mess.Time, -(mess.L_IWA2-mess.L_IWA2[0]), 'b-.', label='Way-IWA2')
    if _opts['OPT_DIC']:
        ax2.plot(dic.Time, dic.Disp_opt_head, 'k:', label='Way-DIC')
        # ax2.plot(dic.Time, dic.DDisp_PM_c, 'm:', label='Way-DIC-P')
        # ax2.plot(dic.Time, dic.DDisp_PC_c, 'g:', label='Way-DIC-C')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), ncol=1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-meas.pdf')
    plt.savefig(out_full+'-meas.png')
    plt.show()
    
    
    
    # =============================================================================
    #%%% 5.2 Resetting way
    timings.loc[5.2]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    # messu.Force=messu.Force-messu.Force.loc[messu_iS]
    messu.Way=messu.Way-messu.Way.loc[messu_iS]
    
    if _opts['OPT_DIC']:    
        messu.Disp_opt_head=messu.Disp_opt_head-messu.Disp_opt_head.loc[messu_iS]
        
    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Measuring (used)'%plt_name)
    color1 = 'tab:red'
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('Force / N', color=color1)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=messu.Time.loc[messu_iS], color='gray', linestyle='-')
    ax1.axvline(x=messu.Time.loc[messu_iE], color='gray', linestyle='-')
    ax1.plot(messu.Time, messu.Force, 'r-', label='Force')
    ax1.grid()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Way / mm', color=color2)  # we already handled the x-label with 
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.plot(messu.Time, messu.Way, 'b--', label='Way')
    if _opts['OPT_DIC']:
        ax2.plot(messu.Time, messu.Disp_opt_head, 'k:', label='Way-DIC')
        # ax2.plot(messu.Time, messu.DDisp_PM_c, 'm:', label='Way-DIC-P')
        # ax2.plot(messu.Time, messu.DDisp_PC_c, 'g:', label='Way-DIC-C')
    #fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.15), ncol=2)
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), ncol=1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-meas_u.pdf')
    plt.savefig(out_full+'-meas_u.png')
    plt.show()
    plt.close(fig)   
    
    # =============================================================================
    #%% 6 Evaluation
    Evac.MG_strlog("\n "+"="*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### 6 Evaluation ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.0]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    #%%% 6.2 Determine evaluation curves
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### -6.2 Determine evaluation curves ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.2]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    messu['Strain']=messu.Way/Length
    messu['Stress']=messu.Force/Area

    if _opts['OPT_DIC']:

    #Test Anstieg
        tmon=Evac.test_pdmon(messu,['Stress','Strain',
                                    'Strain_opt_d_A','Strain_opt_d_S','Strain_opt_d_M',
                                    'Strain_opt_c_A','Strain_opt_c_S','Strain_opt_c_M'],1,10)
    else:
        tmon=Evac.test_pdmon(messu,['Stress','Strain'],1,10)
    
    Evac.MG_strlog("\n   Last 10 monoton increasing periods:\n    %s"
                   %tmon.to_frame(name='Epoche').T.to_string().replace('\n','\n    '),log_mg,output_lvl)
    
    
    
    # =============================================================================
    #%%% 6.3 Determine points of interest
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### -6.3 Determine points of interest ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.31]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    VIP_messu=pd.Series([],dtype='int64',name='VIP_messu')
    VIP_messu['S']=messu.driF.index[0]
    VIP_messu['E']=messu.driF.index[-1]
    VIP_messu['U']=messu.Force.idxmax()
    
    
    
    
    if _opts['OPT_pre_load_cycles']>0:
        ### Zeitdifferenz zwischen Messverfahren und Zuordnung von Zyklen
        dt_drif_schg=1/_opts['OPT_Resampling_Frequency']    
        mun_tmp=messu.Time[messu.driF_schg].loc[(VIP_messu['S']+_opts['OPT_Determination_Distance'][1]):(VIP_messu['U']-_opts['OPT_Determination_Distance'][1])]
        dt=pd.DataFrame(data={'d1':mun_tmp.diff(),'d2':-mun_tmp.diff(-1)})
        dt['dmin']=dt.loc(axis=1)['d1':'d2'].min(axis=1).round(10)
        # drif_schg_counter=dt.dmin.loc[abs((dt.dmin-dt.dmin.max())/dt.dmin.max())<=dt_drif_schg]
        drif_schg_counter=dt.dmin.loc[abs((dt.dmin-dt.dmin.iloc[1:-1].max())/dt.dmin.iloc[1:-1].max())<=dt_drif_schg]
        
        i=0
        while (drif_schg_counter.count()!=_opts['OPT_pre_load_cycles']*2):
            i+=1
            if (drif_schg_counter.count()>_opts['OPT_pre_load_cycles']*2):
                dt_drif_schg=dt_drif_schg/1.1
            else:
                dt_drif_schg=dt_drif_schg*1.1
            drif_schg_counter=dt.dmin.loc[abs((dt.dmin-dt.dmin.iloc[1:-1].max())/dt.dmin.iloc[1:-1].max())<=dt_drif_schg]
            if i==20:
                Evac.MG_strlog("\n  %d cykles, without result!\n    -> Using last %d as counter." %(i,drif_schg_counter.count()),log_mg,output_lvl,printopt=False)
                break
        Evac.MG_strlog("\n  %d cykles, with result!\n    -> Using last %.3f s as delimiter." %(i,dt_drif_schg),log_mg,output_lvl,printopt=False)
        
        zyklus=1
        VIP_PMf=pd.Series([],dtype='int64') #konventionell-Kraft
        VIP_PMw=pd.Series([],dtype='int64') #konventionell-Weg
        VIP_DICw=pd.Series([],dtype='int64') #Bildkorrelation-Weg
    
        for i in drif_schg_counter.index:
            if (np.invert(i in VIP_messu)) and (_opts['OPT_pre_load_cycles']!=0):            #wenn Periode noch nicht benannt ist und keine zyk. Belastung
                if zyklus<=_opts['OPT_pre_load_cycles'] and (i>VIP_messu['S']+_opts['OPT_Determination_Distance'][1]):
                    if np.sign(messu.dcuF.loc[i]) == -1.0:
                        VIP_messu['C'+str(zyklus) + '-']=abs(messu.Force.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmax()
                        VIP_PMf['C'+str(zyklus) + '-']  =abs(messu.Force.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmax()
                        VIP_PMw['C'+str(zyklus) + '-']  =abs(messu.Way.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmax()
                        if _opts['OPT_DIC']: VIP_DICw['C'+str(zyklus) + '-']=abs(messu.DWay.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmax()               
                        zyklus += 1
                    else:
                        VIP_messu['C'+str(zyklus) + '+']=abs(messu.Force.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmin()
                        VIP_PMf['C'+str(zyklus) + '+']  =abs(messu.Force.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmin()
                        VIP_PMw['C'+str(zyklus) + '+']  =abs(messu.Way.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmin()   
                        if _opts['OPT_DIC']: VIP_DICw['C'+str(zyklus) + '+']=abs(messu.DWay.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmin()              
                elif zyklus==(_opts['OPT_pre_load_cycles']+1):
                    VIP_messu['F1']=abs(messu.Force.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmin()
                    VIP_PMf['F1']  =abs(messu.Force.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmin()
                    VIP_PMw['F1']  =abs(messu.Way.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmin() 
                    if _opts['OPT_DIC']: VIP_DICw['F1+']=abs(messu.DWay.loc[i-_opts['OPT_Determination_Distance'][1]/2:i+_opts['OPT_Determination_Distance'][1]/2+1]).idxmin()
                    zyklus += 1            
        VIP_messu['C1+']=(abs(messu.Force.loc[VIP_messu['S']:VIP_messu['C1-']]-messu.Force.loc[VIP_messu[VIP_messu.index.str.endswith('+')]].mean())).idxmin()
        VIP_messu['F2']=(abs(messu.Force.loc[VIP_messu['F1']:VIP_messu['U']]-messu.Force.loc[VIP_messu[VIP_messu.index.str.endswith('-')]].mean())).idxmin()

        
        VIP_PMdiff=VIP_PMf-VIP_PMw
        VIP_PDdiff=VIP_PMf-VIP_DICw
        VIP_mess_diffs=pd.DataFrame(data={'PM_Force':VIP_PMf,'PM_Way':VIP_PMw,'DIC_way':VIP_DICw,'Diff_PM':VIP_PMdiff,'Diff_PD':VIP_PDdiff})
        
        if output_lvl>=1:
            Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
            Evac.MG_strlog("\n  Differenz Extremwertvergleich: \n  %s" %VIP_mess_diffs.to_string(),log_mg,output_lvl,printopt=False)
            Evac.MG_strlog("\n  Differenz Kraft- zu Wegextreme-PM (MW):  %f" %VIP_PMdiff.mean(),log_mg,output_lvl,printopt=False)
            Evac.MG_strlog("\n  Differenz Kraft- zu Wegextreme-DIC (MW): %f" %VIP_PDdiff.mean(),log_mg,output_lvl,printopt=False)      

    
    else:
        VL_soll=_opts['OPT_YM_Determination_range'][0]*messu.Force.max() # VL_soll Krücke, da nan in Protokoll
        ZL_soll=_opts['OPT_YM_Determination_range'][1]*messu.Force.max() # ZL_soll Krücke, da nan in Protokoll
        VIP_messu['F1']=(abs(messu.Force.loc[VIP_messu['S']:VIP_messu[_opts['OPT_YM_Determination_range'][2]]]-VL_soll)).idxmin()
        VIP_messu['F2']=(abs(messu.Force.loc[VIP_messu['S']:VIP_messu[_opts['OPT_YM_Determination_range'][2]]]-ZL_soll)).idxmin()
    

    # B    
    mun_tmp = messu.loc[VIP_messu['F1']+_opts['OPT_Determination_Distance'][1]:VIP_messu['U']-1]
    if mun_tmp.driF_schg.any()==True: # 
        VIP_messu['Y']=mun_tmp.loc[mun_tmp.driF_schg==True].index[0]-1
    else:
        VIP_messu['Y']=VIP_messu['U']
        Evac.MG_strlog('\n    Fy set on datapoint of Fu!',log_mg,output_lvl) 
        
    # mun_tmp = messu.loc[VIP_messu['U']:VIP_messu['E']-1]
    mun_tmp = messu.loc[VIP_messu['U']-1:VIP_messu['E']-1]
    if mun_tmp.driF_schg.any():
        i=mun_tmp.loc[mun_tmp.driF_schg].index[0]
        VIP_messu['B']  =mun_tmp.driF.loc[i:i+_opts['OPT_Determination_Distance'][1]].idxmin()-2 # statt allgemeinem Minimum bei größtem Kraftabfall nahe Maximalkraft, -2 da differenz aussage über vorherigen punkt
        if VIP_messu['B']<VIP_messu['U']: VIP_messu['B']=VIP_messu['U']
    # # if (mun_tmp['driF'].min()/mun_tmp['driF'].quantile(0.25))>=2:
    # if (mun_tmp['driF'].min()/mun_tmp['driF'].quantile(0.25))>=1.0:
    #     VIP_messu['B']=mun_tmp['driF'].idxmin()-1
    else:
        Evac.MG_strlog('\n   Fb not reliably determinable!',log_mg,output_lvl)
            
    
    # ftmp=float(messu.Force.loc[VIP_messu[_opts['OPT_YM_Determination_range'][2]]]*_opts['OPT_YM_Determination_range'][0])
    # VIP_messu['F1']=abs(messu.Force.loc[:VIP_messu[_opts['OPT_YM_Determination_range'][2]]]-ftmp).idxmin()
    # ftmp=float(messu.Force.loc[VIP_messu[_opts['OPT_YM_Determination_range'][2]]]*_opts['OPT_YM_Determination_range'][1])
    # VIP_messu['F2']=abs(messu.Force.loc[:VIP_messu[_opts['OPT_YM_Determination_range'][2]]]-ftmp).idxmin()
    
        
    
    if (VIP_messu['Y']>VIP_messu['F1']) and (VIP_messu['Y']<VIP_messu['F2']): # Test ob Streckgrenze zwischen F1 und F2 liegt
        VIP_messu['F2']=VIP_messu['Y']
        # VIP_messu['F4']=VIP_messu['Y']
        # VIP_dicu['F2']=VIP_dicu['Y']
        # VIP_dicu['F4']=VIP_dicu['Y']
        Evac.MG_strlog("\n   F2 set on Y (Force-rise between F1 and old F2)",log_mg,output_lvl)
    
    
    VIP_messu=VIP_messu.sort_values()
    if _opts['OPT_DIC']:
        VIP_dicu=VIP_messu.copy(deep=True)
        VIP_dicu.name='VIP_dicu'
    
    #%%%% 6.3.2 Improvement of evaluation range
    timings.loc[6.32]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    
    
    # (siehe Keuerleber, M. (2006). Bestimmung des Elastizitätsmoduls von Kunststoffen bei hohen Dehnraten am Beispiel von PP. Von der Fakultät Maschinenbau der Universität Stuttgart zur Erlangung der Würde eines Doktor-Ingenieurs (Dr.-Ing.) genehmigte Abhandlung. Doktorarbeit. Universität Stuttgart, Stuttgart.)
    if _opts['OPT_pre_load_cycles'] == 0:
        iS='S'
    else:
        iS='F1'
        
    # ttmp = messu.Stress.loc[[VIP_messu[iS],VIP_messu[_opts['OPT_YM_Determination_refinement'][2]]]]
    # ftmp=float(ttmp.iloc[0]+(ttmp.iloc[-1]-ttmp.iloc[0])*_opts['OPT_YM_Determination_refinement'][0])
    # Lbord=abs(messu.Stress.loc[VIP_messu[iS]:VIP_messu[_opts['OPT_YM_Determination_refinement'][2]]]-ftmp).idxmin()
    
    # DQcon=pd.concat(Evac.Diff_Quot(messu.loc[:,'Strain'],
    #                               messu.loc[:,'Stress'],
    #                               _opts['OPT_YM_Determination_refinement'][4],
    #                               _opts['OPT_YM_Determination_refinement'][5]), axis=1)
    #                               # True,4), axis=1)
    # DQcon=messu.loc(axis=1)[['Strain','Stress']].join(DQcon,how='outer')
    # DQcons=DQcon.loc[Lbord:VIP_messu[_opts['OPT_YM_Determination_refinement'][2]]]
    # VIP_messu['FM']=DQcons.DQ1.idxmax()
    # try:
    #     VIP_messu['F3']=DQcons.loc[:VIP_messu['FM']].iloc[::-1].loc[(DQcons.DQ1/DQcons.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]+1
    #     # VIP_messu['F3']=DQcons.loc[:VIP_messu['FM']].iloc[::-1].loc[(DQcons.DQ1/DQcons.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]
    # except IndexError:
    #     VIP_messu['F3']=DQcons.index[0]
    # try: # Hinzugefügt am 16.09.2021
    #     VIP_messu['F4']=DQcons.loc[VIP_messu['FM']:].loc[(DQcons.DQ1/DQcons.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]-1
    #     # VIP_messu['F4']=DQcons.loc[VIP_messu['FM']:].loc[(DQcons.DQ1/DQcons.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]
    # except IndexError:
    #     VIP_messu['F4']=VIP_messu['FM']-1 #-1 könnte zu Problemen führen
    # VIP_messu=VIP_messu.sort_values()
    
    VIP_messu, DQcons, txt = Evac.YM_eva_range_refine(m_df=messu, VIP=VIP_messu,
                                                      n_strain='Strain', n_stress='Stress',
                                                      n_loBo=iS, 
                                                      n_upBo=_opts['OPT_YM_Determination_refinement'][2],
                                                      d_loBo=_opts['OPT_YM_Determination_refinement'][0],
                                                      d_max=_opts['OPT_YM_Determination_refinement'][1], 
                                                      rise_det=_opts['OPT_YM_Determination_refinement'][-2:],
                                                      n_Outlo='F3',n_Outmi='FM',n_Outhi='F4')
        
    Evac.MG_strlog(Evac.str_indent(txt),
                   log_mg,output_lvl,printopt=False)
    if True:
        # fig, (ax1,ax3) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize = (6.3,1*3.54))
        fig.suptitle('%s - Improvement of evaluation range for Youngs Modulus'%plt_name)
        ax1.set_title('Conventional measured strain')
        ax1.set_xlabel('Strain / -')
        ax1.set_ylabel('Stress / MPa')
        ax1.plot(messu.loc[VIP_messu[iS]:VIP_messu[_opts['OPT_YM_Determination_refinement'][2]],'Strain'], messu.loc[VIP_messu[iS]:VIP_messu[_opts['OPT_YM_Determination_refinement'][2]],'Stress'], 'r.',label='$\sigma$-$\epsilon$')
        a, b=messu.loc[VIP_messu[iS:_opts['OPT_YM_Determination_refinement'][2]],'Strain'], messu.loc[VIP_messu[iS:_opts['OPT_YM_Determination_refinement'][2]],'Stress']
        j=np.int64(-1)
        ax1.plot(a, b, 'bx')
        for x in VIP_messu[iS:_opts['OPT_YM_Determination_refinement'][2]].index:
            j+=1
            if j%2: c=(6,-6)
            else:   c=(-6,6)
            ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                         xytext=c, ha="center", va="center", textcoords='offset points')
        ax1.grid()
        ax2=ax1.twinx()
        ax2.set_ylabel('Normalized derivatives / -')
        # ax2.plot(DQcons['Strain'], 
        #          DQcons['DQ1']/abs(DQcons['DQ1']).max(), 'b:',label='DQ1')
        ax2.plot(DQcons['Strain'], 
                 DQcons['DQ1']/DQcons['DQ1'].max(), 'b:',label='DQ1')
        ax2.plot(DQcons['Strain'], 
                 DQcons['DQ2']/abs(DQcons['DQ2']).max(), 'g:',label='DQ2')
        ax2.plot(DQcons['Strain'], 
                 DQcons['DQ3']/abs(DQcons['DQ3']).max(), 'y:',label='DQ3')
        # ax2.axvline(x=DQcons.loc[VIP_messu['FM'],'Strain'],color='gray', linestyle='-')
        # ax2.axvline(x=DQcons.loc[VIP_messu['F3'],'Strain'],color='gray', linestyle=':')
        # ax2.axvline(x=DQcons.loc[VIP_messu['F4'],'Strain'],color='gray', linestyle=':')
        ax2.axvline(x=messu.loc[VIP_messu['FM'],'Strain'],color='gray', linestyle='-')
        ax2.axvline(x=messu.loc[VIP_messu['F3'],'Strain'],color='gray', linestyle=':')
        ax2.axvline(x=messu.loc[VIP_messu['F4'],'Strain'],color='gray', linestyle=':')
        ax2.axhline(y=_opts['OPT_YM_Determination_refinement'][1],color='gray', linestyle='--')
        ax2.set_yticks([-1,0,1])
        ax2.grid(which='major',axis='y',linestyle=':')
        fig.legend(loc='lower right', ncol=4)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(out_full+'-YMRange_Imp.pdf')
        plt.savefig(out_full+'-YMRange_Imp.png')
        plt.show()
    
    if _opts['OPT_DIC']:
        Evac.MG_strlog("\n   Datapoints (con/opt) between F1-F2: %d/%d and F3-F4: %d/%d."
                       %(VIP_messu['F2']-VIP_messu['F1'],VIP_dicu['F2']-VIP_dicu['F1'],
                         VIP_messu['F4']-VIP_messu['F3'],VIP_dicu['F4']-VIP_dicu['F3']),log_mg,output_lvl)

    else:
        Evac.MG_strlog("\n   Datapoints (con/opt) between F1-F2: %d and F3-F4: %d."
                       %(VIP_messu['F2']-VIP_messu['F1'],
                         VIP_messu['F4']-VIP_messu['F3']),log_mg,output_lvl)

    # =====================================================================================
    #%%% 6.4 Determine Youngs-Moduli
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### -6.4 Determine Youngs-Moduli ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.4]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    d_stress_mid = messu.Stress.diff()
    d_strain_mid = messu.Strain.diff()
    d_Force = messu.Force.diff()
    
    Ind_YM_f=['F1','F2']
    Ind_YM_r=['F3','F4']
    sf_eva_con = messu.loc[VIP_messu[Ind_YM_f[0]]:VIP_messu[Ind_YM_f[1]]].index
    sr_eva_con = messu.loc[VIP_messu[Ind_YM_r[0]]:VIP_messu[Ind_YM_r[1]]].index    

    # -------------------------------------------------------------------------------------
    #%%%% 6.4.1 Method A
    timings.loc[6.41]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    
    
    A0Al_ser = Evac.YM_eva_com_sel(stress_ser=d_stress_mid,
                                   strain_ser=d_strain_mid,
                                   comp=_opts['OPT_Compression'],
                                   name='A0Al', 
                                   det_opt='incremental')
    E_A_df = pd.concat([A0Al_ser],axis=1)
    # cols_con=E_A_df.columns.str.contains('0')
    # E_A_con = Evac.pd_agg(E_A_df.loc[sr_eva_con,cols_con])
    # E_A_opt = Evac.pd_agg(E_A_df.loc[sr_eva_dic,np.invert(cols_con)])
    # E_A = pd.concat([E_A_con,E_A_opt],axis=1)
    E_A = Evac.pd_agg(E_A_df.loc[sr_eva_con])

    if True:
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig.suptitle('%s - Compare method A'%(plt_name))
        ax1.set_title('All Steps')
        ax1.set_xlabel('Step / -')
        ax1.set_ylabel('E / MPa')
        cc={'A0Al':'m:'}
        for k in cc:
            ax1.plot(E_A_df.loc[:VIP_messu['U']].index,
                     E_A_df.loc[:VIP_messu['U']][k],
                     cc[k], label='%s - %.2f MPa'%(k,E_A.at['mean',k]))
        ax1.axvline(x=VIP_messu['F3'], color='brown', linestyle=':')
        ax1.axvline(x=VIP_messu['F4'], color='brown', linestyle='--')
        if _opts['OPT_DIC']:
            ax1.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
            ax1.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax1.grid()
        ax1.legend()
        ax2.set_title('Improved determination range')
        ax2.set_xlabel('Step / -')
        ax2.set_ylabel('E / MPa')
        cr={'A0Al':sr_eva_con}
        for k in cc:
            ax2.plot(cr[k], E_A_df[k].loc[cr[k]], cc[k])
        ax2.axvline(x=VIP_messu['F3'], color='brown', linestyle=':')
        ax2.axvline(x=VIP_messu['F4'], color='brown', linestyle='--')
        if _opts['OPT_DIC']:
            ax2.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
            ax2.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax2.grid()
        fig.tight_layout()
        plt.savefig(out_full+'-YM-Me_A.pdf')
        plt.savefig(out_full+'-YM-Me_A.png')
        plt.show()
        plt.close(fig)
        
    #least-square fit
    E_lsq_F_A0Al = Evac.YM_eva_com_sel(stress_ser=messu.Stress,
                                       strain_ser=messu.Strain,
                                       comp=_opts['OPT_Compression'],
                                       name='E_lsq_F_A0Al', 
                                       det_opt='leastsq',
                                       **{'ind_S':VIP_messu[Ind_YM_f[0]],
                                          'ind_E':VIP_messu[Ind_YM_f[1]]})
    E_lsq_F_A0Al = pd.Series(E_lsq_F_A0Al, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_F_A0Al')
    
    
    E_lsq_R_A0Al = Evac.YM_eva_com_sel(stress_ser=messu.Stress,
                                       strain_ser=messu.Strain,
                                       comp=_opts['OPT_Compression'],
                                       name='E_lsq_R_A0Al', 
                                       det_opt='leastsq',
                                       **{'ind_S':VIP_messu[Ind_YM_r[0]],
                                          'ind_E':VIP_messu[Ind_YM_r[1]]})
    E_lsq_R_A0Al = pd.Series(E_lsq_R_A0Al, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_R_A0Al')
    
    E_lsq_A = pd.concat([E_lsq_F_A0Al, E_lsq_R_A0Al],axis=1)
    
    for i in VIP_messu[VIP_messu.index.str.endswith('+')].index:
        j = VIP_messu.index[VIP_messu.index.get_indexer_for([i])[0]+1]
        E_lsq_C_A0Al = Evac.YM_eva_com_sel(stress_ser=messu.Stress,
                                       strain_ser=messu.Strain,
                                       comp=_opts['OPT_Compression'],
                                       name='E_lsq_%s_A0Al'%i, 
                                       det_opt='leastsq',
                                       **{'ind_S':VIP_messu[i],
                                          'ind_E':VIP_messu[j]})
        E_lsq_C_A0Al = pd.Series(E_lsq_C_A0Al, index=['E','E_abs','Rquad','Fit_result'],
                                name='E_lsq_%s_A0Al'%i)
        E_lsq_A = pd.concat([E_lsq_A, E_lsq_C_A0Al],axis=1)
        
        k = VIP_messu.index[VIP_messu.index.get_indexer_for([j])[0]+1]
        E_lsq_C_A0Al = Evac.YM_eva_com_sel(stress_ser=messu.Stress,
                                       strain_ser=messu.Strain,
                                       comp=_opts['OPT_Compression'],
                                       name='E_lsq_%s_A0Al'%j, 
                                       det_opt='leastsq',
                                       **{'ind_S':VIP_messu[j],
                                          'ind_E':VIP_messu[k]})
        E_lsq_C_A0Al = pd.Series(E_lsq_C_A0Al, index=['E','E_abs','Rquad','Fit_result'],
                                name='E_lsq_%s_A0Al'%j)
        E_lsq_A = pd.concat([E_lsq_A, E_lsq_C_A0Al],axis=1)
    
    
    del A0Al_ser
    del E_lsq_F_A0Al, E_lsq_R_A0Al
    
    
        
    ind = VIP_messu[VIP_messu.index.str.contains('\+|F1')]
    strain_plastic = (messu.Strain.loc[ind]).diff()
    strain_plastic.index = ind.index.str.replace('F1','F').str.replace('+','')
    strain_plastic=strain_plastic.shift(-1)
    
    E_asc_ser = E_lsq_A.loc['E',E_lsq_A.columns.str.contains('\+')]
    E_asc_ser = pd.concat([E_asc_ser,E_lsq_A.loc['E',E_lsq_A.columns.str.contains('F')]])
    E_des_ser = E_lsq_A.loc['E',E_lsq_A.columns.str.contains('\-')]
    E_des_ser['F']=np.nan
    df_Ee = pd.DataFrame({'E_asc':E_asc_ser.values,'E_des':E_des_ser.values,
                          'e_pl':strain_plastic}, index = strain_plastic.index)
    
    #%%%% 6.4.8 Method compare
    timings.loc[6.48]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    


    E_lsq=E_lsq_A

    E_Methods_df = E_A_df
    E_agg_funcs = ['mean',Evac.meanwoso,'median','std','max','min']
    
    E_inc_F_comp = E_Methods_df.loc[sf_eva_con].agg(E_agg_funcs)
    E_inc_F_comp.loc['stdn']=E_inc_F_comp.loc['std']/E_inc_F_comp.loc['mean'].abs()
    E_inc_F_comp.loc['stdnwoso']=E_inc_F_comp.loc['std']/E_inc_F_comp.loc['meanwoso'].abs()
    
    E_inc_R_comp = E_Methods_df.loc[sr_eva_con].agg(E_agg_funcs)
    E_inc_R_comp.loc['stdn']=E_inc_R_comp.loc['std']/E_inc_R_comp.loc['mean'].abs()
    E_inc_R_comp.loc['stdnwoso']=E_inc_R_comp.loc['std']/E_inc_R_comp.loc['meanwoso'].abs()
    
        
    Evac.MG_strlog("\n\n  Method comaparison:",log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n  - least square fit",log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent('\n'+E_lsq.loc[['E','Rquad']].T.to_string()),
                   log_mg,output_lvl,printopt=True)
    
    Evac.MG_strlog("\n\n  - incremental (F,R,S):",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent('\n'+E_inc_F_comp.T.to_string()),
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent('\n'+E_inc_R_comp.T.to_string()),
                   log_mg,output_lvl,printopt=True)

    # --------------------------------------------------------------------------
    #%%% 6.5 Determine yield point
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### -6.5 Determine yield point ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.5]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    strain_offset = 0.002
    #     mit 0.2% Dehnversatz E(F+-F-) finden
    Evac.MG_strlog("\n  Determination of yield strain-conventional:",log_mg,output_lvl)
    VIP_messu,txt = Evac.Yield_redet(m_df=messu, VIP=VIP_messu,
                                      n_strain='Strain', n_stress='Stress',
                                      # n_loBo=['F1','F3'], n_upBo=['U'], n_loBo_int=['F2','F4'], 
                                      # n_loBo=['F3'], n_upBo=['U'], n_loBo_int=['F2','F4'], 
                                       n_loBo=['F4'], n_upBo=['U'], n_loBo_int=['F4'], 
                                      YM     = E_lsq_A['E_lsq_R_A0Al']['E'],
                                      YM_abs = E_lsq_A['E_lsq_R_A0Al']['E_abs'],
                                      strain_offset=strain_offset,
                                      rise_det=[True,4], n_yield='Y')
    Evac.MG_strlog(Evac.str_indent(txt,3),
                    log_mg,output_lvl)

    
    if _opts['OPT_DIC']:    
        Evac.MG_strlog("\n  Determination of yield strain-optical:",log_mg,output_lvl)
        VIP_dicu,txt = Evac.Yield_redet(m_df=messu, VIP=VIP_dicu,
                                        n_strain=dic_used_Strain, n_stress='Stress',
                                        # n_loBo=['F1','F3'], n_upBo=['U'], n_loBo_int=['F2','F4'], 
                                        n_loBo=['F3'], n_upBo=['U'], n_loBo_int=['F2','F4'], 
                                        YM     = E_lsq_A[loc_Yd_tmp]['E'],
                                        YM_abs = E_lsq_A[loc_Yd_tmp]['E_abs'],
                                        strain_offset=strain_offset,
                                        rise_det=[True,4], n_yield='Y')
        Evac.MG_strlog(Evac.str_indent(txt,3),
                       log_mg,output_lvl)

    
    # =============================================================================
    #%%% 6.6 Final curve
    if ('B' in VIP_messu.index) and (VIP_messu['B']>=VIP_messu['U']):
        FP_end='B'
    else:
        FP_end='U'
    FP_sta=VIP_messu[['F1','F3']].idxmin()
    # messu_FP=messu.loc[VIP_messu['F1']:VIP_messu[FP_end]].copy(deep=True)
    messu_FP=messu.loc[VIP_messu[FP_sta]:VIP_messu[FP_end]].copy(deep=True)
    messu_FP.Strain=messu_FP.Strain-(-E_lsq_A['E_lsq_R_A0Al']['E_abs']/E_lsq_A['E_lsq_R_A0Al']['E'])
    # ============================================================================
    #%% 7 Outputs
    Evac.MG_strlog("\n "+"="*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### 7 Outputs ###",log_mg,output_lvl,printopt=False)
    timings.loc[7.0]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    # ============================================================================
    #%%% 7.1 Prepare outputs
    timings.loc[7.1]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    

    out_tab               = pd.Series([],name=prot_ser.name,dtype='float64')
    out_tab['Date_eva']   = datetime.now().strftime('%d.%m.%Y')
        
    
    out_tab['fy']         = messu_FP.loc[VIP_messu['Y'],'Stress']
    out_tab['ey_con']     = messu_FP.loc[VIP_messu['Y'],'Strain']
    if _opts['OPT_DIC']:
        out_tab['ey_opt'] = messu_FP.loc[VIP_dicu['Y'],dic_used_Strain]
    out_tab['Wy_con']     = np.trapz(messu_FP.loc[:VIP_messu['Y'],'Force'],
                                     x=messu_FP.loc[:VIP_messu['Y'],'Way'])
    if _opts['OPT_DIC']:
        out_tab['Wy_opt'] = np.trapz(messu_FP.loc[:VIP_dicu['Y'],'Force'],
                                     x=messu_FP.loc[:VIP_dicu['Y'],dic_used_Disp])
    out_tab['fu']         = messu_FP.loc[VIP_messu['U'],'Stress']
    out_tab['eu_con']     = messu_FP.loc[VIP_messu['U'],'Strain']
    if _opts['OPT_DIC']:
        out_tab['eu_opt'] = messu_FP.loc[VIP_dicu['U'],dic_used_Strain]
    out_tab['Wu_con']     = np.trapz(messu_FP.loc[:VIP_messu['U'],'Force'],
                                     x=messu_FP.loc[:VIP_messu['U'],'Way'])
    if _opts['OPT_DIC']:
        out_tab['Wu_opt'] = np.trapz(messu_FP.loc[:VIP_dicu['U'],'Force'],
                                     x=messu_FP.loc[:VIP_dicu['U'],dic_used_Disp])
    if 'B' in VIP_messu.index:
        out_tab['fb']         = messu_FP.loc[VIP_messu['B'],'Stress']
        out_tab['eb_con']     = messu_FP.loc[VIP_messu['B'],'Strain']
    else:
        out_tab['fb']         = np.nan
        out_tab['eb_con']     = np.nan
    if _opts['OPT_DIC']:
        if 'B' in VIP_dicu.index:
            out_tab['eb_opt']     = messu_FP.loc[VIP_dicu['B'],dic_used_Strain]
        else:
            out_tab['eb_opt']     = np.nan
    if 'B' in VIP_messu.index:
        out_tab['Wb_con']     = np.trapz(messu_FP.loc[:VIP_messu['B'],'Force'],
                                         x=messu_FP.loc[:VIP_messu['B'],'Way'])
    else:
        out_tab['Wb_con']     = np.nan
    if _opts['OPT_DIC']:
        if 'B' in VIP_dicu.index:
            out_tab['Wb_opt']     = np.trapz(messu_FP.loc[:VIP_dicu['B'],'Force'],
                                             x=messu_FP.loc[:VIP_dicu['B'],dic_used_Disp])
        else:
            out_tab['Wb_opt']     = np.nan
        
    
    out_tab['Fy']         = messu_FP.loc[VIP_messu['Y'],'Force']
    out_tab['sy_con']     = messu_FP.loc[VIP_messu['Y'],'Way']
    out_tab['Fu']         = messu_FP.loc[VIP_messu['U'],'Force']
    out_tab['su_con']     = messu_FP.loc[VIP_messu['U'],'Way']
    if 'B' in VIP_messu.index:
        out_tab['Fb']         = messu_FP.loc[VIP_messu['B'],'Force']
        out_tab['sb_con']     = messu_FP.loc[VIP_messu['B'],'Way']
    else:
        out_tab['Fb']         = np.nan
        out_tab['sb_con']     = np.nan
        
    if _opts['OPT_pre_load_cycles'] > 0:
        ind = VIP_messu[VIP_messu.index.str.endswith('+')]
        out_tab['cyc_f_lo']   = pd.Series(messu.Stress.loc[ind]).reset_index(drop=True).mean()
        ind = VIP_messu[VIP_messu.index.str.endswith('-')]
        out_tab['cyc_f_hi']   = pd.Series(messu.Stress.loc[ind]).reset_index(drop=True).mean()
    
        out_tab['cyc_comp_E_con_first']   = df_Ee.loc['F','E_asc']/df_Ee.loc['C1','E_asc']
        out_tab['cyc_comp_E_con_rest']    = df_Ee.loc['F','E_asc']/df_Ee['E_asc'].iloc[1:-1].mean()
        # out_tab['cyc_comp_epl_con_first'] = df_Ee.loc['F','e_pl']/df_Ee.loc['C2','e_pl']
        out_tab['cyc_comp_epl_con_first'] = df_Ee.loc['C10','e_pl']/df_Ee.loc['C1','e_pl']
        # out_tab['cyc_comp_epl_con_rest']  = df_Ee.loc['F','e_pl']/df_Ee['e_pl'].iloc[2:-1].mean()
        out_tab['cyc_comp_epl_con_rest']  = df_Ee.loc['C10','e_pl']/df_Ee['e_pl'].iloc[1:-2].mean()
    # ============================================================================
    #%%% 7.2 Generate plots
    timings.loc[7.2]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
        
    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Analyzing meas. force'%plt_name)
    color1 = 'tab:red'
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('Force / N', color=color1)
    ax1.plot(messu.Time, messu.Force, 'r-', label='Force')
    a, b=messu.Time[VIP_messu],messu.Force[VIP_messu]
    j=np.int64(-1)
    ax1.plot(a, b, 'bx')
    for x in VIP_messu.index:
        j+=1
        if j%2: c=(6,-6)
        else:   c=(-6,6)
        ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                     xytext=c, ha="center", va="center", textcoords='offset points')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Rise / curve /  (N/s)', color=color2)
    ax2.plot(messu.Time, messu.driF, 'b:', label='Force-rise')
    ax2.plot(messu.Time, messu.dcuF, 'g:', label='Force-curve')
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.legend(loc='lower right', bbox_to_anchor=(0.85, 0.15))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Fdricu.pdf')
    plt.savefig(out_full+'-Fdricu.png')
    plt.show()
    plt.close(fig)
    
    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Stress vs. strain curve with labels'%plt_name)
    ax1.set_xlabel('Strain / -')
    ax1.set_ylabel('Stress / MPa')
    ind_E='E'
    ax1.plot(messu.loc[:VIP_messu[ind_E]]['Strain'],
             messu.loc[:VIP_messu[ind_E]]['Stress'], 'r--',label='con')
    if _opts['OPT_DIC']:
        ax1.plot(messu.loc[:VIP_messu[ind_E]][dic_used_Strain],
                 messu.loc[:VIP_messu[ind_E]]['Stress'], 'm--',label='opt')
    a,b = Evac.stress_linfit_plt(messu['Strain'], VIP_messu[['F3','F4']],
                                 *E_lsq_A['E_lsq_R_A0Al'][['E','E_abs']])
    ax1.plot(a, b, 'g-',label='$E_{con}$')
    VIP_mwoC=VIP_messu[np.invert(VIP_messu.index.str.contains('C'))]
    a, b=messu.Strain[VIP_mwoC[:ind_E]],messu.Stress[VIP_mwoC[:ind_E]]
    j=np.int64(-1)
    ax1.plot(a, b, 'bx')
    for x in VIP_mwoC[:ind_E].index:
        j+=1
        if j%2: c=(6,-6)
        else:   c=(-6,6)
        ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                      xytext=c, ha="center", va="center", textcoords='offset points')
    if _opts['OPT_DIC']:
        a,b=messu.loc[VIP_dicu[:'SE'],dic_used_Strain],messu.Stress[VIP_dicu[:'SE']]
        j=np.int64(-1)
        ax1.plot(a, b, 'yx')
        a,b = Evac.stress_linfit_plt(messu[dic_used_Strain], VIP_dicu[['F3','F4']],
                                     *E_lsq_A[loc_Yd_tmp][['E','E_abs']])
        ax1.plot(a, b, 'b-',label='$E_{opt}$')
    ax1.grid()
    ftxt=('$f_{y}$ = % 6.3f MPa ($\epsilon_{y}$ = %4.3f %%)'%(out_tab['fy'],out_tab['ey_con']*100),
          '$f_{u}$ = % 6.3f MPa ($\epsilon_{u}$ = %4.3f %%)'%(out_tab['fu'],out_tab['eu_con']*100),
          '$E_{con}$ = % 8.3f MPa ($R²$ = %4.3f)'%(*E_lsq_A['E_lsq_R_A0Al'][['E','Rquad']],))    
    fig.text(0.95,0.15,'\n'.join(ftxt),
              ha='right',va='bottom', bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
    fig.legend(loc='upper left', bbox_to_anchor=(0.10, 0.91))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-sigeps_wl.pdf')
    plt.savefig(out_full+'-sigeps_wl.png')
    plt.show()
    plt.close(fig)

    

    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Youngs-modulus and res. strain vs. cycles'%plt_name)
    color1 = 'tab:red'
    ax1.set_xlabel('Cycle [-]')
    ax1.set_ylabel('E [MPa]', color=color1)
    lns =ax1.plot(df_Ee.index, df_Ee.E_asc, 'rx--', label='$E_{con,load}$')
    lns+=ax1.plot(df_Ee.index, df_Ee.E_des, 'g+--', label='$E_{con,unload}$')
    if _opts['OPT_DIC']:
        lns+=ax1.plot(E_Dy.index, E_Dy, 'g+--', label='E-DIC-y')
        lns+=ax1.plot(E_D1.index, E_D1, 'm+--', label='E-DIC-1')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Plastic strain [-]', color=color2)  # we already handled the x-label with ax1
    lns+=ax2.plot(df_Ee.index, df_Ee.e_pl, 'bo:', label='$\epsilon_{con,pl}$')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax1.grid(axis='x')
    labs=[l.get_label() for l in lns]
    ax2.legend(lns,labs,loc='center right',ncol=2)
    # fig.legend(bbox_to_anchor=(0.89,0.45),ncol=2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-Eepszyk.pdf')
    plt.savefig(out_full+'-Eepszyk.png')
    plt.show(block=False)
    plt.close()
    
    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Stress vs. strain curve, final part, with labels'%plt_name)
    ax1.set_xlabel('Strain [-]')
    ax1.set_ylabel('Stress [MPa]')
    # ax1.set_xlim(0, (np.round(messu_FP.Strain.max(),2)+0.01))
    # ax1.set_ylim(0, np.round(messu_FP.Stress.max()*1.15,0))
    ax1.grid()
    ax1.plot(messu_FP['Strain'], messu_FP['Stress'], 'r--', label='meas. curve')
    # ax1.plot([0,1.1*messu_FP.Stress.loc[VIP_messu['U']]/E['F1']], [0,1.1*messu_FP.Stress.loc[VIP_messu['U']]], 'b:', label='lin. E')
    a,b = Evac.stress_linfit_plt(messu_FP['Strain'], VIP_messu[['F3','F4']],
                                 E_lsq_A['E_lsq_R_A0Al']['E'],0)
    ax1.plot(a, b, 'g-',label='$E_{con}$')
    a, b=messu_FP.Strain[VIP_messu[FP_sta:FP_end]],messu_FP.Stress[VIP_messu[FP_sta:FP_end]]
    j=np.int64(-1)
    ax1.plot(a, b, 'bx')
    for x in VIP_messu[FP_sta:FP_end].index:
        j+=1
        if j%2: c=(6,-6)
        else:   c=(-6,6)
        ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data', xytext=c, ha="center", va="center", textcoords='offset points')
    ftxt=('$f_{y}$ = %3.3f MPa ($\epsilon_{y}$ = %.3f %%)'%(out_tab['fy'],out_tab['ey_con']*100),
         '$f_{u}$ = %3.3f MPa ($\epsilon_{u}$ = %.3f %%)'%(out_tab['fu'],out_tab['eu_con']*100),
         '$E_{con}$ = % 8.3f MPa ($R²$ = %4.3f)'%(*E_lsq_A['E_lsq_R_A0Al'][['E','Rquad']],))
    fig.text(0.95,0.15,'\n'.join(ftxt),
             ha='right',va='bottom', bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
    fig.legend(loc='upper left', bbox_to_anchor=(0.10, 0.91))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(out_full+'-sigeps_fin.pdf')
    plt.savefig(out_full+'-sigeps_fin.png')
    plt.show(block=False)
    plt.close()

    # =============================================================================
    #%%% 7.3 Generate outputs
    timings.loc[7.3]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    

    # Federsteifigkeit
    for i in E_lsq.loc(axis=1)[E_lsq.columns.str.contains('_F_|_R_')].columns:
        nvar=i.replace('E','D')
        out_tab[nvar]=E_lsq.loc['E',i]*prot_ser['Area_CS']/prot_ser['Length_test']
    
    t=E_lsq.loc[['E','Rquad']].T.stack()
    t.index=t.index.to_series().apply(lambda x: '{0}_{1}'.format(*x)).values
    t.name=prot_ser.name
    out_tab=pd.concat([out_tab,t])
    
    t=E_inc_F_comp.T.stack()
    t.index=t.index.to_series().apply(lambda x: '{0}_{1}'.format(*x)).values
    t.name=prot_ser.name
    t=t.add_prefix('E_inc_F_')
    out_tab=pd.concat([out_tab,t])

    t=E_inc_R_comp.T.stack()
    t.index=t.index.to_series().apply(lambda x: '{0}_{1}'.format(*x)).values
    t.name=prot_ser.name
    t=t.add_prefix('E_inc_R_')
    out_tab=pd.concat([out_tab,t])
    
    t=df_Ee['e_pl'][:-1]
    t.name=prot_ser.name
    t=t.add_prefix('epl_con_')
    out_tab=pd.concat([out_tab,t])
   
    out_tab=pd.DataFrame([out_tab])

    vnew=pd.Series(VIP_messu.drop_duplicates().index.values,
                   index=VIP_messu.drop_duplicates(),dtype='O')
    VIP_messu=VIP_messu.sort_values()
    for i in VIP_messu.index:
        if VIP_messu.duplicated().loc[i]:
            vnew.loc[VIP_messu.loc[i]]=vnew.loc[VIP_messu.loc[i]]+','+i
    messu['VIP_m']=vnew
    messu.VIP_m.fillna('', inplace=True)
    
    if _opts['OPT_DIC']:
        vnew=pd.Series(VIP_dicu.drop_duplicates().index.values,
                       index=VIP_dicu.drop_duplicates(),dtype='O')
        VIP_dicu=VIP_dicu.sort_values()
        for i in VIP_dicu.index:
            if VIP_dicu.duplicated().loc[i]:
                vnew.loc[VIP_dicu.loc[i]]=vnew.loc[VIP_dicu.loc[i]]+','+i
        messu['VIP_d']=vnew
        messu.VIP_d.fillna('', inplace=True)
    
    messu.to_csv(out_full+'.csv',sep=';')   
    out_tab.to_csv(out_full+'-Mat.csv',sep=';')
    
    HDFst=pd.HDFStore(out_full+'.h5')
    HDFst['Measurement'] = messu
    HDFst['Measurement_final_loading'] = messu_FP
    HDFst['Material_Parameters'] = out_tab
    HDFst['Timings'] = timings
        
    if _opts['OPT_DIC']:
        HDFst['VIP'] = pd.concat([VIP_messu,VIP_dicu],axis=1)
    else:
        HDFst['VIP'] = pd.concat([VIP_messu],axis=1)
    
    HDFst['DQcon'] = DQcons
    HDFst['E_lsq'] = E_lsq
    HDFst['E_inc_df'] = E_Methods_df
    HDFst['E_inc'] = pd.concat([E_inc_F_comp.add_prefix('E_inc_F_'),
                                E_inc_R_comp.add_prefix('E_inc_R_')],axis=1)
    
    HDFst.close()

    
    timings.loc[10.0]=time.perf_counter()
    if output_lvl>=1: log_mg.close()
    
    return timings

#%% 9 Main
def ATT_series(paths, no_stats_fc, var_suffix):
    prot = pd.read_excel(paths['prot'],
                         header=11, skiprows=range(12,13),
                         index_col=0)
    
    logfp = paths['out'] + paths['prot'].split('/')[-1].replace('.xlsx','.log')
    if output_lvl>=1: log_mg=open(logfp,'w')
        
    # prot.Failure_code = prot.Failure_code.astype(str)
    # prot.Failure_code = prot.Failure_code.agg(lambda x: x.split(','))
    # eva_b = prot.Failure_code.agg(lambda x: False if len(set(x).intersection(set(no_stats_fc)))>0 else True)
    prot.Failure_code  = Evac.list_cell_compiler(prot.Failure_code)
    eva_b = Evac.list_interpreter(prot.Failure_code, no_stats_fc)
    
    # Evac.MG_strlog("\n paths:\n%s"%str(paths).replace(',','\n'),
    #                         log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n paths:",log_mg,output_lvl,printopt=False)
    for path in paths.index:
        Evac.MG_strlog("\n  %s: %s"%(path,paths[path]),
                       log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n evaluation: %d / %d"%(prot.loc[eva_b].count()[0],prot.count()[0]),
                            log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n%s"%prot.loc[eva_b].Designation.values,
                            log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n not evaluated: %d / %d"%(prot.loc[eva_b==False].count()[0],prot.count()[0]),
                            log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n%s"%prot.loc[eva_b==False].Designation.values,
                            log_mg,output_lvl,printopt=False)

    for eva in prot[eva_b].index:
        for mfile_add in var_suffix:
            Evac.MG_strlog("\n %s"%prot.loc[eva].Designation+mfile_add,
                            log_mg,output_lvl,printopt=False)  
            try:
                timings = ATT_single(prot_ser = prot.loc[eva],
                                     paths = paths, mfile_add=mfile_add)
                Evac.MG_strlog("\n   Eva_time: %.5f s"%(timings.iloc[-1]-timings.iloc[0]),
                                log_mg,output_lvl,printopt=False)  
            except Exception:
                # txt = '\n   Exception:\n    at line {} - {}:{}'.format(sys.exc_info()[-1].tb_lineno,type(e).__name__, e)
                txt = '\n   Exception:'
                txt+=Evac.str_indent('\n{}'.format(traceback.format_exc()),5)
                Evac.MG_strlog(txt, log_mg,output_lvl,printopt=False)  

    if output_lvl>=1: log_mg.close()
  

def main():
       
    # option = 'single'
    # option = 'series'
    # option = 'complete'
    option = 'pack-complete'
    
    # no_stats_fc = ['1.11','1.12','1.21','1.22','1.31','2.21','3.11']
    no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
                   'B01.1','B01.2','B01.3', 'B02.3',
                   'C01.1','C01.2','C01.3', 'C02.3',
                   'D01.1','D01.2','D01.3', 'D02.3',
                   'F01.1','F01.2','F01.3', 'F02.3']
    # var_suffix = ["A","B","C","D"] #Suffix of variants of measurements (p.E. diffferent moistures)
    var_suffix = [""] #Suffix of variants of measurements (p.E. diffferent moistures)
        
    protpaths = pd.DataFrame([],dtype='string')
    combpaths = pd.DataFrame([],dtype='string')
    # protpaths.loc['ZR','path_main'] = "F:/Messung/002-190813-ZESBO_Rückenfaszien-ZZV/"
    # protpaths.loc['ZR','name_prot'] = "190813_ZESBO_ZZV_Protokoll_new.xlsx"
    # protpaths.loc['B1','path_main'] = "F:/Messung/003-190821-Becken1-ZZV/"
    # protpaths.loc['B1','name_prot'] = "190821_Becken1_ZZV_Protokoll_new.xlsx"
    # protpaths.loc['B2','path_main'] = "F:/Messung/004-200512-Becken2-AZV/"
    # protpaths.loc['B2','name_prot'] = "200512_Becken2_AZV_Protokoll_new.xlsx"
    protpaths.loc['B3','path_main'] = "F:/Messung/005-200721_Becken3-AZV/"
    protpaths.loc['B3','name_prot'] = "200721_Becken3-AZV_Protokoll_new.xlsx"
    protpaths.loc['B4','path_main'] = "F:/Messung/006-200910_Becken4-AZV/"
    protpaths.loc['B4','name_prot'] = "200910_Becken4-AZV_Protokoll_new.xlsx"
    protpaths.loc['B5','path_main'] = "F:/Messung/007-201009_Becken5-AZV/"
    protpaths.loc['B5','name_prot'] = "201009_Becken5-AZV_Protokoll_new.xlsx"
    protpaths.loc['B6','path_main'] = "F:/Messung/008-201117_Becken6-AZV/"
    protpaths.loc['B6','name_prot'] = "201117_Becken6-AZV_Protokoll_new.xlsx"
    protpaths.loc['B7','path_main'] = "F:/Messung/009-210114_Becken7-AZV/"
    protpaths.loc['B7','name_prot'] = "210114_Becken7-AZV_Protokoll_new.xlsx"

    protpaths.loc[:,'path_con']     = "Messdaten/Messkurven/"
    protpaths.loc[:,'path_dic']     = "Messdaten/DIC/"
    protpaths.loc[:,'path_eva1']    = "Auswertung/"
    protpaths.loc[:,'path_eva2']    = "Test_py/"
    
    # t=protpaths[['path_main','path_eva1','name_prot']].stack()
    # t.groupby(level=0).apply(lambda x: '{0}{1}{2}'.format(*x))
    combpaths['prot'] = protpaths['path_main']+protpaths['path_eva1']+protpaths['name_prot']
    combpaths['meas'] = protpaths['path_main']+protpaths['path_con']
    combpaths['dic']  = protpaths['path_main']+protpaths['path_dic']
    combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']
    
    # mfile_add="" # suffix für Feuchte
    
    
    if option == 'single':
        ser='B4'
        des='sr09'
        mfile_add = var_suffix[0] #Suffix of variants of measurements (p.E. diffferent moistures)
        
        prot=pd.read_excel(combpaths.loc[ser,'prot'],
                           header=11, skiprows=range(12,13),
                           index_col=0)
        _=ATT_single(prot_ser=prot[prot.Designation==des].iloc[0], 
                     paths=combpaths.loc[ser],
                     mfile_add = mfile_add)
        
    elif option == 'series':
        ser='B4'
        ATT_series(paths = combpaths.loc[ser],
                   no_stats_fc = no_stats_fc,
                   var_suffix = var_suffix)
        
    elif option == 'complete':
        for ser in combpaths.index:
            ATT_series(paths = combpaths.loc[ser],
                       no_stats_fc = no_stats_fc,
                       var_suffix = var_suffix)
    elif option == 'pack-complete':        
        out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/230919/ATT/B3-B7_ATT-Summary"
        packpaths = combpaths[['prot','out']]
        packpaths.columns=packpaths.columns.str.replace('out','hdf')
        Evac.pack_hdf(in_paths=packpaths, out_path = out_path,
                      hdf_naming = 'Designation', var_suffix = var_suffix,
                      h5_conc = 'Material_Parameters', h5_data = 'Measurement',
                      opt_pd_out = False, opt_hdf_save = True)
        
    else:
        raise NotImplementedError('%s not implemented!'%option)
        

if __name__ == "__main__":
    main()
