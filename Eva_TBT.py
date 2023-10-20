# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:44:49 2021

@author: mgebhard

ToDo:
    - Ende Auswertung neu bestimmen (ab Fmax)? 
    - Rquad disp bei Nan Nan
    - opt-DIC Funktionalität
    - DIC resampling
    - Implementierung genutze Dehnung aus Methoden (range, plot, output)
    - Dehnung für Ausgabe auf E-Modul beziehen (relevant, wenn Auflagereindrückung nicht eliminiert)
    - Automatik Compression / Anstiegswechsel (Be-Entlasten)
        - VIP-Bestimmung
        - E-Methoden (Spannungs-/Dehnungsmaximum)
    - Ende Plot variabel (wenn kein B, dann E?)
    - stdn und stdnwoso sollte in coefficient of variation umbenannt werden (siehe Eva_common)
    
Changelog:
    - 21-09-16: Anpassung 6.3.2 (try except bei F4 und Ende FM-1)
    - 21-10-25: Ende Auswertung erst nach maximaler Kraft
    - 21-12-03: Methodenumbenennung (C:E, D:G, E:F, F:C, G:D)
    - 21-12-17: Variantenanpassung Methoden (C2Al, C2Sl, C2Cl; E4A*; F4Ag**M G3Ag, G3Mg, G3Cg)
    - 21-12-17: Korrektur stdnwoso über hinzufügen stdwoso
    - 23-10-13: - Hinzufügen finale Spannungs-Dehnungskurve (6.6) mit Ermittlung
                - Hinzufügen mehrere Streckgrenzen (6.5, 0%-0.2%-0.007%), siehe DOI: 10.1007/s10439-020-02719-2 
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
from datetime import datetime, timedelta
import time
import warnings
import json

import Bending as Bend
import Eva_common as Evac

warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings('ignore',category=FutureWarning)

figsize=[16.0/2.54, 9.0/2.54]
plt.rcParams['figure.figsize'] = figsize
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size']= 8.0
plt.rcParams['lines.linewidth']= 1.0
plt.rcParams['lines.markersize']= 4.0
plt.rcParams['markers.fillstyle']= 'none'
plt.rcParams['axes.grid']= True
#pd.set_option('display.expand_frame_repr', False)

output_lvl= 1 # 0=none, 1=only text, 2=add_diagramms
plt_Fig_dict={'tight':True, 'show':True, 
              'save':True, 's_types':["pdf","png"], 
              'clear':True, 'close':True}
MG_logopt={'logfp':None,'output_lvl':output_lvl,'logopt':True,'printopt':False}

#%% 0.1 Add. modules
def TBT_Option_reader(options):
    def check_empty(x):
        t = (x == '' or (isinstance(x, float) and  np.isnan(x)))
        return t
    for o in options.index:
        if (o=='OPT_File_Meas') or (o=='OPT_File_Meas'):
            if check_empty(options[o]):
                options[o]='Test'
            else:
                options[o]=str(options[o])
        elif (o=='OPT_Measurement_file'):
            if check_empty(options[o]):
                options[o]='{"header":48,"head_names":["Time","Trigger","F_WZ","L_PM","F_PM","L_IWA1","L_IWA2"], "used_names_dict":{"Time":"Time","Force":"F_WZ","Way":["L_IWA1","L_IWA2"]}}'
            options[o]=json.loads(options[o])
            
        
        elif (o=='OPT_End'):
            if check_empty(options[o]):
                options[o]=np.nan
        
        elif (o=='OPT_Resampling'):
            if check_empty(options[o]):
                options[o]=True
        elif (o=='OPT_Resampling_moveave'):
            if check_empty(options[o]):
                options[o]=True
        elif (o=='OPT_Resampling_Frequency'):
            if check_empty(options[o]):
                options[o]=4.0
                
        
        elif (o=='OPT_Springreduction'):
            if check_empty(options[o]):
                options[o]=False
        elif (o=='OPT_Springreduction_K'):
            if check_empty(options[o]):
                options[o]=-0.116
        elif (o=='OPT_LVDT_failure'):
            if check_empty(options[o]):
                options[o]=False
        elif (o=='OPT_Compression'):
            if check_empty(options[o]):
                options[o]=True
                
        
        elif (o=='OPT_DIC'):
            if check_empty(options[o]):
                options[o]=True
        elif (o=='OPT_DIC_Points_device_prefix') or (o=='OPT_DIC_Points_meas_prefix'):
            if check_empty(options[o]):
                if o=='OPT_DIC_Points_device_prefix':
                    options[o]='S'
                else:
                    options[o]='P'                    
            else:
                options[o]=str(options[o]).replace('"','')        
        elif (o=='OPT_DIC_Points_TBT_device'):
            if check_empty(options[o]):
                options[o]=['S1','S2','S3']
            else:
                options[o]=str(options[o]).replace('"','').split(',')
        elif (o=='OPT_DIC_Points_meas_fork'):
            if check_empty(options[o]):
                options[o]=['P3','P4','P5']
            else:
                options[o]=str(options[o]).replace('"','').split(',')
        elif (o=='OPT_DIC_Fitting'):
            if check_empty(options[o]):
                # options[o]='{"Bending_Legion_Builder":"FSE_fixed","Bending_Legion_Name":"FSE fit","Bending_MCFit_opts":{"error_weights_pre":[   1,   0,1000, 100],"error_weights_bend":[   1,   0,1000, 100],"error_weights_pre_inc":[   1,   0,1000, 100],"error_weights_bend_inc":[   1,   0,1000, 100],"fit_max_nfev_pre": 500,"fit_max_nfev_bend": 500,"fit_max_nfev_pre_inc": 500,"fit_max_nfev_bend_inc": 500},"pwargs":{"param_val": {"xmin":-10,"xmax":10,"FP":"pi/(xmax-xmin)","b1":-1.0,"b2":-0.1,"b3":-0.01,"b4":-0.001,"c":0.0,"d":0.0},"param_min": {},"param_max": {"b1":1.0,"b2":1.0,"b3":1.0,"b4":1.0}}}'
                options[o]='{"Bending_Legion_Builder":"FSE_fixed","Bending_Legion_Name":"FSE fit","Bending_MCFit_opts":{"error_weights_pre":[   1,   0,1000, 100],"error_weights_bend":[   1,  10,1000, 100],"error_weights_pre_inc":[   1,   0,1000, 100],"error_weights_bend_inc":[   1,  10,1000, 100],"fit_max_nfev_pre": 500,"fit_max_nfev_bend": 500,"fit_max_nfev_pre_inc": 500,"fit_max_nfev_bend_inc": 500},"pwargs":{"param_val": {"xmin":-10,"xmax":10,"FP":"pi/(xmax-xmin)","b1":-1.0,"b2":-0.1,"b3":-0.01,"b4":-0.001,"c":0.0,"d":0.0},"param_min": {},"param_max": {"b1":1.0,"b2":1.0,"b3":1.0,"b4":1.0}}}'
            options[o]=json.loads(options[o])
        elif (o=='OPT_DIC_Tester'):
            if check_empty(options[o]):
                # options[o]=[0.005, 50.0, 3, 7]
                options[o]=[0.005, 50.0, 3, 6]
            else:
                options[o]=str(options[o]).replace('"','').split(',')
            options[o][0]=float(options[o][0])
            options[o][1]=float(options[o][1])
            options[o][2]=  int(options[o][2])
            options[o][3]=  int(options[o][3])
        elif (o=='OPT_Poisson_prediction'):
            if check_empty(options[o]):
                options[o]=0.30
            else:
                options[o]=float(options[o])
        elif (o=='OPT_Determination_Distance'):
            if check_empty(options[o]):
                options[o]=30
            else:
                options[o]=int(options[o])
                
        elif (o=='OPT_YM_Determination_range'):
            if check_empty(options[o]):
                # options[o]=[0.25,0.75,'U']
                options[o]=[0.25,0.50,'U']
            else:
                options[o]=str(options[o]).replace('"','').split(',')
            options[o][0]=float(options[o][0])
            options[o][1]=float(options[o][1])
        elif(o=='OPT_YM_Determination_refinement'):
            if check_empty(options[o]):
                # options[o]=[0.15,0.75,'U','DStrain_M']
                # options[o]=[0.15,0.75,'U','c_M']
                # options[o]=[0.10,0.75,'U','d_M']
                # options[o]=[0.15,0.75,'U','d_M',True,4]
                options[o]=[0.15,0.75,'U','d_M',True,8]
            else:
                options[o]=str(options[o]).replace('"','').split(',')
            options[o][0]=float(options[o][0])
            options[o][1]=float(options[o][1])
            options[o][4]= bool(options[o][4])
            options[o][5]=  int(options[o][5])
    return options
#%% 1.0 Evaluation
def TBT_single(prot_ser, paths, mfile_add=''):
    # out_name = prot_ser['Donor']+'_'+prot_ser['Designation']
    out_name = prot_ser['Designation']+mfile_add
    # out_name = prot_ser.name
    out_full = paths['out']+out_name
    _opts = prot_ser[prot_ser.index.str.startswith('OPT_')]
    _opts = TBT_Option_reader(_opts)
    
    # path_meas = paths['meas']+_opts['OPT_File_Meas']+".xlsx"
    # path_dic = paths['dic']+_opts['OPT_File_DIC']+".csv"
    path_meas = paths['meas']+_opts['OPT_File_Meas']+mfile_add+".xlsx"
    path_dic = paths['dic']+_opts['OPT_File_DIC']+mfile_add+".csv"
    
    plt_name = prot_ser['Designation']+mfile_add
    timings=pd.Series([],dtype='float64')
    timings.loc[0.0]=time.perf_counter()
    
    Length = prot_ser['Length_test']
    Bo_Le = -Length/2
    Bo_Ri =  Length/2
    xlin = np.linspace(-Length/2,Length/2,101)
    
    rel_time_digs = Evac.sigdig(_opts['OPT_Resampling_Frequency'], 4)
    rel_time_digs = 2
    
    bl = Bend.Bend_func_legion(name=_opts['OPT_DIC_Fitting']['Bending_Legion_Name'])
    bl.Builder(option=_opts['OPT_DIC_Fitting']['Bending_Legion_Builder'])
    
    # pwargs={'param_val': {'xmin':-Length/2,'xmax':Length/2,'FP':'pi/(xmax-xmin)',
    #                   'b1':-1.0,'b2':-0.1,'b3':-0.01,'b4':-0.001,'c':0.0,'d':0.0},
    #         'param_min': {'b1':-np.inf,'b2':-np.inf,'b3':-np.inf,'b4':-np.inf},
    #         'param_max': {'b1':1.0,'b2':1.0,'b3':1.0,'b4':1.0}}
    pwargs=_opts['OPT_DIC_Fitting']['pwargs']
    # krücke
    pwargs['param_val']['xmin']=Bo_Le
    pwargs['param_val']['xmax']=Bo_Ri
        
    dic_used_Strain="Strain_opt_"+_opts['OPT_YM_Determination_refinement'][3]
    tmp_in = _opts['OPT_YM_Determination_refinement'][3].split('_')[-1]
    dic_used_Disp="Disp_opt_"+tmp_in
    
    if _opts['OPT_YM_Determination_refinement'][3].split('_')[-2] == 'd':
        tmp_md = '2'
    elif _opts['OPT_YM_Determination_refinement'][3].split('_')[-2] == 'c':
        tmp_md = '4'
    else:
        raise ValueError('OPT_YM_Determination_refinement seams to be wrong')
    # loc_Yd_tmp = 'E_lsq_R_A%s%sl'%(tmp_md,tmp_in)
    loc_Yd_tmp = 'E_inc_R_D%s%sgwt'%(tmp_md,tmp_in)
        
    
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
          ("   Distance between points: %d steps " %_opts['OPT_Determination_Distance']),
          ("   Improvment of Youngs-Modulus-determination between %f*Stress_max and point %s," %(_opts['OPT_YM_Determination_refinement'][0],_opts['OPT_YM_Determination_refinement'][2])),
          ("    with smoothing on difference-quotient (%s, %d)" %(_opts['OPT_YM_Determination_refinement'][4],_opts['OPT_YM_Determination_refinement'][5])),
          ("    with allowable deviation of %f * difference-quotient_max in determination range" %(_opts['OPT_YM_Determination_refinement'][1])),
          ("   DIC-Measurement = %s" %(_opts['OPT_DIC'])),
          ("   DIC-Strain-suffix for range refinement and plots = %s" %(_opts['OPT_YM_Determination_refinement'][3])),
          ("   DIC-minimal points (special / specimen) = %d / %d" %(_opts['OPT_DIC_Tester'][-2],_opts['OPT_DIC_Tester'][-1])),
          ("   DIC-names of special points (l,r,head), = %s, %s, %s" %(*_opts['OPT_DIC_Points_TBT_device'],)),
          ("   DIC-names of meas. points for fork (l,m,r), = %s, %s, %s" %(*_opts['OPT_DIC_Points_meas_fork'],)),
          ("   DIC-maximal SD = %.3f mm and maximal displacement between steps %.1f mm" %(_opts['OPT_DIC_Tester'][0],_opts['OPT_DIC_Tester'][1])))
    Evac.MG_strlog('\n'.join(ftxt),log_mg,output_lvl,printopt=False)
    #%% 2 Geometry
    Evac.MG_strlog("\n "+"="*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### 2 Geometry ###",log_mg,output_lvl,printopt=False)
    timings.loc[2.0]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    geo = pd.DataFrame({'x': [Bo_Le, 0, Bo_Ri],
                        't': [prot_ser['thickness_1'], prot_ser['thickness_2'], prot_ser['thickness_3']],
                        'w': [prot_ser['width_1'], prot_ser['width_2'], prot_ser['width_3']]}) # Geometrie auf x-Koordinaten bezogen
    
    func_t = np.poly1d(np.polyfit(geo.loc[:,'x'], geo.loc[:,'t'],2), variable='x') # Polynom 2'ter Ordnung für Dicke über x-Koordinate
    func_w = np.poly1d(np.polyfit(geo.loc[:,'x'], geo.loc[:,'w'],2), variable='x') # Polynom 2'ter Ordnung für Breite über x-Koordinate
    func_A = func_t * func_w # Polynom für Querschnitt über x-Koordinate
    func_I = func_t**3*func_w/12 # Polynom für Flächenträgheitsmoment 2'ten Grades über x-Koordinate
    
    gamma_V=Bend.gamma_V_det(_opts['OPT_Poisson_prediction'], geo['t'].mean(),
                             Length, CS_type=prot_ser['CS_type'])
    
    if True:
        fig, (ax1,ax3) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, 
                                      figsize = np.multiply(figsize,[1,2]))
        
        ax1.set_title('%s - Width and Thickness'%plt_name)
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('Thickness / mm')
        lns=ax1.plot(xlin, func_t(xlin), 'b-', label = 'Thickness-fit')
        lns+=ax1.plot(geo.loc[:,'x'],geo.loc[:,'t'], 'bs', label = 'Thickness')
        ax2 = ax1.twinx() 
        ax2.grid(False)
        ax2.set_ylabel('Width / mm')
        lns+=ax2.plot(xlin, func_w(xlin), 'r-', label = 'Width-fit')
        lns+=ax2.plot(geo.loc[:,'x'],geo.loc[:,'w'], 'ro', label = 'Width')
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)
        ax3.set_title('%s - Moment of Inertia'%plt_name)
        ax3.set_xlabel('x / mm')
        ax3.set_ylabel('MoI / mm^4')
        ax3.plot(xlin, func_I(xlin), 'g-', label = 'MoI-fit')
        ax3.plot(geo.loc[:,'x'],geo.loc[:,'t']**3*geo.loc[:,'w']/12, 'gh', label = 'MoI')
        ax3.legend()
        Evac.plt_handle_suffix(fig,path=out_full+"-Geo",**plt_Fig_dict)
    
    Evac.MG_strlog("\n  Measured Geometry:"+Evac.str_indent(geo),log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n  Function of thickness:"+Evac.str_indent(func_t),log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n  Function of width:"+Evac.str_indent(func_w),log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n  Function of area:"+Evac.str_indent(func_A),log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n  Function of moment of inertia:"+Evac.str_indent(func_I),log_mg,output_lvl,printopt=False)

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
        
    if _opts['OPT_Compression']==True:
        messu.Force=messu.Force*(-1)
    if np.invert(np.isnan(_opts['OPT_End'])):
        messu=messu.loc[messu.Time.round(rel_time_digs) <= _opts['OPT_End']]
    
    # =============================================================================
    #%%% 3.3 Read in optical measurement data
    timings.loc[3.3]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    if _opts['OPT_DIC']:
        dic = pd.read_csv(path_dic,
                          sep=";",index_col=0,skiprows=[0,1],usecols=[0,1],
                          names=['DStep','Time'], na_values={"Nan"," Nan","nan"})
        dic.index=pd.Int64Index(dic.index)
        if np.invert(np.isnan(_opts['OPT_End'])):
            dic = dic.loc[dic.Time <= _opts['OPT_End']]
    
        dic_dt = dic.Time.diff().mean()
        dic_f  = round(1/dic_dt)
        
        
        dic_tmp=pd.read_csv(path_dic,
                            sep=";",index_col=[0,1],header=[0,1],
                            na_values={"Nan"," Nan","nan"})
        dic_tmp.index.names=['Step','Time']
        dic_tmp.columns.names=['Points','Vars']
        dic_tmp=dic_tmp.reset_index()
        dic_tmp=dic_tmp.drop(columns=['Step','Time'])
        dic_tmp = dic_tmp.sort_index(axis=1)
        if np.invert(np.isnan(_opts['OPT_End'])):
            dic_tmp=dic_tmp.loc[:dic.index[-1]]
    
        
        testert={}
        testert[0]=(dic_tmp.loc(axis=1)[pd.IndexSlice[:, ['Sx','Sy','Sz']]].abs()>=_opts['OPT_DIC_Tester'][0]).any(axis=1,level=0) # Test, ob eine Standrardabweichung größer als die Grenzstd. ist
        testert[1]=(dic_tmp.loc(axis=1)[pd.IndexSlice[:, ['Sx','Sy','Sz']]].abs()==0                         ).any(axis=1,level=0) # Test, ob eine Standrardabweichung gleich 0 ist
        testert[2]=(dic_tmp.loc(axis=1)[pd.IndexSlice[:, ['Sx','Sy','Sz']]].isna()                           ).any(axis=1,level=0) # Test, ob eine Standrardabweichung NaN ist
        testert[3]=(dic_tmp.loc(axis=1)[pd.IndexSlice[:, ['Sx','Sy','Sz']]].std(axis=1,level=0)<=1e-06       ).any(axis=1,level=0) # Test, ob die Standrardabweichungen eines Punktes weniger als 1e-06 von einander abweichen (faktisch gleich sind)
        testert[4]=(dic_tmp.loc(axis=1)[pd.IndexSlice[:,    ['x','y','z']]].abs()>=_opts['OPT_DIC_Tester'][1]).any(axis=1,level=0) # Test, ob die erhaltenen Koordinaten im Bereich der erwarteten liegen
        vt=((dic_tmp.diff().loc(axis=1)[pd.IndexSlice[:,    ['x','y','z']]]**2).sum(axis=1,level=0))**0.5 # Verschiebung zwischen einzelnen Epochen
        testert[5]=(vt>=vt.median()*200) # Test ob die Verschiebung zwischen einzelnen Epochen größer als das 200-fache ihres Medians sind

        # testert[0]=(Bend.Point_df_idx(df=dic_tmp, coords= ['Sx','Sy','Sz']).abs()>=_opts['OPT_DIC_Tester'][0]).any(axis=1,level=0) # Test, ob eine Standrardabweichung größer als die Grenzstd. ist
        # testert[1]=(Bend.Point_df_idx(df=dic_tmp, coords= ['Sx','Sy','Sz']).abs()==0                         ).any(axis=1,level=0) # Test, ob eine Standrardabweichung gleich 0 ist
        # testert[2]=(Bend.Point_df_idx(df=dic_tmp, coords= ['Sx','Sy','Sz']).isna()                           ).any(axis=1,level=0) # Test, ob eine Standrardabweichung NaN ist
        # testert[3]=(Bend.Point_df_idx(df=dic_tmp, coords= ['Sx','Sy','Sz']).std(axis=1,level=0)<=1e-06       ).any(axis=1,level=0) # Test, ob die Standrardabweichungen eines Punktes weniger als 1e-06 von einander abweichen (faktisch gleich sind)
        # testert[4]=(Bend.Point_df_idx(df=dic_tmp, coords= ['x','y','z']   ).abs()>=_opts['OPT_DIC_Tester'][1]).any(axis=1,level=0) # Test, ob die erhaltenen Koordinaten im Bereich der erwarteten liegen
        # vt=((Bend.Point_df_idx(df=dic_tmp, coords= ['x','y','z']   ).diff()**2).sum(axis=1,level=0))**0.5 # Verschiebung zwischen einzelnen Epochen
        # testert[5]=(vt>=vt.median()*200) # Test ob die Verschiebung zwischen einzelnen Epochen größer als das 200-fache ihres Medians sind
        tester=testert[0]|testert[1]|testert[2]|testert[3]|testert[4]|testert[5]
        dic_tmp[pd.IndexSlice[tester]] = np.nan
        
        step_range_dic=tester.index
        anp=tester.loc[0,tester.loc[0].index.str.contains(pat='P',na=False,regex=True)].count()
        ans=tester.loc[0,tester.loc[0].index.str.contains(pat='S',na=False,regex=True)].count()
        dic_t_str=pd.Series([],dtype='O')
        Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
        Evac.MG_strlog("\n  DIC-Tester:",log_mg,output_lvl,printopt=False)
        for s in tester.index:
            dic_t_str[s]=""
            if tester.loc[s].any():
                fpnp=tester.loc[s,tester.loc[s].index.str.contains(pat='P',na=False,regex=True)].loc[tester.loc[s]==True].count()
                fpns=tester.loc[s,tester.loc[s].index.str.contains(pat='S',na=False,regex=True)].loc[tester.loc[s]==True].count()
                if ((anp-fpnp) < _opts['OPT_DIC_Tester'][3])|((ans-fpns) < _opts['OPT_DIC_Tester'][2]): 
                    step_range_dic = step_range_dic.drop(s)
                if output_lvl>=1:
                    fps=[]
                    for l in tester.loc[s].index:
                        if tester.loc[s].loc[l]:
                            fps.append(l)
                    # print("  -Series:  %d" %s)    
                    log_mg.write("\n    Failed points in series %d: %d of %d meas. | %d of %d special %s" %(s,fpnp,anp,fpns,ans,fps))
                    print("    Failed points in series %d: %d of %d meas. | %d of %d special %s" %(s,fpnp,anp,fpns,ans,fps))
                    dic_t_str[s]=dic_t_str[s]+("\n    Failed points: %d of %d meas. | %d of %d special %s" %(fpnp,anp,fpns,ans,fps))
                    if ((anp-fpnp) < _opts['OPT_DIC_Tester'][3])|((ans-fpns) < _opts['OPT_DIC_Tester'][2]): dic_t_str[s]=dic_t_str[s]+"\n    -> DIC-curving of step dropped!!!"
    
        Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
        Evac.MG_strlog("\n  Coordinate transformation and curvature calculation:",log_mg,output_lvl,printopt=False)
        
        Points_T=pd.Series([],dtype='O')
        Points_L_T=pd.Series([],dtype='O')
        for step in step_range_dic:
            if output_lvl>=1:
                log_mg.write("\n  Step %d coordinate transformation:" %step)
                log_mg.write(dic_t_str[step])
            dic_c_tmp = pd.DataFrame([dic_tmp.loc[step].loc[:,'x'],
                                      dic_tmp.loc[step].loc[:,'y'],
                                      dic_tmp.loc[step].loc[:,'z']],
                                     index=['x','y','z'])
            dic_Sdev_tmp = pd.DataFrame([dic_tmp.loc[step].loc[:,'Sx'],
                                         dic_tmp.loc[step].loc[:,'Sy'],
                                         dic_tmp.loc[step].loc[:,'Sz']],
                                        index=['Sx','Sy','Sz'])
    
            # Krücke, muss besser gehen:
            fps=[]
            for l in tester.loc[step].index:
                if tester.loc[step].loc[l]:
                    fps.append(l)
            dic_c_tmp = dic_c_tmp.drop(fps,axis=1)
            dic_Sdev_tmp  = dic_Sdev_tmp.drop(fps,axis=1)
            
            Pmeas      = dic_c_tmp.iloc[:,dic_c_tmp.columns.str.contains(pat=_opts['OPT_DIC_Points_meas_prefix'],na=False,regex=True)]
            Pspec      = dic_c_tmp.iloc[:,dic_c_tmp.columns.str.contains(pat=_opts['OPT_DIC_Points_device_prefix'],na=False,regex=True)]
            Pmeas_Sdev = dic_Sdev_tmp.iloc[:,dic_Sdev_tmp.columns.str.contains(pat=_opts['OPT_DIC_Points_meas_prefix'],na=False,regex=True)]
            Pspec_Sdev = dic_Sdev_tmp.iloc[:,dic_Sdev_tmp.columns.str.contains(pat=_opts['OPT_DIC_Points_device_prefix'],na=False,regex=True)]
            
            # Start MG-3D point transformation
            Points_T[step] , Points_L_T[step] = Bend.Point_df_transform(Pmeas = Pmeas,
                                                                        Pspec = Pspec,
                                                                        Pmeas_Sdev = Pmeas_Sdev,
                                                                        Pspec_Sdev = Pspec_Sdev,
                                                                        dic_P_name_org1 = _opts['OPT_DIC_Points_TBT_device'][0],
                                                                        dic_P_name_org2 = _opts['OPT_DIC_Points_TBT_device'][1],
                                                                        output_lvl = output_lvl,
                                                                        log_mg = log_mg)
    
        Points_L_T_stacked=dic_tmp.copy(deep=True)
        Points_L_T_stacked.loc[:]=np.nan
        Points_T_stacked=dic_tmp.copy(deep=True)
        Points_T_stacked.loc[:]=np.nan
        
        for i in step_range_dic:
            Points_L_T_stacked.loc[i]=Points_L_T[i].T.stack()
            Points_T_stacked.loc[i]=Points_T[i].T.stack()
            # Pcoord_val_m[i]=np.polyval(Pcoord[i],0)
            # Pcoord_val_lr[i]=np.polyval(Pcoord[i],[-Pcoord_val_L/2,Pcoord_val_L/2]).mean()
        
        pt=Bend.Point_df_idx(Points_L_T_stacked, coords=['x','y'], deepcopy=True)
        pt_S=Bend.Point_df_idx(pt, points='S', deepcopy=True, option='Regex')
        pt_P=Bend.Point_df_idx(pt, points='P', deepcopy=True, option='Regex')
        geo_fit = Bend.Perform_Fit(BFL=bl, Fit_func_key='w_A', P_df=pt_P,
                                    lB=Bo_Le, rB=Bo_Ri, s_range=[0],
                                    Shear_func_key='w_S', gamma_V=gamma_V, 
                                    err_weights=[1,0,0,0],
                                    max_nfev=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['fit_max_nfev_pre'],
                                    option='Pre', pb_b=False,**pwargs).loc[0]
        
        geo_d2_max = bl['w_A']['d2'](xlin,geo_fit.loc['Fit_params_dict']).max()
        geo_d2_min = bl['w_A']['d2'](xlin,geo_fit.loc['Fit_params_dict']).min()
        geo_d2_mid = bl['w_A']['d2'](0.0,geo_fit.loc['Fit_params_dict'])
        
        fig, ax1 = plt.subplots()
        ax1.set_title('%s - Geometry'%(plt_name))
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('y / mm')
        ax1.plot(xlin, bl['w_A']['d0'](xlin,geo_fit.loc['Fit_params_dict']), 'r:', label='Org.')
        ax1.plot(pt_P.loc[0].loc[:,'x'],pt_P.loc[0].loc[:,'y'], 'ko', label='org. P')
        ax1.plot(pt_S.loc[0].loc[:,'x'],pt_S.loc[0].loc[:,'y'], 'k+', label='org. S')
        for x in pt.columns.droplevel(1):
            ax1.annotate('%s' % x, xy=(pt.loc[0].loc[x,'x'],pt.loc[0].loc[x,'y']),
                         xycoords='data', xytext=(7, -7), textcoords='offset points')
        ax1.legend()
        Evac.plt_handle_suffix(fig,path=out_full+"-DIC_fit_Geo",**plt_Fig_dict)
        
        P_xcoord_ydisp = Points_L_T_stacked.loc(axis=1)[:,['x','y']].copy()
        P_xcoord_ydisp.loc(axis=1)[:,'y'] = P_xcoord_ydisp.loc(axis=1)[:,'y']-P_xcoord_ydisp.loc(axis=1)[:,'y'].loc[0]
        P_xcoord_ydisp_meas = P_xcoord_ydisp.loc(axis=1)[P_xcoord_ydisp.columns.droplevel(level=1).str.contains(pat='P',na=False,regex=True)]
        P_xcoord_ydisp_spec = P_xcoord_ydisp.loc(axis=1)[P_xcoord_ydisp.columns.droplevel(level=1).str.contains(pat='S',na=False,regex=True)]

        dic['Disp_opt_head'] = P_xcoord_ydisp_spec.loc(axis=1)[(_opts['OPT_DIC_Points_TBT_device'][2],'y')]
        if _opts['OPT_Compression']: dic['Disp_opt_head']=dic['Disp_opt_head']*(-1)
    
    
    dicu=dic.copy(deep=True)
    # dicu=pd.DataFrame({'Time': dic.Time, 'DWay': dic.DWay, 'DStrain': dic.DL1GS, 'DStrainq': dic.DL2GS})
    del dic_tmp,anp,ans
    del dic_c_tmp,dic_Sdev_tmp
    del Pmeas,Pspec,Pmeas_Sdev,Pspec_Sdev
    del Points_T,Points_L_T
    del pt,pt_S,pt_P
    
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
            ax1.legend()
            ftxt=('$t_{S,PM}$  = % 2.4f s '%(tsm),
                  '$t_{S,DIC}$ = % 2.4f s '%(tsd))
            fig.text(0.95,0.15,'\n'.join(ftxt),
                     ha='right',va='bottom', bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
            Evac.plt_handle_suffix(fig,path=out_full+"-toff",**plt_Fig_dict)
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
        dic_f=round(1/dic_dt)
        
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
    messu['driF'],messu['dcuF'],messu['driF_schg']=Evac.rise_curve(messu.loc[:dic_to_mess_End]['Force'],True,2)
    
    for i in messu.index: # Startpunkt über Vorzeichenwechsel im Anstieg
        if messu.loc[i,'driF_schg']:
            if not messu.loc[i+1:i+max(int(_opts['OPT_Determination_Distance']/2),1),'driF_schg'].any():
                messu_iS=i
                break
    
    messu_iS,_=Evac.find_SandE(messu.loc[messu_iS:+messu_iS+_opts['OPT_Determination_Distance'],
                                         'driF'],abs(messu['driF']).quantile(0.5),"pgm_other",0.1)
    # _,messu_iE=Evac.find_SandE(messu['driF'],0,"qua_self",0.5) # not changed 251022 (B5-sr09)
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
        P_xcoord_ydisp_meas=P_xcoord_ydisp_meas.loc[messu_iS:messu_iE]
        P_xcoord_ydisp_spec=P_xcoord_ydisp_spec.loc[messu_iS:messu_iE]
        step_range_dic = Evac.pd_combine_index(step_range_dic,messu.index)
        # step_range_dic = Evac.pd_combine_index(step_range_dic[1:],messu.index)
        
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
    ax1.plot(mess.Time, -mess.F_WZ, 'r-', label='Force-WZ')
    ax1.plot(mess.Time, -mess.F_PM, 'm-', label='Force-PM')
    if _opts['OPT_Springreduction']: 
        ax1.plot(mess.Time, -mess.F_IWA_red, 'b:', label='Force-IWA')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.grid(False)
    color2 = 'tab:blue'
    ax2.set_ylabel('Way / mm', color=color2)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.plot(mess.Time, -(mess.L_PM), 'y--', label='Way-PM')
    ax2.plot(mess.Time, mess.L_IWA1-mess.L_IWA1[0], 'b--', label='Way-IWA1')
    ax2.plot(mess.Time, mess.L_IWA2-mess.L_IWA2[0], 'b-.', label='Way-IWA2')
    if _opts['OPT_DIC']:
        ax2.plot(dic.Time, dic.Disp_opt_head, 'k:', label='Way-DIC')
        # ax2.plot(dic.Time, dic.DDisp_PM_c, 'm:', label='Way-DIC-P')
        # ax2.plot(dic.Time, dic.DDisp_PC_c, 'g:', label='Way-DIC-C')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), ncol=1)
    Evac.plt_handle_suffix(fig,path=out_full+"-meas",**plt_Fig_dict)
    
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
        P_xcoord_ydisp_meas.loc(axis=1)[:,'y'] = P_xcoord_ydisp_meas.loc(axis=1)[:,'y']-P_xcoord_ydisp_meas.loc(axis=1)[:,'y'].loc[messu_iS]
        P_xcoord_ydisp_spec.loc(axis=1)[:,'y'] = P_xcoord_ydisp_spec.loc(axis=1)[:,'y']-P_xcoord_ydisp_spec.loc(axis=1)[:,'y'].loc[messu_iS]

    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Measuring (used)'%plt_name)
    color1 = 'tab:red'
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('Force / N', color=color1)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axvline(x=messu.Time.loc[messu_iS], color='gray', linestyle='-')
    ax1.axvline(x=messu.Time.loc[messu_iE], color='gray', linestyle='-')
    ax1.plot(messu.Time, messu.Force, 'r-', label='Force')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.grid(False)
    color2 = 'tab:blue'
    ax2.set_ylabel('Way / mm', color=color2)  # we already handled the x-label with 
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.plot(messu.Time, messu.Way, 'b--', label='Way')
    if _opts['OPT_DIC']:
        ax2.plot(messu.Time, messu.Disp_opt_head, 'k:', label='Way-DIC')
        # ax2.plot(messu.Time, messu.DDisp_PM_c, 'm:', label='Way-DIC-P')
        # ax2.plot(messu.Time, messu.DDisp_PC_c, 'g:', label='Way-DIC-C')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), ncol=1)
    Evac.plt_handle_suffix(fig,path=out_full+"-meas_u",**plt_Fig_dict)
    
    # =============================================================================
    #%% 6 Evaluation
    Evac.MG_strlog("\n "+"="*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### 6 Evaluation ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.0]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    # =============================================================================
    #%%% 6.0 Perform DIC-fittings
    #%%%% 6.11 Pre-Fit with all components (Indentaion, shear force deformation and bending deformation)
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog('\n## Pre-fit on displacements:',log_mg,output_lvl,printopt=False)
    timings.loc[6.11]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    Pre_fit_df = Bend.Perform_Fit(BFL=bl, Fit_func_key='w_A', 
                                  P_df=P_xcoord_ydisp_meas,
                                  lB=Bo_Le, rB=Bo_Ri, 
                                  # s_range=step_range_dic, #Verschiebung an 0 is 0
                                  s_range=step_range_dic[1:],
                                  Shear_func_key='w_S', gamma_V=gamma_V, 
                                  err_weights=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['error_weights_pre'], 
                                  max_nfev=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['fit_max_nfev_pre'],
                                  nan_policy='omit',
                                  option='Pre', pb_b=True,**pwargs)
    for step in Pre_fit_df.index:
        ftxt=lmfit.fit_report(Pre_fit_df['Fit_Result'].loc[step])
        ftxt=ftxt.replace("\n[[Variables]]","\n    R-square-multi     = %1.5f\n[[Variables]]"%Pre_fit_df['Rquad_multi'].loc[step])
        ftxt=ftxt.replace("\n[[Variables]]","\n    R-square-disp      = %1.5f\n[[Variables]]"%Pre_fit_df['Rquad_disp'].loc[step])
        Evac.MG_strlog('\n  Step %d'%step+Evac.str_indent(ftxt),
                       log_mg,output_lvl,printopt=False)

    
    # =============================================================================
    #%%%% 6.12 Refit to adjusted bending deformation
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog('\n## Bend-fit on displacements:',log_mg,output_lvl,printopt=False)
    timings.loc[6.12]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    P_xcoord_ydisp_meas_M=P_xcoord_ydisp_meas.loc[step_range_dic].copy(deep=True)
    P_xcoord_ydisp_meas_S=P_xcoord_ydisp_meas.loc[step_range_dic].copy(deep=True)
    iy=bl['w_I']['d0'](Bend.Point_df_idx(P_xcoord_ydisp_meas_M,coords='x'),
                       Pre_fit_df.loc[:,'Fit_params_dict'])
    vy=bl['w_V']['d0'](Bend.Point_df_idx(P_xcoord_ydisp_meas_M,coords='x'),
                       Pre_fit_df.loc[:,'Fit_params_dict'])
    P_xcoord_ydisp_meas_S.loc(axis=1)[:,'y']=P_xcoord_ydisp_meas_S.loc(axis=1)[:,'y']-iy
    P_xcoord_ydisp_meas_M.loc(axis=1)[:,'y']=P_xcoord_ydisp_meas_M.loc(axis=1)[:,'y']-iy-vy
    
    
    Bend_fit_df = Bend.Perform_Fit(BFL=bl, Fit_func_key='w_M',
                                   P_df=P_xcoord_ydisp_meas_M,
                                   lB=Bo_Le, rB=Bo_Ri, 
                                   s_range=step_range_dic[1:], 
                                   err_weights=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['error_weights_bend'], 
                                   max_nfev=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['fit_max_nfev_bend'],
                                   nan_policy='omit',
                                   option='Bend', pb_b=True,**pwargs)
    for step in Bend_fit_df.index:
        ftxt=lmfit.fit_report(Bend_fit_df['Fit_Result'].loc[step])
        ftxt=ftxt.replace("\n[[Variables]]","\n    R-square-multi     = %1.5f\n[[Variables]]"%Bend_fit_df['Rquad_multi'].loc[step])
        ftxt=ftxt.replace("\n[[Variables]]","\n    R-square-disp      = %1.5f\n[[Variables]]"%Bend_fit_df['Rquad_disp'].loc[step])
        Evac.MG_strlog('\n  Step %d'%step+Evac.str_indent(ftxt),
                       log_mg,output_lvl,printopt=False)
    
    # =============================================================================
    #%%%% Incremental:
    # P_xcoord_ydiff = Bend.Points_diff(Bend.Point_df_idx(Points_L_T_stacked, coords=['x','y'], deepcopy=True))
    P_xcoord_ydiff = Bend.Points_diff(P_xcoord_ydisp.loc[step_range_dic])

    P_xcoord_ydiff_meas = Bend.Point_df_idx(P_xcoord_ydiff, steps=step_range_dic, 
                                            points='P', deepcopy=False, option='Regex')
    P_xcoord_ydiff_spec = Bend.Point_df_idx(P_xcoord_ydiff, steps=step_range_dic, 
                                            points='S', deepcopy=False, option='Regex')
    
    # step_range_dic_inc = step_range_dic.drop(P_xcoord_ydiff.loc[step_range_dic].loc[P_xcoord_ydisp.loc[step_range_dic].isna().any(axis=1)].index)
    # step_range_dic_inc = (Bend.Point_df_idx(P_xcoord_ydiff_meas, steps=step_range_dic, coords='y').isna().sum(axis=1) < (_opts['OPT_DIC_Tester'][3])).index
    step_range_dic_inc = step_range_dic[np.where(np.invert(Bend.Point_df_idx(P_xcoord_ydiff_meas, steps=step_range_dic, coords='y').isna()).sum(axis=1) >= (_opts['OPT_DIC_Tester'][3]))]
    # =============================================================================
    #%%%% 6.13 Pre-Fit with all components (Indentaion, shear force deformation and bending deformation)
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog('\n## Pre-fit on increments:',log_mg,output_lvl,printopt=False)
    timings.loc[6.13]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    Pre_inc_fit_df = Bend.Perform_Fit(BFL=bl, Fit_func_key='w_A', 
                                      P_df=P_xcoord_ydiff_meas,
                                      lB=Bo_Le, rB=Bo_Ri, 
                                      s_range=step_range_dic_inc[1:],
                                      Shear_func_key='w_S', gamma_V=gamma_V, 
                                      err_weights=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['error_weights_pre_inc'], 
                                      max_nfev=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['fit_max_nfev_pre_inc'],
                                      nan_policy='omit',
                                      option='Pre', pb_b=True,**pwargs)
    for step in Pre_inc_fit_df.index:
        ftxt=lmfit.fit_report(Pre_inc_fit_df['Fit_Result'].loc[step])
        ftxt=ftxt.replace("\n[[Variables]]","\n    R-square-multi     = %1.5f\n[[Variables]]"%Pre_inc_fit_df['Rquad_multi'].loc[step])
        ftxt=ftxt.replace("\n[[Variables]]","\n    R-square-disp      = %1.5f\n[[Variables]]"%Pre_inc_fit_df['Rquad_disp'].loc[step])
        Evac.MG_strlog('\n  Step %d'%step+Evac.str_indent(ftxt),
                       log_mg,output_lvl,printopt=False)
    
    
    # =============================================================================
    #%%%% 6.14 Refit to adjusted bending deformation
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog('\n## Bend-fit on increments:',log_mg,output_lvl,printopt=False)
    timings.loc[6.14]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    P_xcoord_ydiff_meas_M=P_xcoord_ydiff_meas.loc[step_range_dic].copy(deep=True)
    P_xcoord_ydiff_meas_S=P_xcoord_ydiff_meas.loc[step_range_dic].copy(deep=True)
    iy=bl['w_I']['d0'](Bend.Point_df_idx(P_xcoord_ydiff_meas_M,coords='x'),
                       Pre_inc_fit_df.loc[:,'Fit_params_dict'])
    vy=bl['w_V']['d0'](Bend.Point_df_idx(P_xcoord_ydiff_meas_M,coords='x'),
                       Pre_inc_fit_df.loc[:,'Fit_params_dict'])
    P_xcoord_ydiff_meas_S.loc(axis=1)[:,'y']=P_xcoord_ydiff_meas_S.loc(axis=1)[:,'y']-iy
    P_xcoord_ydiff_meas_M.loc(axis=1)[:,'y']=P_xcoord_ydiff_meas_M.loc(axis=1)[:,'y']-iy-vy
    
    
    Bend_inc_fit_df = Bend.Perform_Fit(BFL=bl, Fit_func_key='w_M',
                                       P_df=P_xcoord_ydiff_meas_M,
                                       lB=Bo_Le, rB=Bo_Ri, 
                                       s_range=step_range_dic_inc[1:], 
                                       err_weights=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['error_weights_bend_inc'], 
                                       max_nfev=_opts['OPT_DIC_Fitting']['Bending_MCFit_opts']['fit_max_nfev_bend_inc'],
                                       nan_policy='omit',
                                       option='Bend', pb_b=True,**pwargs)
    for step in Bend_inc_fit_df.index:
        ftxt=lmfit.fit_report(Bend_inc_fit_df['Fit_Result'].loc[step])
        ftxt=ftxt.replace("\n[[Variables]]","\n    R-square-multi     = %1.5f\n[[Variables]]"%Bend_inc_fit_df['Rquad_multi'].loc[step])
        ftxt=ftxt.replace("\n[[Variables]]","\n    R-square-disp      = %1.5f\n[[Variables]]"%Bend_inc_fit_df['Rquad_disp'].loc[step])
        Evac.MG_strlog('\n  Step %d'%step+Evac.str_indent(ftxt),
                       log_mg,output_lvl,printopt=False)
    


    if True:
        # Fit comparison ----------------------------------------------------------
        # step=messu.Force.idxmax()
        step = Evac.pd_combine_index(messu.loc[:messu.Force.idxmax()],step_range_dic_inc)[-1]
        
        fig, ax1 = plt.subplots()
        ax1.set_title('%s - Fit-compare - Displacement for step %i'%(plt_name,step))
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('y displacement / mm')
        ax1.plot(xlin, bl['w_A']['d0'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'r:', label='Org.')
        ax1.plot(xlin, bl['w_I']['d0'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'g:', label='Ind.')
        ax1.plot(xlin, bl['w_S']['d0'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'b:', label='Wo. ind.')
        ax1.plot(xlin, bl['w_V']['d0'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'm--', label='V')
        ax1.plot(xlin, bl['w_M_ima']['d0'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'g--', label='Imag. M')
        ax1.plot(xlin, bl['w_M']['d0'](xlin,Bend_fit_df.loc[step,'Fit_params_dict']), 'r--', label='M')
        ax1.plot(P_xcoord_ydisp_meas.loc[step].loc[:,'x'],P_xcoord_ydisp_meas.loc[step].loc[:,'y'], 'ko', label='org. P')
        ax1.plot(P_xcoord_ydisp_meas_M.loc[step].loc[:,'x'],P_xcoord_ydisp_meas_M.loc[step].loc[:,'y'], 'go', label='M P')
        ax1.legend()
        Evac.plt_handle_suffix(fig,path=out_full+"-DIC_fit-bl_U-d0",**plt_Fig_dict)
        
        fig, ax1 = plt.subplots()
        ax1.set_title('%s - Fit-compare - Slope for step %i'%(plt_name,step))
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('slope / (mm/mm)')
        ax1.plot(xlin, bl['w_A']['d1'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'r:', label='Org.')
        ax1.plot(xlin, bl['w_S']['d1'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'b:', label='Wo. ind.')
        ax1.plot(xlin, bl['w_V']['d1'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'm--', label='V')
        ax1.plot(xlin, bl['w_M']['d1'](xlin,Bend_fit_df.loc[step,'Fit_params_dict']), 'r--', label='M')
        ax1.legend()
        Evac.plt_handle_suffix(fig,path=out_full+"-DIC_fit-bl_U-d1",**plt_Fig_dict)
        
        fig, ax1 = plt.subplots()
        ax1.set_title('%s - Fit-compare - Curvature for step %i'%(plt_name,step))
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('curvature / (1/mm)')
        ax1.plot(xlin, bl['w_A']['d2'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'r:', label='Org.')
        ax1.plot(xlin, bl['w_S']['d2'](xlin,Pre_fit_df.loc[step,'Fit_params_dict']), 'b:', label='Wo. ind.')
        ax1.plot(xlin, bl['w_M']['d2'](xlin,Bend_fit_df.loc[step,'Fit_params_dict']), 'r--', label='M')
        ax1.legend()
        Evac.plt_handle_suffix(fig,path=out_full+"-DIC_fit-bl_U-d2",**plt_Fig_dict)
        
        fig, ax1 = plt.subplots()
        ax1.set_title('%s - Fit-compare-inc - Displacement for step %i'%(plt_name,step))
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('y displacement / mm')
        ax1.plot(xlin, bl['w_A']['d0'](xlin,Pre_inc_fit_df.loc[step,'Fit_params_dict']), 'r:', label='Org.')
        ax1.plot(xlin, bl['w_I']['d0'](xlin,Pre_inc_fit_df.loc[step,'Fit_params_dict']), 'g:', label='Ind.')
        ax1.plot(xlin, bl['w_S']['d0'](xlin,Pre_inc_fit_df.loc[step,'Fit_params_dict']), 'b:', label='Wo. ind.')
        ax1.plot(xlin, bl['w_V']['d0'](xlin,Pre_inc_fit_df.loc[step,'Fit_params_dict']), 'm--', label='V')
        ax1.plot(xlin, bl['w_M_ima']['d0'](xlin,Pre_inc_fit_df.loc[step,'Fit_params_dict']), 'g--', label='Imag. M')
        ax1.plot(xlin, bl['w_M']['d0'](xlin,Bend_inc_fit_df.loc[step,'Fit_params_dict']), 'r--', label='M')
        ax1.plot(P_xcoord_ydiff_meas.loc[step].loc[:,'x'],P_xcoord_ydiff_meas.loc[step].loc[:,'y'], 'ko', label='org. P')
        ax1.plot(P_xcoord_ydiff_meas_M.loc[step].loc[:,'x'],P_xcoord_ydiff_meas_M.loc[step].loc[:,'y'], 'go', label='M P')
        ax1.legend()
        Evac.plt_handle_suffix(fig,path=out_full+"-INC_fit-bl_U-d0",**plt_Fig_dict)
        
        # Pre-Fit -----------------------------------------------------------------
        Bend.Plotter.colplt_funcs_all(x=xlin, func_cohort=bl['w_A'],
                                         params=Pre_fit_df.loc(axis=1)['Fit_params_dict'],
                                         step_range=step_range_dic[1:], 
                                         title=('%s - Fit-full'%plt_name),
                                         xlabel='x  / mm',
                                         Point_df=P_xcoord_ydisp_meas,
                                         savefig=True,
                                         savefigname=out_full+'-DIC_fit-A')
                                         # cblabel='VIP', cbtick=cbtick)
        # Bending-Fit -------------------------------------------------------------
        Bend.Plotter.colplt_funcs_all(x=xlin, func_cohort=bl['w_M'],
                                         params=Bend_fit_df.loc(axis=1)['Fit_params_dict'],
                                         step_range=step_range_dic[1:], 
                                         title=('%s - Fit-Bending'%plt_name),
                                         xlabel='x / mm',
                                         Point_df=P_xcoord_ydisp_meas_M,
                                         savefig=True,
                                         savefigname=out_full+'-DIC_fit-M')
        # Pre-Fit -----------------------------------------------------------------
        Bend.Plotter.colplt_funcs_all(x=xlin, func_cohort=bl['w_A'],
                                         params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                         step_range=step_range_dic_inc[1:], 
                                         title=('%s - Incremental Fit-full'%plt_name),
                                         xlabel='x / mm',
                                         Point_df=P_xcoord_ydiff_meas,
                                         savefig=True,
                                         savefigname=out_full+'-INC_fit-A')
        # Bending-Fit -------------------------------------------------------------
        Bend.Plotter.colplt_funcs_all(x=xlin, func_cohort=bl['w_M'],
                                         params=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                         step_range=step_range_dic_inc[1:], 
                                         title=('%s - Incremental Fit-Bending'%plt_name),
                                         xlabel='x / mm',
                                         Point_df=P_xcoord_ydiff_meas_M,
                                         savefig=True,
                                         savefigname=out_full+'-INC_fit-M')

    
    
    #%%% 6.2 Determine evaluation curves
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### -6.2 Determine evaluation curves ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.2]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    # messu['Moment']=messu.Force*Length/4
    messu['Moment']=Bend.Moment_perF_func(0.0,Bo_Le,Bo_Ri)*messu.Force
    # messu['Stress']=messu.Moment*6/(prot_ser['width_2']*prot_ser['thickness_2']**2)
    messu['Stress']=Bend.stress_perF(0.0,func_I,func_t,Bo_Le,Bo_Ri)*messu.Force
    
    messu['Strain']=6*prot_ser['thickness_2']*messu.Way/Length**2

    if _opts['OPT_DIC']:
        messu['Disp_opt_A']=bl['w_A']['d0'](0.0,Pre_fit_df['Fit_params_dict'])
        if _opts['OPT_Compression']: messu['Disp_opt_A']=messu['Disp_opt_A']*(-1)
        messu['Strain_opt_d_A']=6*prot_ser['thickness_2']*messu['Disp_opt_A']/Length**2
        messu['Disp_opt_S']=bl['w_S']['d0'](0.0,Pre_fit_df['Fit_params_dict'])
        if _opts['OPT_Compression']: messu['Disp_opt_S']=messu['Disp_opt_S']*(-1)
        messu['Strain_opt_d_S']=6*prot_ser['thickness_2']*messu['Disp_opt_S']/Length**2
        messu['Disp_opt_M']=bl['w_M']['d0'](0.0,Bend_fit_df['Fit_params_dict'])
        if _opts['OPT_Compression']: messu['Disp_opt_M']=messu['Disp_opt_M']*(-1)
        messu['Strain_opt_d_M']=6*prot_ser['thickness_2']*messu['Disp_opt_M']/Length**2
    
        
        messu['Strain_opt_c_A']=Bend.straindf_from_curve(0.0,bl['w_A']['d2'],
                                                    Pre_fit_df['Fit_params_dict'], func_t)
        messu['Strain_opt_c_S']=Bend.straindf_from_curve(0.0,bl['w_S']['d2'],
                                                    Pre_fit_df['Fit_params_dict'], func_t)
        messu['Strain_opt_c_M']=Bend.straindf_from_curve(0.0,bl['w_M']['d2'],
                                                    Bend_fit_df['Fit_params_dict'], func_t)
        # Set first value of optical displacement and strain to 0.0 instead of NaN
        messu.loc[messu_iS,[    'Disp_opt_A',    'Disp_opt_S',    'Disp_opt_M',]]=0.0
        messu.loc[messu_iS,['Strain_opt_d_A','Strain_opt_d_S','Strain_opt_d_M',]]=0.0
        messu.loc[messu_iS,['Strain_opt_c_A','Strain_opt_c_S','Strain_opt_c_M',]]=0.0
        
        # messu['Strain_opt_c_P']=12*messu['DDisp_PM']*prot.loc[prot_lnr]['t2']*prot.loc[prot_lnr]['lE']/(prot.loc[prot_lnr]['lE']**2*(3*messu['DDisp_PM_L']-prot.loc[prot_lnr]['lE'])+(prot.loc[prot_lnr]['lE']-messu['DDisp_PM_L'])**3)
        # # Ermittlung über drei imaginäre Punkte auf gefitteter Mittellinie (Gabelaufnehmer):
        # messu['Strain_opt_c_C']=12*messu['DDisp_PC']*prot.loc[prot_lnr]['t2']*prot.loc[prot_lnr]['lE']/(prot.loc[prot_lnr]['lE']**2*(3*Pcoord_val_L-prot.loc[prot_lnr]['lE'])+(prot.loc[prot_lnr]['lE']-Pcoord_val_L)**3)

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
    
    mun_tmp = messu.loc[VIP_messu['S']+_opts['OPT_Determination_Distance']:VIP_messu['U']-1]
    if mun_tmp.driF_schg.any()==True: # 
        VIP_messu['Y']=mun_tmp.loc[mun_tmp.driF_schg==True].index[0]-1
    else:
        VIP_messu['Y']=VIP_messu['U']
        Evac.MG_strlog('\n    Fy set on datapoint of Fu!',log_mg,output_lvl)
        
    # mun_tmp = messu.loc[VIP_messu['U']:VIP_messu['E']-1]
    mun_tmp = messu.loc[VIP_messu['U']-1:VIP_messu['E']-1]
    if mun_tmp.driF_schg.any():
        i=mun_tmp.loc[mun_tmp.driF_schg].index[0]
        VIP_messu['B']  =mun_tmp.driF.loc[i:i+_opts['OPT_Determination_Distance']].idxmin()-2 # statt allgemeinem Minimum bei größtem Kraftabfall nahe Maximalkraft, -2 da differenz aussage über vorherigen punkt
        if VIP_messu['B']<VIP_messu['U']: VIP_messu['B']=VIP_messu['U']
    # # if (mun_tmp['driF'].min()/mun_tmp['driF'].quantile(0.25))>=2:
    # if (mun_tmp['driF'].min()/mun_tmp['driF'].quantile(0.25))>=1.0:
    #     VIP_messu['B']=mun_tmp['driF'].idxmin()-1
    else:
        Evac.MG_strlog('\n   Fb not reliably determinable!',log_mg,output_lvl)
            
    
    ftmp=float(messu.Force.loc[VIP_messu[_opts['OPT_YM_Determination_range'][2]]]*_opts['OPT_YM_Determination_range'][0])
    VIP_messu['F1']=abs(messu.Force.loc[:VIP_messu[_opts['OPT_YM_Determination_range'][2]]]-ftmp).idxmin()
    ftmp=float(messu.Force.loc[VIP_messu[_opts['OPT_YM_Determination_range'][2]]]*_opts['OPT_YM_Determination_range'][1])
    VIP_messu['F2']=abs(messu.Force.loc[:VIP_messu[_opts['OPT_YM_Determination_range'][2]]]-ftmp).idxmin()
    
        
    
    if (VIP_messu['Y']>VIP_messu['F1']) and (VIP_messu['Y']<VIP_messu['F2']): # Test ob Streckgrenze zwischen F1 und F2 liegt
        VIP_messu['F2']=VIP_messu['Y']
        # VIP_messu['F4']=VIP_messu['Y']
        # VIP_dicu['F2']=VIP_dicu['Y']
        # VIP_dicu['F4']=VIP_dicu['Y']
        Evac.MG_strlog("\n   F2 set on Y (Force-rise between F1 and old F2)",log_mg,output_lvl)
    
    ## Krücke
    # VIP_messu['F3']=VIP_messu['F1']
    # VIP_messu['F4']=VIP_messu['F2']
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
    ftmp=float(messu.Stress.loc[VIP_messu[_opts['OPT_YM_Determination_range'][2]]]*_opts['OPT_YM_Determination_refinement'][0])
    Lbord=abs(messu.Stress.loc[:VIP_messu[_opts['OPT_YM_Determination_range'][2]]]-ftmp).idxmin()
    
    DQcon=pd.concat(Evac.Diff_Quot(messu.loc[:,'Strain'],
                              messu.loc[:,'Stress'],
                              _opts['OPT_YM_Determination_refinement'][4],
                              _opts['OPT_YM_Determination_refinement'][5]), axis=1)
                              # True,4), axis=1)
    DQcon=messu.loc(axis=1)[['Strain','Stress']].join(DQcon,how='outer')
    DQcons=DQcon.loc[Lbord:VIP_messu[_opts['OPT_YM_Determination_refinement'][2]]]
    VIP_messu['FM']=DQcons.DQ1.idxmax()
    try:
        VIP_messu['F3']=DQcons.loc[:VIP_messu['FM']].iloc[::-1].loc[(DQcons.DQ1/DQcons.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]+1
        # VIP_messu['F3']=DQcons.loc[:VIP_messu['FM']].iloc[::-1].loc[(DQcons.DQ1/DQcons.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]
    except IndexError:
        VIP_messu['F3']=DQcons.index[0]
    try: # Hinzugefügt am 16.09.2021
        VIP_messu['F4']=DQcons.loc[VIP_messu['FM']:].loc[(DQcons.DQ1/DQcons.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]-1
        # VIP_messu['F4']=DQcons.loc[VIP_messu['FM']:].loc[(DQcons.DQ1/DQcons.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]
    except IndexError:
        VIP_messu['F4']=VIP_messu['FM']-1 #-1 könnte zu Problemen führen
    VIP_messu=VIP_messu.sort_values()
    
    
    # optisch:    
    DQopt=pd.concat(Evac.Diff_Quot(messu.loc[:,dic_used_Strain],
                                   messu.loc[:,'Stress'],
                                   _opts['OPT_YM_Determination_refinement'][4],
                                   _opts['OPT_YM_Determination_refinement'][5]), axis=1)
                                   # True,4), axis=1)
    DQopt=messu.loc(axis=1)[[dic_used_Strain,'Stress']].join(DQopt,how='outer')
    DQopts=DQopt.loc[Lbord:VIP_dicu[_opts['OPT_YM_Determination_refinement'][2]]]
    VIP_dicu['FM']=DQopts.DQ1.idxmax()
    try:
        VIP_dicu['F3']=DQopts.loc[:VIP_dicu['FM']].iloc[::-1].loc[(DQopts.DQ1/DQopts.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]+1
        # VIP_dicu['F3']=DQopts.loc[:VIP_dicu['FM']].iloc[::-1].loc[(DQopts.DQ1/DQopts.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]
    except IndexError:
        VIP_dicu['F3']=DQopts.index[0]
    try: # Hinzugefügt am 16.09.2021
        VIP_dicu['F4']=DQopts.loc[VIP_dicu['FM']:].loc[(DQopts.DQ1/DQopts.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]-1
        # VIP_dicu['F4']=DQopts.loc[VIP_dicu['FM']:].loc[(DQopts.DQ1/DQopts.DQ1.max())<_opts['OPT_YM_Determination_refinement'][1]].index[0]
    except IndexError:
        VIP_dicu['F4']=VIP_dicu['FM']-1 #-1 könnte zu Problemen führen
    VIP_dicu=VIP_dicu.sort_values()
    
    if True:
        fig, (ax1,ax3) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, 
                                      figsize = np.multiply(figsize,[1,2]))
        fig.suptitle('%s - Improvement of evaluation range for Youngs Modulus'%plt_name)
        ax1.set_title('Conventional measured strain')
        ax1.set_xlabel('Strain / -')
        ax1.set_ylabel('Stress / MPa')
        ax1.plot(messu.loc[:VIP_messu[_opts['OPT_YM_Determination_refinement'][2]],'Strain'], messu.loc[:VIP_messu[_opts['OPT_YM_Determination_refinement'][2]],'Stress'], 'r.',label='$\sigma$-$\epsilon$')
        a, b=messu.loc[VIP_messu[:_opts['OPT_YM_Determination_refinement'][2]],'Strain'],messu.loc[VIP_messu[:_opts['OPT_YM_Determination_refinement'][2]],'Stress']
        j=np.int64(-1)
        ax1.plot(a, b, 'bx')
        for x in VIP_messu[:_opts['OPT_YM_Determination_refinement'][2]].index:
            j+=1
            if j%2: c=(6,-6)
            else:   c=(-6,6)
            ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                         xytext=c, ha="center", va="center", textcoords='offset points')
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
        ax2.axvline(x=messu.loc[VIP_messu['FM'],'Strain'],color='gray', linestyle='-')
        ax2.axvline(x=messu.loc[VIP_messu['F3'],'Strain'],color='gray', linestyle=':')
        ax2.axvline(x=messu.loc[VIP_messu['F4'],'Strain'],color='gray', linestyle=':')
        ax2.axhline(y=_opts['OPT_YM_Determination_refinement'][1],color='gray', linestyle='--')
        ax2.set_yticks([-1,0,1])
        ax2.grid(which='major',axis='y',linestyle=':')
        fig.legend(loc='lower right', ncol=4)
        
        ax3.set_title('Optical measured strain')
        ax3.set_xlabel('Strain / -')
        ax3.set_ylabel('Stress / MPa')
        ax3.plot(messu.loc[:VIP_dicu[_opts['OPT_YM_Determination_refinement'][2]],
                           dic_used_Strain],
                 messu.loc[:VIP_dicu[_opts['OPT_YM_Determination_refinement'][2]],
                           'Stress'], 'r.',label='$\sigma$-$\epsilon$')
        a, b=messu.loc[VIP_dicu[:_opts['OPT_YM_Determination_refinement'][2]],
                       dic_used_Strain], messu.loc[VIP_dicu[:_opts['OPT_YM_Determination_refinement'][2]],'Stress']
        j=np.int64(-1)
        ax3.plot(a, b, 'bx')
        for x in VIP_dicu[:_opts['OPT_YM_Determination_refinement'][2]].index:
            j+=1
            if j%2: c=(6,-6)
            else:   c=(-6,6)
            ax3.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                         xytext=c, ha="center", va="center", textcoords='offset points')
        ax4=ax3.twinx()
        ax4.set_ylabel('Normalized derivatives / -')
        # ax4.plot(DQopts[dic_used_Strain],
        #          DQopts['DQ1']/abs(DQopts['DQ1']).max(), 'b:',label='DQ1')
        ax4.plot(DQopts[dic_used_Strain],
                 DQopts['DQ1']/DQopts['DQ1'].max(), 'b:',label='DQ1')
        ax4.plot(DQopts[dic_used_Strain],
                 DQopts['DQ2']/abs(DQopts['DQ2']).max(), 'g:',label='DQ2')
        ax4.plot(DQopts[dic_used_Strain],
                 DQopts['DQ3']/abs(DQopts['DQ3']).max(), 'y:',label='DQ3')
        # ax4.axvline(x=DQopts.loc[VIP_dicu['FM'],dic_used_Strain],color='gray', linestyle='-')
        # ax4.axvline(x=DQopts.loc[VIP_dicu['F3'],dic_used_Strain],color='gray', linestyle=':')
        # ax4.axvline(x=DQopts.loc[VIP_dicu['F4'],dic_used_Strain],color='gray', linestyle=':')
        ax4.axvline(x=messu.loc[VIP_dicu['FM'],dic_used_Strain],
                    color='gray', linestyle='-')
        ax4.axvline(x=messu.loc[VIP_dicu['F3'],dic_used_Strain],
                    color='gray', linestyle=':')
        ax4.axvline(x=messu.loc[VIP_dicu['F4'],dic_used_Strain],
                    color='gray', linestyle=':')
        ax4.axhline(y=_opts['OPT_YM_Determination_refinement'][1],
                    color='gray', linestyle='--')
        ax4.set_yticks([-1,0,1])
        ax4.grid(which='major',axis='y',linestyle=':')
        Evac.plt_handle_suffix(fig,path=out_full+"-YMRange_Imp",**plt_Fig_dict)
    
    
    Evac.MG_strlog("\n   Datapoints (con/opt) between F1-F2: %d/%d and F3-F4: %d/%d."
                   %(VIP_messu['F2']-VIP_messu['F1'],VIP_dicu['F2']-VIP_dicu['F1'],
                     VIP_messu['F4']-VIP_messu['F3'],VIP_dicu['F4']-VIP_dicu['F3']),log_mg,output_lvl)
    
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
    d_Strain_opt_d_A_mid = messu.Strain_opt_d_A.diff()
    d_Strain_opt_d_S_mid = messu.Strain_opt_d_S.diff()
    d_Strain_opt_d_M_mid = messu.Strain_opt_d_M.diff()
    d_Strain_opt_c_A_mid = messu.Strain_opt_c_A.diff()
    d_Strain_opt_c_S_mid = messu.Strain_opt_c_S.diff()
    d_Strain_opt_c_M_mid = messu.Strain_opt_c_M.diff()
    d_Force = messu.Force.diff()
    
    # sr_eva_con = messu.loc[VIP_messu['F3']:VIP_messu['F4']].index
    # sr_eva_dic = Evac.pd_combine_index(step_range_dic, 
    #                                    messu.loc[VIP_dicu['F3']:VIP_dicu['F4']])
    Ind_YM_f=['F1','F2']
    Ind_YM_r=['F3','F4']
    sf_eva_con = messu.loc[VIP_messu[Ind_YM_f[0]]:VIP_messu[Ind_YM_f[1]]].index
    sf_eva_dic = Evac.pd_combine_index(step_range_dic, 
                                       messu.loc[VIP_dicu[Ind_YM_f[0]]:VIP_dicu[Ind_YM_f[1]]])
    sr_eva_con = messu.loc[VIP_messu[Ind_YM_r[0]]:VIP_messu[Ind_YM_r[1]]].index
    sr_eva_dic = Evac.pd_combine_index(step_range_dic, 
                                       messu.loc[VIP_dicu[Ind_YM_r[0]]:VIP_dicu[Ind_YM_r[1]]])
    # -------------------------------------------------------------------------------------
    #%%%% 6.4.1 Method A
    timings.loc[6.41]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    
    
    
    A0Al_ser = Bend.YM_eva_method_A(stress_mid_ser=d_stress_mid,
                                    strain_mid_ser=d_strain_mid,
                                    comp=_opts['OPT_Compression'],
                                    name='A0Al', 
                                    det_opt='incremental')
    A2Al_ser = Bend.YM_eva_method_A(stress_mid_ser=d_stress_mid,
                                    strain_mid_ser=d_Strain_opt_d_A_mid,
                                    comp=_opts['OPT_Compression'],
                                    name='A2Al', 
                                    det_opt='incremental')
    A2Sl_ser = Bend.YM_eva_method_A(stress_mid_ser=d_stress_mid,
                                    strain_mid_ser=d_Strain_opt_d_S_mid,
                                    comp=_opts['OPT_Compression'],
                                    name='A2Sl', 
                                    det_opt='incremental')
    A2Ml_ser = Bend.YM_eva_method_A(stress_mid_ser=d_stress_mid,
                                    strain_mid_ser=d_Strain_opt_d_M_mid,
                                    comp=_opts['OPT_Compression'],
                                    name='A2Ml', 
                                    det_opt='incremental')
    A4Al_ser = Bend.YM_eva_method_A(stress_mid_ser=d_stress_mid,
                                    strain_mid_ser=d_Strain_opt_c_A_mid,
                                    comp=_opts['OPT_Compression'],
                                    name='A4Al', 
                                    det_opt='incremental')
    A4Sl_ser = Bend.YM_eva_method_A(stress_mid_ser=d_stress_mid,
                                    strain_mid_ser=d_Strain_opt_c_S_mid,
                                    comp=_opts['OPT_Compression'],
                                    name='A4Sl', 
                                    det_opt='incremental')
    A4Ml_ser = Bend.YM_eva_method_A(stress_mid_ser=d_stress_mid,
                                    strain_mid_ser=d_Strain_opt_c_M_mid,
                                    comp=_opts['OPT_Compression'],
                                    name='A4Ml', 
                                    det_opt='incremental')
    E_A_df = pd.concat([A0Al_ser,
                        A2Al_ser, A2Sl_ser, A2Ml_ser,
                        A4Al_ser, A4Sl_ser, A4Ml_ser],axis=1)
    # E_A = Evac.pd_agg(E_A_df.loc[sr_eva_con]) # geändert 21-09-20
    cols_con=E_A_df.columns.str.contains('0')
    E_A_con = Evac.pd_agg(E_A_df.loc[sr_eva_con,cols_con])
    E_A_opt = Evac.pd_agg(E_A_df.loc[sr_eva_dic,np.invert(cols_con)])
    E_A = pd.concat([E_A_con,E_A_opt],axis=1)

    if True:
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig.suptitle('%s - Compare method A'%(plt_name))
        ax1.set_title('All Steps')
        ax1.set_xlabel('Step / -')
        ax1.set_ylabel('E / MPa')
        cc={'A0Al':'m:',
            'A2Al':'r:','A2Sl':'b:','A2Ml':'g:',
            'A4Al':'r--','A4Sl':'b--','A4Ml':'g--'}
        for k in cc:
            ax1.plot(E_A_df.loc[:VIP_messu['U']].index,
                     E_A_df.loc[:VIP_messu['U']][k],
                     cc[k], label='%s - %.2f MPa'%(k,E_A.at['mean',k]))
        ax1.axvline(x=VIP_messu['F3'], color='brown', linestyle=':')
        ax1.axvline(x=VIP_messu['F4'], color='brown', linestyle='--')
        ax1.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
        ax1.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax1.legend()
        ax2.set_title('Improved determination range')
        ax2.set_xlabel('Step / -')
        ax2.set_ylabel('E / MPa')
        cr={'A0Al':sr_eva_con,
            'A2Al':sr_eva_dic,'A2Sl':sr_eva_dic,'A2Ml':sr_eva_dic,
            'A4Al':sr_eva_dic,'A4Sl':sr_eva_dic,'A4Ml':sr_eva_dic}
        for k in cc:
            ax2.plot(cr[k], E_A_df[k].loc[cr[k]], cc[k])
        ax2.axvline(x=VIP_messu['F3'], color='brown', linestyle=':')
        ax2.axvline(x=VIP_messu['F4'], color='brown', linestyle='--')
        ax2.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
        ax2.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        Evac.plt_handle_suffix(fig,path=out_full+"-YM-Me_A",**plt_Fig_dict)
        
    #least-square fit
    E_lsq_F_A0Al = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_F_A0Al', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_messu[Ind_YM_f[0]],
                                       'ind_E':VIP_messu[Ind_YM_f[1]]})
    E_lsq_F_A0Al = pd.Series(E_lsq_F_A0Al, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_F_A0Al')
    
    E_lsq_F_A2Al = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_d_A,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_F_A2Al', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                       'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_A2Al = pd.Series(E_lsq_F_A2Al, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_F_A2Al')
    E_lsq_F_A2Sl = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_d_S,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_F_A2Sl', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                       'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_A2Sl = pd.Series(E_lsq_F_A2Sl, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_F_A2Sl')
    E_lsq_F_A2Ml = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_d_M,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_F_A2Ml', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                       'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_A2Ml = pd.Series(E_lsq_F_A2Ml, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_F_A2Ml')
    
    E_lsq_F_A4Al = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_c_A,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_F_A4Al', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                       'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_A4Al = pd.Series(E_lsq_F_A4Al, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_F_A4Al')
    E_lsq_F_A4Sl = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_c_S,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_F_A4Sl', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                       'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_A4Sl = pd.Series(E_lsq_F_A4Sl, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_F_A4Sl')
    E_lsq_F_A4Ml = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_c_M,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_F_A4Ml', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                       'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_A4Ml = pd.Series(E_lsq_F_A4Ml, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_F_A4Ml')
    
    
    E_lsq_R_A0Al = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_R_A0Al', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_messu[Ind_YM_r[0]],
                                       'ind_E':VIP_messu[Ind_YM_r[1]]})
    E_lsq_R_A0Al = pd.Series(E_lsq_R_A0Al, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_R_A0Al')
    
    E_lsq_R_A2Al = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_d_A,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_R_A2Al', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                       'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_A2Al = pd.Series(E_lsq_R_A2Al, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_R_A2Al')
    E_lsq_R_A2Sl = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_d_S,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_R_A2Sl', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                       'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_A2Sl = pd.Series(E_lsq_R_A2Sl, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_R_A2Sl')
    E_lsq_R_A2Ml = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_d_M,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_R_A2Ml', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                       'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_A2Ml = pd.Series(E_lsq_R_A2Ml, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_R_A2Ml')
    
    E_lsq_R_A4Al = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_c_A,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_R_A4Al', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                       'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_A4Al = pd.Series(E_lsq_R_A4Al, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_R_A4Al')
    E_lsq_R_A4Sl = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_c_S,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_R_A4Sl', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                       'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_A4Sl = pd.Series(E_lsq_R_A4Sl, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_R_A4Sl')
    E_lsq_R_A4Ml = Bend.YM_eva_method_A(stress_mid_ser=messu.Stress,
                                    strain_mid_ser=messu.Strain_opt_c_M,
                                    comp=_opts['OPT_Compression'],
                                    name='E_lsq_R_A4Ml', 
                                    det_opt='leastsq',
                                    **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                       'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_A4Ml = pd.Series(E_lsq_R_A4Ml, index=['E','E_abs','Rquad','Fit_result'],
                            name='E_lsq_R_A4Ml')

    E_lsq_A = pd.concat([E_lsq_F_A0Al,
                         E_lsq_F_A2Al, E_lsq_F_A2Sl, E_lsq_F_A2Ml,
                         E_lsq_F_A4Al, E_lsq_F_A4Sl, E_lsq_F_A4Ml,
                         E_lsq_R_A0Al,
                         E_lsq_R_A2Al, E_lsq_R_A2Sl, E_lsq_R_A2Ml,
                         E_lsq_R_A4Al, E_lsq_R_A4Sl, E_lsq_R_A4Ml],axis=1)

    del A0Al_ser, A2Al_ser, A2Sl_ser, A2Ml_ser, A4Al_ser, A4Sl_ser, A4Ml_ser
    del E_lsq_F_A0Al, E_lsq_F_A2Al, E_lsq_F_A2Sl, E_lsq_F_A2Ml, E_lsq_F_A4Al, E_lsq_F_A4Sl, E_lsq_F_A4Ml
    del E_lsq_R_A0Al, E_lsq_R_A2Al, E_lsq_R_A2Sl, E_lsq_R_A2Ml, E_lsq_R_A4Al, E_lsq_R_A4Sl, E_lsq_R_A4Ml
    
    #%%%% 6.4.2 Method B
    timings.loc[6.42]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    B1Al_ser, B1Al_strain_ser = Bend.YM_eva_method_B(option="Points", 
                                                     stress_mid_ser=d_stress_mid,
                                                     thickness = func_t(0.0), 
                                                     Length = Length,
                                                     P_df = P_xcoord_ydiff_meas, 
                                                     P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                                     comp=_opts['OPT_Compression'],
                                                     name='B1Al')
    B1Sl_ser, B1Sl_strain_ser = Bend.YM_eva_method_B(option="Points",
                                                     stress_mid_ser=d_stress_mid,
                                                     thickness = func_t(0.0), 
                                                     Length = Length,
                                                     P_df = P_xcoord_ydiff_meas_S, 
                                                     P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                                     comp=_opts['OPT_Compression'],
                                                     name='B1Sl')
    B1Ml_ser, B1Ml_strain_ser = Bend.YM_eva_method_B(option="Points",
                                                     stress_mid_ser=d_stress_mid,
                                                     thickness = func_t(0.0), 
                                                     Length = Length,
                                                     P_df = P_xcoord_ydiff_meas_M, 
                                                     P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                                     comp=_opts['OPT_Compression'],
                                                     name='B1Ml')
    
    
    B2Al_ser, B2Al_strain_ser = Bend.YM_eva_method_B(option="Fit",
                                                     stress_mid_ser=d_stress_mid,
                                                     thickness = func_t(0.0),
                                                     Length=Length,
                                                     w_func=bl['w_A']['d0'],
                                                     w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                                     Length_det=10.0,
                                                     comp=_opts['OPT_Compression'],
                                                     name='B2Al')
    B2Sl_ser, B2Sl_strain_ser = Bend.YM_eva_method_B(option="Fit", 
                                                     stress_mid_ser=d_stress_mid,
                                                     thickness = func_t(0.0),
                                                     Length=Length,
                                                     w_func=bl['w_S']['d0'],
                                                     w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                                     Length_det=10.0, 
                                                     comp=_opts['OPT_Compression'],
                                                     name='B2Sl')
    B2Ml_ser, B2Ml_strain_ser = Bend.YM_eva_method_B(option="Fit", 
                                                     stress_mid_ser=d_stress_mid,
                                                     thickness = func_t(0.0), 
                                                     Length=Length,
                                                     w_func=bl['w_M']['d0'],
                                                     w_params=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                                     Length_det=10.0, 
                                                     comp = _opts['OPT_Compression'],
                                                     name = 'B2Ml')
    
    
    E_B_df = pd.concat([B1Al_ser,B1Sl_ser,B1Ml_ser,
                        B2Al_ser,B2Sl_ser,B2Ml_ser],axis=1)
    E_B = Evac.pd_agg(E_B_df.loc[sr_eva_dic])
    
    if True:
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig.suptitle('%s - Compare method B'%(plt_name))
        ax1.set_title('All Steps')
        ax1.set_xlabel('Step / -')
        ax1.set_ylabel('E / MPa')
        cc={'B1Al':'r:','B1Sl':'b:','B1Ml':'g:','B2Al':'r--','B2Sl':'b--','B2Ml':'g--'}
        for k in cc:
            ax1.plot(E_B_df.loc[:VIP_messu['U']].index,
                     E_B_df.loc[:VIP_messu['U']][k],
                     cc[k], label='%s - %.2f MPa'%(k,E_B.at['mean',k]))
        ax1.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
        ax1.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax1.legend()
        ax2.set_title('Improved determination range')
        ax2.set_xlabel('Step / -')
        ax2.set_ylabel('E / MPa')
        for k in cc:
            ax2.plot(sr_eva_dic, E_B_df[k].loc[sr_eva_dic], cc[k])
        Evac.plt_handle_suffix(fig,path=out_full+"-YM-Me_B",**plt_Fig_dict)
        
    # least-square-fit
    E_lsq_F_B1Al = Bend.YM_eva_method_B(option="Points",
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length = Length,
                                 P_df = P_xcoord_ydisp_meas,
                                 P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                 comp = _opts['OPT_Compression'],
                                 name='E_lsq_F_B1Al',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                    'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_B1Al = pd.Series(E_lsq_F_B1Al, index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_F_B1Al')
    
    E_lsq_F_B1Sl = Bend.YM_eva_method_B(option="Points",
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 P_df = P_xcoord_ydisp_meas_S,
                                 P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_F_B1Sl',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                    'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_B1Sl = pd.Series(E_lsq_F_B1Sl, index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_F_B1Sl')
    E_lsq_F_B1Ml = Bend.YM_eva_method_B(option="Points",
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 P_df = P_xcoord_ydisp_meas_M,
                                 P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_F_B1Ml',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                    'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_B1Ml = pd.Series(E_lsq_F_B1Ml, index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_F_B1Ml')
    
    E_lsq_F_B2Al = Bend.YM_eva_method_B(option="Fit", 
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 w_func=bl['w_A']['d0'],
                                 w_params=Pre_fit_df.loc(axis=1)['Fit_params_dict'],
                                 Length_det=10.0,
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_F_B2Al',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                    'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_B2Al = pd.Series(E_lsq_F_B2Al,index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_F_B2Al')
    E_lsq_F_B2Sl = Bend.YM_eva_method_B(option="Fit", 
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 w_func=bl['w_S']['d0'],
                                 w_params=Pre_fit_df.loc(axis=1)['Fit_params_dict'],
                                 Length_det=10.0,
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_F_B2Sl',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                    'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_B2Sl = pd.Series(E_lsq_F_B2Sl,index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_F_B2Sl')
    E_lsq_F_B2Ml = Bend.YM_eva_method_B(option="Fit", 
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 w_func=bl['w_M']['d0'], 
                                 w_params=Bend_fit_df.loc(axis=1)['Fit_params_dict'],
                                 Length_det=10.0, 
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_F_B2Ml',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_f[0]],
                                    'ind_E':VIP_dicu[Ind_YM_f[1]]})
    E_lsq_F_B2Ml = pd.Series(E_lsq_F_B2Ml,index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_F_B2Ml')
    
    E_lsq_R_B1Al = Bend.YM_eva_method_B(option="Points",
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length = Length,
                                 P_df = P_xcoord_ydisp_meas,
                                 P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                 comp = _opts['OPT_Compression'],
                                 name='E_lsq_R_B1Al',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                    'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_B1Al = pd.Series(E_lsq_R_B1Al, index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_R_B1Al')
    
    E_lsq_R_B1Sl = Bend.YM_eva_method_B(option="Points",
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 P_df = P_xcoord_ydisp_meas_S,
                                 P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_R_B1Sl',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                    'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_B1Sl = pd.Series(E_lsq_R_B1Sl, index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_R_B1Sl')
    E_lsq_R_B1Ml = Bend.YM_eva_method_B(option="Points",
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 P_df = P_xcoord_ydisp_meas_M,
                                 P_fork_names = _opts['OPT_DIC_Points_meas_fork'],
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_R_B1Ml',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                    'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_B1Ml = pd.Series(E_lsq_R_B1Ml, index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_R_B1Ml')
    
    E_lsq_R_B2Al = Bend.YM_eva_method_B(option="Fit", 
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 w_func=bl['w_A']['d0'],
                                 w_params=Pre_fit_df.loc(axis=1)['Fit_params_dict'],
                                 Length_det=10.0,
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_R_B2Al',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                    'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_B2Al = pd.Series(E_lsq_R_B2Al,index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_R_B2Al')
    E_lsq_R_B2Sl = Bend.YM_eva_method_B(option="Fit", 
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 w_func=bl['w_S']['d0'],
                                 w_params=Pre_fit_df.loc(axis=1)['Fit_params_dict'],
                                 Length_det=10.0,
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_R_B2Sl',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                    'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_B2Sl = pd.Series(E_lsq_R_B2Sl,index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_R_B2Sl')
    E_lsq_R_B2Ml = Bend.YM_eva_method_B(option="Fit", 
                                 stress_mid_ser=messu.Stress,
                                 thickness = func_t(0.0),
                                 Length=Length,
                                 w_func=bl['w_M']['d0'], 
                                 w_params=Bend_fit_df.loc(axis=1)['Fit_params_dict'],
                                 Length_det=10.0, 
                                 comp=_opts['OPT_Compression'],
                                 name='E_lsq_R_B2Ml',
                                 det_opt='leastsq',
                                 **{'ind_S':VIP_dicu[Ind_YM_r[0]],
                                    'ind_E':VIP_dicu[Ind_YM_r[1]]})
    E_lsq_R_B2Ml = pd.Series(E_lsq_R_B2Ml,index=['E','E_abs','Rquad','Fit_result','strain'],
                        name='E_lsq_R_B2Ml')
        
    E_lsq_B = pd.concat([E_lsq_F_B1Al,E_lsq_F_B1Sl,E_lsq_F_B1Ml,
                         E_lsq_F_B2Al,E_lsq_F_B2Sl,E_lsq_F_B2Ml,
                         E_lsq_R_B1Al,E_lsq_R_B1Sl,E_lsq_R_B1Ml,
                         E_lsq_R_B2Al,E_lsq_R_B2Sl,E_lsq_R_B2Ml],axis=1)
    
    del B1Al_strain_ser,B1Sl_strain_ser,B1Ml_strain_ser,B2Al_strain_ser,B2Sl_strain_ser,B2Ml_strain_ser
    del B1Al_ser,B1Sl_ser,B1Ml_ser,B2Al_ser,B2Sl_ser,B2Ml_ser
    del E_lsq_F_B1Al,E_lsq_F_B1Sl,E_lsq_F_B1Ml
    del E_lsq_F_B2Al,E_lsq_F_B2Sl,E_lsq_F_B2Ml
    del E_lsq_R_B1Al,E_lsq_R_B1Sl,E_lsq_R_B1Ml
    del E_lsq_R_B2Al,E_lsq_R_B2Sl,E_lsq_R_B2Ml

    #%%%% 6.4.3 Method C
    timings.loc[6.43]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    
    E_C2Al_ser = Bend.YM_eva_method_C(Force_ser=d_Force, w_func=bl['w_A']['d0'], 
                                  w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                  length=Length, I_func=func_I, A_func=func_A,
                                  CS_type='Rectangle', kappa=None,
                                  poisson=_opts['OPT_Poisson_prediction'],
                                  comp=_opts['OPT_Compression'], option="M", name="C2Al")
    E_C2Sl_ser = Bend.YM_eva_method_C(Force_ser=d_Force, w_func=bl['w_S']['d0'], 
                                  w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                  length=Length, I_func=func_I, A_func=func_A,
                                  CS_type='Rectangle', kappa=None,
                                  poisson=_opts['OPT_Poisson_prediction'],
                                  comp=_opts['OPT_Compression'], option="M", name="C2Sl")
    E_C2Ml_ser = Bend.YM_eva_method_C(Force_ser=d_Force, w_func=bl['w_M']['d0'], 
                                  w_params=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                  length=Length, I_func=func_I, A_func=func_A,
                                  CS_type='Rectangle', kappa=None,
                                  poisson=_opts['OPT_Poisson_prediction'],
                                  comp=_opts['OPT_Compression'], option="M", name="C2Ml")
    E_C2Cl_ser = Bend.YM_eva_method_C(Force_ser=d_Force, w_func=bl['w_S']['d0'], 
                                  w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                  length=Length, I_func=func_I, A_func=func_A,
                                  CS_type='Rectangle', kappa=None,
                                  poisson=_opts['OPT_Poisson_prediction'],
                                  comp=_opts['OPT_Compression'], option="M+V", name="C2Cl")
    
    E_C_df = pd.concat([E_C2Al_ser,E_C2Sl_ser,E_C2Ml_ser,E_C2Cl_ser],axis=1)
    E_C = Evac.pd_agg(E_C_df.loc[sr_eva_dic])
    
    if True:    
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig.suptitle('%s - Compare method C'%(plt_name))
        ax1.set_title('All Steps')
        ax1.set_xlabel('Step / -')
        ax1.set_ylabel('E / MPa')
        cc={'C2Al':'r:','C2Sl':'b:','C2Ml':'m--','C2Cl':'g--'}
        for k in cc:
            ax1.plot(E_C_df.loc[:VIP_messu['U']].index,
                     E_C_df.loc[:VIP_messu['U']][k],
                     cc[k], label='%s - %.2f MPa'%(k,E_C.at['mean',k]))
        ax1.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
        ax1.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax1.legend()
        ax2.set_title('Improved determination range')
        ax2.set_xlabel('Step / -')
        ax2.set_ylabel('E / MPa')
        for k in cc:
            ax2.plot(sr_eva_dic,E_C_df[k].loc[sr_eva_dic], cc[k])
        Evac.plt_handle_suffix(fig,path=out_full+"-YM-Me_C",**plt_Fig_dict)

    del E_C2Al_ser, E_C2Sl_ser, E_C2Ml_ser
    
    #%%%% 6.4.4 Method D
    timings.loc[6.44]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    
    # D2Mg_mwt,D2Mg_mwt_df=Bend.YM_eva_method_D(P_df=P_xcoord_ydiff_meas_M, Force_ser=d_Force,
    #                                             Length=Length, func_I=func_I, n=100,
    #                                             weighted=True, weight_func=Bend.Weight_func,
    #                                             wargs=['Custom_cut', bl['w_M']['d0']],
    #                                             wkwargs=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
    #                                             pb_b=True, name='D2Mg_mwt')  
    # D2Mg_muw,D2Mg_muw_df=Bend.YM_eva_method_D(P_df=P_xcoord_ydiff_meas_M, Force_ser=d_Force,
    #                                                Length=Length, func_I=func_I, n=100,
    #                                                weighted=True, weight_func=Bend.Weight_func,
    #                                                wkwargs={'option':'Cut',
    #                                                         'xmin':-Length/2,'xmax':Length/2},
    #                                                pb_b=True, name='D2Mg_muw')
    D1Agwt,D1Agwt_df=Bend.YM_eva_method_D(P_df=P_xcoord_ydiff_meas, Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wargs=['Custom_cut', bl['w_A']['d0']],
                                          wkwargs=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                          pb_b=True, name='D1Agwt')  
    D1Aguw,D1Aguw_df=Bend.YM_eva_method_D(P_df=P_xcoord_ydiff_meas, Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wkwargs={'option':'Cut',
                                                   'xmin':Bo_Le,'xmax':Bo_Ri},
                                          pb_b=True, name='D1Aguw')
    D1Sgwt,D1Sgwt_df=Bend.YM_eva_method_D(P_df=P_xcoord_ydiff_meas_S, Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wargs=['Custom_cut', bl['w_S']['d0']],
                                          wkwargs=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                          pb_b=True, name='D1Sgwt')  
    D1Sguw,D1Sguw_df=Bend.YM_eva_method_D(P_df=P_xcoord_ydiff_meas_S, Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wkwargs={'option':'Cut',
                                                   'xmin':Bo_Le,'xmax':Bo_Ri},
                                          pb_b=True, name='D1Sguw')
    D1Mgwt,D1Mgwt_df=Bend.YM_eva_method_D(P_df=P_xcoord_ydiff_meas_M, Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wargs=['Custom_cut', bl['w_M']['d0']],
                                          wkwargs=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                          pb_b=True, name='D1Mgwt')  
    D1Mguw,D1Mguw_df=Bend.YM_eva_method_D(P_df=P_xcoord_ydiff_meas_M, Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wkwargs={'option':'Cut',
                                                   'xmin':Bo_Le,'xmax':Bo_Ri},
                                          pb_b=True, name='D1Mguw')
    
    dftmp = Bend.Point_df_from_lin(xlin,step_range_dic)
    Diff_A_d0 = bl['w_A']['d0'](xlin,Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'])
    A_d0_inc_df = Bend.Point_df_combine(dftmp,Diff_A_d0)
    Diff_S_d0 = bl['w_S']['d0'](xlin,Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'])
    S_d0_inc_df = Bend.Point_df_combine(dftmp,Diff_S_d0)
    Diff_M_d0 = bl['w_M']['d0'](xlin,Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'])
    M_d0_inc_df = Bend.Point_df_combine(dftmp,Diff_M_d0)
    D2Agwt,D2Agwt_df=Bend.YM_eva_method_D(P_df=A_d0_inc_df,Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wargs=['Custom_cut', bl['w_A']['d0']],
                                          wkwargs=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                          pb_b=True, name='D2Agwt')
    D2Aguw,D2Aguw_df=Bend.YM_eva_method_D(P_df=A_d0_inc_df,Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wkwargs={'option':'Cut',
                                                   'xmin':Bo_Le,'xmax':Bo_Ri},
                                          pb_b=True, name='D2Aguw')
    D2Sgwt,D2Sgwt_df=Bend.YM_eva_method_D(P_df=S_d0_inc_df,Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wargs=['Custom_cut', bl['w_S']['d0']],
                                          wkwargs=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                          pb_b=True, name='D2Sgwt')
    D2Sguw,D2Sguw_df=Bend.YM_eva_method_D(P_df=S_d0_inc_df,Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wkwargs={'option':'Cut',
                                                   'xmin':Bo_Le,'xmax':Bo_Ri},
                                          pb_b=True, name='D2Sguw')
    D2Mgwt,D2Mgwt_df=Bend.YM_eva_method_D(P_df=M_d0_inc_df,Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wargs=['Custom_cut', bl['w_M']['d0']],
                                          wkwargs=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                          pb_b=True, name='D2Mgwt')
    D2Mguw,D2Mguw_df=Bend.YM_eva_method_D(P_df=M_d0_inc_df,Force_ser=d_Force,
                                          rel_steps = step_range_dic_inc,
                                          Length=Length, func_I=func_I, n=100,
                                          weighted=True, weight_func=Bend.Weight_func,
                                          wkwargs={'option':'Cut',
                                                   'xmin':Bo_Le,'xmax':Bo_Ri},
                                          pb_b=True, name='D2Mguw')
    
    E_D_df = pd.concat([D1Agwt,D1Aguw,D1Sgwt,D1Sguw,D1Mgwt,D1Mguw,
                        D2Agwt,D2Aguw,D2Sgwt,D2Sguw,D2Mgwt,D2Mguw],axis=1)
    E_D = Evac.pd_agg(E_D_df.loc[sr_eva_dic])
    
    if True:        
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig.suptitle('%s - Compare method D'%(plt_name))
        ax1.set_title('All Steps')
        ax1.set_xlabel('Step / -')
        ax1.set_ylabel('E / MPa')
        cc={'D1Agwt':'r:','D1Sgwt':'b:','D1Mguw':'m:','D1Mgwt':'g:',
            'D2Agwt':'r--','D2Sgwt':'b--','D2Mguw':'m--','D2Mgwt':'g--'}
        for k in cc:
            ax1.plot(E_D_df.loc[:VIP_messu['U']].index,
                     E_D_df.loc[:VIP_messu['U']][k],
                     cc[k], label='%s - %.2f MPa'%(k,E_D.at['mean',k]))
        ax1.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
        ax1.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax1.legend()
        ax2.set_title('Improved determination range')
        ax2.set_xlabel('Step / -')
        ax2.set_ylabel('E / MPa')
        for k in cc:
            ax2.plot(sr_eva_dic,E_D_df[k].loc[sr_eva_dic],
                     cc[k], label='%s - %.2f MPa'%(k,E_D.at['mean',k]))
        Evac.plt_handle_suffix(fig,path=out_full+"-YM-Me_D",**plt_Fig_dict)
    
    del D1Agwt,D1Aguw,D1Sgwt,D1Sguw,D1Mgwt,D1Mguw
    del D2Agwt,D2Aguw,D2Sgwt,D2Sguw,D2Mgwt,D2Mguw
    del D1Agwt_df,D1Aguw_df,D1Sgwt_df,D1Sguw_df,D1Mgwt_df,D1Mguw_df
    del D2Agwt_df,D2Aguw_df,D2Sgwt_df,D2Sguw_df,D2Mgwt_df,D2Mguw_df
    
    #%%%% 6.4.5 Method E
    timings.loc[6.45]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    
    E4A = Bend.YM_eva_method_E(Force_ser=d_Force, length=Length,
                               func_curve=bl['w_A']['d2'], 
                               params_curve=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                               func_MoI=func_I,
                               func_thick=func_t, evopt=1/2,
                               opt_det='all', n=100,
                               opt_g='strain', opt_g_lim=0.5, name='E4A')
    E4A_E_df, E4A_sig_eps_df, E4A_E_to_x, E4A_stress_df, E4A_strain_df, E4A_E_to_x_g = E4A

    E4S = Bend.YM_eva_method_E(Force_ser=d_Force, length=Length,
                               func_curve=bl['w_S']['d2'], 
                               params_curve=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                               func_MoI=func_I,
                               func_thick=func_t, evopt=1/2,
                               opt_det='all', n=100,
                               opt_g='strain', opt_g_lim=0.5, name='E4S')
    E4S_E_df, E4S_sig_eps_df, E4S_E_to_x, E4S_stress_df, E4S_strain_df, E4S_E_to_x_g = E4S

    E4M = Bend.YM_eva_method_E(Force_ser=d_Force, length=Length,
                               func_curve=bl['w_M']['d2'], 
                               params_curve=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                               func_MoI=func_I, 
                               func_thick=func_t, evopt=1/2,
                               opt_det='all', n=100,
                               opt_g='strain', opt_g_lim=0.5, name='E4M')
    E4M_E_df, E4M_sig_eps_df, E4M_E_to_x, E4M_stress_df, E4M_strain_df, E4M_E_to_x_g = E4M

    E_E_df = pd.concat([E4A_E_df, E4S_E_df, E4M_E_df],axis=1)
    E_E = Evac.pd_agg(E_E_df.loc[sr_eva_dic])
    
    if True:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize = (6.3,3*3.54))
        fig.suptitle("%s - M - Stress, Strain, Youngs-Modulus to x-coordinate for F3 to F4"%(plt_name))
        ax1 = Bend.Plotter.colplt_common_ax(xdata=E4M_stress_df.columns,
                                               ydata=E4M_stress_df,
                                               step_range=sr_eva_dic, ax=ax1,
                                               title='Stress to x over Increment',
                                               xlabel='x / mm', 
                                               ylabel='$\sigma$ / MPa')
        ax1.axvline(x=0, color='gray', linestyle='--')
        ax2 = Bend.Plotter.colplt_common_ax(xdata=E4M_strain_df.columns,
                                               ydata=E4M_strain_df,
                                               step_range=sr_eva_dic, ax=ax2,
                                               title='Strain to x over Increment',
                                               xlabel='x / mm', 
                                               ylabel='$\sigma$ / MPa')
        ax2.axvline(x=0, color='gray', linestyle='--')
        ax3 = Bend.Plotter.colplt_common_ax(xdata=E4M_E_to_x_g.columns,
                                               ydata=E4M_E_to_x_g,
                                               step_range=sr_eva_dic, ax=ax3,
                                               title='Youngs Modulus to x over Increment',
                                               xlabel='x / mm',
                                               ylabel='E / MPa')
        ax3.axvline(x=0, color='gray', linestyle='--')
        Evac.plt_handle_suffix(fig,path=None,**plt_Fig_dict)
        
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig.suptitle('%s - Compare method E'%(plt_name))
        ax1.set_title('All Steps')
        ax1.set_xlabel('Step / -')
        ax1.set_ylabel('E / MPa')
        cc={'E4Sl':'r:','E4Se':'b:','E4Sg':'g:',
            'E4Ml':'r--','E4Me':'b--','E4Mg':'g--'}
        for k in cc:
            ax1.plot(E_E_df.loc[:VIP_messu['U']].index,
                     E_E_df.loc[:VIP_messu['U']][k],
                     cc[k], label='%s - %.2f MPa'%(k,E_E.at['mean',k]))
        ax1.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
        ax1.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax1.legend()
        ax2.set_title('Improved determination range')
        ax2.set_xlabel('Step / -')
        ax2.set_ylabel('E / MPa')
        for k in cc:
            ax2.plot(sr_eva_dic,E_E_df[k].loc[sr_eva_dic], cc[k])
        Evac.plt_handle_suffix(fig,path=out_full+"-YM-Me_E",**plt_Fig_dict)
        
    del E4A, E4A_E_df, E4A_sig_eps_df, E4A_E_to_x, E4A_stress_df, E4A_strain_df, E4A_E_to_x_g
    del E4S, E4S_E_df, E4S_sig_eps_df, E4S_E_to_x, E4S_stress_df, E4S_strain_df, E4S_E_to_x_g
    del E4M, E4M_E_df, E4M_sig_eps_df, E4M_E_to_x, E4M_stress_df, E4M_strain_df, E4M_E_to_x_g

    #%%%% 6.4.6 Method F
    timings.loc[6.46]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    
    F4A_df = Bend.YM_eva_method_F(c_func=bl['w_A']['d2'],
                              c_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                              Force_ser=d_Force, Length=Length, func_I=func_I,
                              weighted=True, weight_func=Bend.Weight_func,
                              wargs=['Custom_cut', bl['w_A']['d0']],
                              wkwargs=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                              xr_dict = {'fu':1/1, 'ha':1/2, 'th':1/3},
                              pb_b=True, name='F4Ag', n=100)
    
    F4S_df = Bend.YM_eva_method_F(c_func=bl['w_S']['d2'],
                              c_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                              Force_ser=d_Force, Length=Length, func_I=func_I,
                              weighted=True, weight_func=Bend.Weight_func,
                              wargs=['Custom_cut', bl['w_S']['d0']],
                              wkwargs=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                              xr_dict = {'fu':1/1, 'ha':1/2, 'th':1/3},
                              pb_b=True, name='F4Sg', n=100)
    
    F4M_df = Bend.YM_eva_method_F(c_func=bl['w_M']['d2'],
                              c_params=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                              Force_ser=d_Force, Length=Length, func_I=func_I,
                              weighted=True, weight_func=Bend.Weight_func,
                              wargs=['Custom_cut', bl['w_M']['d0']],
                              wkwargs=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                              xr_dict = {'fu':1/1, 'ha':1/2, 'th':1/3},
                              pb_b=True, name='F4Mg', n=100)
    
    E_F_df = pd.concat([F4A_df, F4S_df, F4M_df],axis=1)
    E_F = Evac.pd_agg(E_F_df.loc[sr_eva_dic])
    
    if True:    
        cc={'F4Agha':'r:', 'F4Agth':'r--',
            'F4Sgha':'b:', 'F4Sgth':'b--',
            'F4Mgha':'g:', 'F4Mgth':'g--'}
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig.suptitle('%s - Compare method F'%(plt_name))
        ax1.set_title('All Steps')
        ax1.set_xlabel('Step / -')
        ax1.set_ylabel('E / MPa')
        for k in cc:
            ax1.plot(E_F_df.loc[:VIP_messu['U']].index,
                     E_F_df.loc[:VIP_messu['U']][k],
                     cc[k], label='%s - %.2f MPa'%(k,E_F.at['mean',k]))
        ax1.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
        ax1.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax1.legend()
        ax2.set_title('Improved determination range')
        ax2.set_xlabel('Step / -')
        ax2.set_ylabel('E / MPa')
        for k in cc:
            ax2.plot(sr_eva_dic,E_F_df[k].loc[sr_eva_dic], cc[k])
        Evac.plt_handle_suffix(fig,path=out_full+"-YM-Me_F",**plt_Fig_dict)
        
    del F4A_df, F4S_df, F4M_df
    
    #%%%% 6.4.7 Method G
    timings.loc[6.47]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    
    E_G3Ag_ser = Bend.YM_eva_method_G(Force_ser=d_Force,
                                      w_func_f_0=bl['w_A']['d0'], 
                                      w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      c_func=bl['w_A']['d2'],
                                      r_func=None,
                                      c_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      r_params=None,
                                      length=Length, I_func=func_I, A_func=func_A,
                                      CS_type='Rectangle', kappa=None,
                                      poisson=_opts['OPT_Poisson_prediction'],
                                      comp=_opts['OPT_Compression'],
                                      option="ignore_V", name="G3Ag")

    E_G3Sg_ser = Bend.YM_eva_method_G(Force_ser=d_Force,
                                      w_func_f_0=bl['w_S']['d0'], 
                                      w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      c_func=bl['w_S']['d2'],
                                      r_func=None,
                                      c_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      r_params=None,
                                      length=Length, I_func=func_I, A_func=func_A,
                                      CS_type='Rectangle', kappa=None,
                                      poisson=_opts['OPT_Poisson_prediction'],
                                      comp=_opts['OPT_Compression'],
                                      option="ignore_V", name="G3Sg")

    E_G3Mg_ser = Bend.YM_eva_method_G(Force_ser=d_Force,
                                      w_func_f_0=bl['w_M']['d0'], 
                                      w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      c_func=bl['w_M']['d2'],
                                      r_func=None,
                                      c_params=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      r_params=None,
                                      length=Length, I_func=func_I, A_func=func_A,
                                      CS_type='Rectangle', kappa=None,
                                      poisson=_opts['OPT_Poisson_prediction'],
                                      comp=_opts['OPT_Compression'],
                                      option="ignore_V", name="G3Mg")
    
    E_G3Cg_ser = Bend.YM_eva_method_G(Force_ser=d_Force,
                                      w_func_f_0=bl['w_S']['d0'], 
                                      w_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      c_func=bl['w_M']['d2'],
                                      r_func=bl['w_V']['d1'],
                                      c_params=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      r_params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                      length=Length, I_func=func_I, A_func=func_A,
                                      CS_type='Rectangle', kappa=None,
                                      poisson=_opts['OPT_Poisson_prediction'],
                                      comp=_opts['OPT_Compression'],
                                      option="M+V", name="G3Cg")
    
    E_G_df = pd.concat([E_G3Ag_ser, E_G3Sg_ser, E_G3Mg_ser, E_G3Cg_ser],axis=1)
    E_G = Evac.pd_agg(E_G_df.loc[sr_eva_dic])
    
    if True:    
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize = (6.3,2*3.54))
        fig.suptitle('%s - Compare method G'%(plt_name))
        ax1.set_title('All Steps')
        ax1.set_xlabel('Step / -')
        ax1.set_ylabel('E / MPa')
        # cc={'G3Sg':'r:','G3Mg':'b--'}
        cc={'G3Ag':'r:', 'G3Sg':'b:', 'G3Mg':'m--', 'G3Cg':'g--'}
        for k in cc:
            ax1.plot(E_G_df.loc[:VIP_messu['U']].index,
                     E_G_df.loc[:VIP_messu['U']][k],
                     cc[k], label='%s - %.2f MPa'%(k,E_G.at['mean',k]))
        ax1.axvline(x=VIP_dicu['F3'], color='olive', linestyle=':')
        ax1.axvline(x=VIP_dicu['F4'], color='olive', linestyle='--')
        ax1.legend()
        ax2.set_title('Improved determination range')
        ax2.set_xlabel('Step / -')
        ax2.set_ylabel('E / MPa')
        for k in cc:
            ax2.plot(sr_eva_dic,E_G_df[k].loc[sr_eva_dic], cc[k])
        Evac.plt_handle_suffix(fig,path=out_full+"-YM-Me_G",**plt_Fig_dict)
    
    del E_G3Ag_ser, E_G3Sg_ser, E_G3Mg_ser, E_G3Cg_ser

    #%%%% 6.4.8 Method compare
    timings.loc[6.48]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    


    E_lsq=pd.concat([E_lsq_A,E_lsq_B],axis=1)

    # E_Methods = pd.concat([E_A,E_B,E_C,E_D,E_E,E_F,E_G],axis=1)
    E_Methods_df = pd.concat([E_A_df,E_B_df,E_C_df,E_D_df,E_E_df,E_F_df,E_G_df],axis=1)
    # E_agg_funcs = ['mean',Evac.meanwoso,'median','std','max','min']
    E_agg_funcs = ['mean',Evac.meanwoso,'median','std',Evac.stdwoso,'max','min']
    
    cols_con=E_Methods_df.columns.str.contains('0')
    # E_inc_F_comp_con = Evac.pd_agg(E_Methods_df.loc[sf_eva_con,cols_con])
    # E_inc_F_comp_opt = Evac.pd_agg(E_Methods_df.loc[sf_eva_dic,np.invert(cols_con)])
    E_inc_F_comp_con = E_Methods_df.loc[sf_eva_con,cols_con].agg(E_agg_funcs)
    E_inc_F_comp_opt = E_Methods_df.loc[sf_eva_dic,np.invert(cols_con)].agg(E_agg_funcs)
    E_inc_F_comp = pd.concat([E_inc_F_comp_con,E_inc_F_comp_opt],axis=1)
    E_inc_F_comp.loc['stdn']=E_inc_F_comp.loc['std']/E_inc_F_comp.loc['mean'].abs()
    # E_inc_F_comp.loc['stdnwoso']=E_inc_F_comp.loc['std']/E_inc_F_comp.loc['meanwoso'].abs()
    E_inc_F_comp.loc['stdnwoso']=E_inc_F_comp.loc['stdwoso']/E_inc_F_comp.loc['meanwoso'].abs()
    # E_inc_F_comp = Evac.pd_agg(E_Methods_df.loc[sf_eva_dic])

    cols_con=E_Methods_df.columns.str.contains('0')
    # E_inc_R_comp_con = Evac.pd_agg(E_Methods_df.loc[sr_eva_con,cols_con])
    # E_inc_R_comp_opt = Evac.pd_agg(E_Methods_df.loc[sr_eva_dic,np.invert(cols_con)])
    E_inc_R_comp_con = E_Methods_df.loc[sr_eva_con,cols_con].agg(E_agg_funcs)
    E_inc_R_comp_opt = E_Methods_df.loc[sr_eva_dic,np.invert(cols_con)].agg(E_agg_funcs)
    E_inc_R_comp = pd.concat([E_inc_R_comp_con,E_inc_R_comp_opt],axis=1)
    E_inc_R_comp.loc['stdn']=E_inc_R_comp.loc['std']/E_inc_R_comp.loc['mean'].abs()
    # E_inc_R_comp.loc['stdnwoso']=E_inc_R_comp.loc['std']/E_inc_R_comp.loc['meanwoso'].abs()
    E_inc_R_comp.loc['stdnwoso']=E_inc_R_comp.loc['stdwoso']/E_inc_R_comp.loc['meanwoso'].abs()
    
    Evac.MG_strlog("\n\n  Method comaparison:",log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n  - least square fit",log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent('\n'+E_lsq.loc[['E','Rquad']].T.to_string()),
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n\n  - incremental - fixed range:",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent('\n'+E_inc_F_comp.T.to_string()),
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent('\n'+E_inc_F_comp.agg(['idxmax','idxmin'],axis=1).to_string()),
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n\n  - incremental - refined range:",log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent('\n'+E_inc_R_comp.T.to_string()),
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent('\n'+E_inc_R_comp.agg(['idxmax','idxmin'],axis=1).to_string()),
                   log_mg,output_lvl,printopt=True)


    # check_M_tmp=E_Methods_df.loc(axis=1)[E_Methods_df.columns.str.contains('M')]
    check_M_tmp=E_Methods_df.loc(axis=1)[E_Methods_df.columns.str.contains('M|Cl|Ce|Cg')]
    check_M_dict={i: check_M_tmp[i] for i in check_M_tmp.columns}
    Check_M = Bend.YM_check_many_with_method_D(E_dict=check_M_dict, F=d_Force,
                                                 Length=Length,I_func=func_I,
                                                 w_func=bl['w_M']['d0'],
                                                 w_params=Bend_inc_fit_df['Fit_params_dict'],
                                                 rel_steps=sr_eva_dic, n=100,pb_b=False,name='M')
    check_MtoD_g, check_MtoD_x, check_M, w_D_to_M = Check_M
    
    Evac.MG_strlog("\n\n  Check Methods with D (M-Displacement deviation):",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n  - mean over all (except first and last):",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent(check_MtoD_g.mean()),
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n  - only in mid:",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent(check_MtoD_x.mean()),
                   log_mg,output_lvl,printopt=True)
    
    check_S_tmp=E_Methods_df.loc(axis=1)[E_Methods_df.columns.str.contains('S')]
    check_S_dict={i: check_S_tmp[i] for i in check_S_tmp.columns}
    Check_S = Bend.YM_check_many_with_method_D(E_dict=check_S_dict, F=d_Force,
                                               Length=Length,I_func=func_I,
                                               w_func=bl['w_S']['d0'],
                                               w_params=Pre_inc_fit_df['Fit_params_dict'],
                                               rel_steps=sr_eva_dic,
                                               n=100,pb_b=False,name='S')
    check_StoD_g, check_StoD_x, check_S, w_D_to_S = Check_S
    
    Evac.MG_strlog("\n\n  Check Methods with D (S-Displacement deviation):",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n  - mean over all (except first and last):",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent(check_StoD_g.mean()),
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n  - only in mid:",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent(check_StoD_x.mean()),
                   log_mg,output_lvl,printopt=True)
    
    check_A_tmp=E_Methods_df.loc(axis=1)[E_Methods_df.columns.str.contains('Al|Ae|Ag')]
    check_A_dict={i: check_A_tmp[i] for i in check_A_tmp.columns}
    Check_A = Bend.YM_check_many_with_method_D(E_dict=check_A_dict, F=d_Force,
                                               Length=Length,I_func=func_I,
                                               w_func=bl['w_A']['d0'],
                                               w_params=Pre_inc_fit_df['Fit_params_dict'],
                                               rel_steps=sr_eva_dic,
                                               n=100,pb_b=False,name='A')
    check_AtoD_g, check_AtoD_x, check_A, w_D_to_A = Check_A
    
    Evac.MG_strlog("\n\n  Check Methods with D (A-Displacement deviation):",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n  - mean over all (except first and last):",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent(check_AtoD_g.mean()),
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog("\n  - only in mid:",
                   log_mg,output_lvl,printopt=True)
    Evac.MG_strlog(Evac.str_indent(check_AtoD_x.mean()),
                   log_mg,output_lvl,printopt=True)
    
    Check_to_D = pd.Series({'MtoD_g':check_MtoD_g, 'MtoD_x': check_MtoD_x,
                            'M':check_M, 'MwD': w_D_to_M,
                            'StoD_g':check_StoD_g, 'StoD_x': check_StoD_x,
                            'S':check_S, 'SwD': w_D_to_S,
                            'AtoD_g':check_AtoD_g, 'AtoD_x': check_AtoD_x,
                            'A':check_A, 'AwD': w_D_to_A},dtype='O')
    
    # set preffered Method
    YM_pref_con=E_lsq_A['E_lsq_R_A0Al']
    if loc_Yd_tmp.startswith('E_lsq'):
        YM_pref_opt=E_lsq[loc_Yd_tmp]
    elif loc_Yd_tmp.startswith('E_inc_R'):
        YM_pref_opt=Evac.Refit_YM_vals(m_df=messu, 
                                       YM = E_inc_R_comp[loc_Yd_tmp.split('_')[-1]]['meanwoso'], 
                                       VIP=VIP_dicu,
                                       n_strain=dic_used_Strain, n_stress='Stress',
                                       n_loBo=['F3'], n_upBo=['F4'])
    elif loc_Yd_tmp.startswith('E_inc_F'):
        YM_pref_opt=Evac.Refit_YM_vals(m_df=messu, 
                                       YM = E_inc_F_comp[loc_Yd_tmp.split('_')[-1]]['meanwoso'], 
                                       VIP=VIP_dicu,
                                       n_strain=dic_used_Strain, n_stress='Stress',
                                       n_loBo=['F1'], n_upBo=['F2'])
    else:
        raise NotImplementedError('Prefered YM-method not selectable!')
    # --------------------------------------------------------------------------
    #%%% 6.5 Determine yield point
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### -6.5 Determine yield point ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.5]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    
    # #     mit 0.2% Dehnversatz E(F+-F-) finden
    # Evac.MG_strlog("\n  Determination of yield strain-conventional:",log_mg,output_lvl)
    # mun_tmp=messu.loc[min(VIP_messu[['F1','F3']]):VIP_messu['U']]
    # # stress_fit3=(mun_tmp.Strain+(-.002))*E2+E2_abs
    # stress_fit3=(mun_tmp.Strain+(-.002))*E_lsq_A['E_lsq_R_A0Al']['E']+E_lsq_A['E_lsq_R_A0Al']['E_abs']
    # c=mun_tmp.Stress-stress_fit3
    # c_sign = np.sign(c)
    # c_signchange = ((np.roll(c_sign, True) - c_sign) != False).astype(bool)
    # c_signchange = pd.Series(c_signchange,index=c.index)
    # c_signchange = c_signchange.loc[min(VIP_messu[['F2','F4']]):] # Y erst ab F- suchen
    # daF_signchange_F=messu.driF_schg.loc[min(VIP_messu[['F1','F3']])+2:VIP_messu['U']+2]
    
    # if (c_signchange.any()==True)and(daF_signchange_F.any()==True):
    #     if(c_signchange.loc[c_signchange].index[0])<=(daF_signchange_F.loc[daF_signchange_F==True].index[0]):
    #         VIP_messu['Y']=c_signchange.loc[c_signchange].index[0]
    #         Evac.MG_strlog("\n    Fy set on intersection with 0.2% pl. strain!",log_mg,output_lvl)
    #     else:
    #         VIP_messu['Y']=daF_signchange_F.loc[daF_signchange_F==True].index[0]-2 #-2 wegen Anstiegsberechnung
    #         Evac.MG_strlog("\n    Fy on first point between F+ and Fu with rise of 0, instead intersection 0.2% pl. strain! (earlier)",log_mg,output_lvl)
    # elif (c_signchange.any()==True)and(daF_signchange_F.all()==False):
    #     VIP_messu['Y']=c_signchange.loc[c_signchange].index[0]
    #     Evac.MG_strlog("\n    Fy set on intersection with 0.2% pl. strain!",log_mg,output_lvl)
    # else:
    #     # daF_signchange_F=messu.driF_schg.loc[VIP_messu['F+']+2:VIP_messu['U']-1]
    #     VIP_messu['Y']=daF_signchange_F.loc[daF_signchange_F==True].index[0]-2
    #     Evac.MG_strlog("\n    Fy on first point between F+ and Fu with rise of 0, instead intersection 0.2% pl. strain! (No intersection found!)",log_mg,output_lvl)
    # VIP_messu=VIP_messu.sort_values()
    
    # if _opts['OPT_DIC']:    
    #     # VIP_dicu['Y']=VIP_messu['Y']
    #     Evac.MG_strlog("\n  Determination of yield strain-optical:",log_mg,output_lvl)
    #     mun_tmp=messu.loc[min(VIP_dicu[['F1','F3']]):VIP_dicu['U']]
    #     # stress_fit3=(mun_tmp.loc[:,dic_used_Strain]+(-.002))*ED2+ED2_abs
    #     stress_fit3=(mun_tmp.loc[:,dic_used_Strain]+(-.002))*E_lsq_A[loc_Yd_tmp]['E']+E_lsq_A[loc_Yd_tmp]['E_abs']
    #     c=mun_tmp.Stress-stress_fit3
    #     c_sign = np.sign(c)
    #     c_signchange = ((np.roll(c_sign, True) - c_sign) != False).astype(bool)
    #     c_signchange = pd.Series(c_signchange,index=c.index)
    #     c_signchange = c_signchange.loc[min(VIP_dicu[['F2','F4']]):] # Y erst ab F- suchen
    #     daF_signchange_F=messu.driF_schg.loc[min(VIP_dicu[['F1','F3']])+2:VIP_dicu['U']+2]
        
    #     if (c_signchange.any()==True)and(daF_signchange_F.any()==True):
    #         if(c_signchange.loc[c_signchange].index[0])<=(daF_signchange_F.loc[daF_signchange_F==True].index[0]):
    #             VIP_dicu['Y']=c_signchange.loc[c_signchange].index[0]
    #             Evac.MG_strlog("\n    Fy set on intersection with 0.2% pl. strain!",log_mg,output_lvl)
    #         else:
    #             VIP_dicu['Y']=daF_signchange_F.loc[daF_signchange_F==True].index[0]-2 #-2 wegen Anstiegsberechnung
    #             Evac.MG_strlog("\n    Fy on first point between F+ and Fu with rise of 0, instead intersection 0.2% pl. strain! (earlier)",log_mg,output_lvl)
    #     elif (c_signchange.any()==True)and(daF_signchange_F.all()==False):
    #         VIP_dicu['Y']=c_signchange.loc[c_signchange].index[0]
    #         Evac.MG_strlog("\n    Fy set on intersection with 0.2% pl. strain!",log_mg,output_lvl)
    #     else:
    #         # daF_signchange_F=messu.driF_schg.loc[VIP_dicu['F+']+2:VIP_dicu['U']-1]
    #         VIP_dicu['Y']=daF_signchange_F.loc[daF_signchange_F==True].index[0]-2
    #         Evac.MG_strlog("\n    Fy on first point between F+ and Fu with rise of 0, instead intersection 0.2% pl. strain! (No intersection found!)",log_mg,output_lvl)
    #     VIP_dicu=VIP_dicu.sort_values()

    
    strain_osd = {'YK':0.0,'Y0':0.0,'Y1':0.007/100,'Y2':0.1/100,'Y':0.2/100} #acc. Zhang et al. 2021, DOI: 10.1007/s10439-020-02719-2
    strain_osdf={'YK':'F4'}
        
    Evac.MG_strlog("\n  Determination of yield strain-conventional:",log_mg,output_lvl)
    # for i in strain_osd.keys():
    #     if i=='YK':
    #         VIP_messu[i] = VIP_messu['F4']
    #         txt="YK set on F4 (end of Keuherleber approach)"
    #         Evac.MG_strlog(Evac.str_indent(txt,3), log_mg,output_lvl)
    #     else:
    #         tmp = Evac.Yield_redet2(m_df=messu, VIP=VIP_messu,
    #                                   n_strain='Strain', n_stress='Stress',
    #                                   n_loBo=['F3'], n_upBo=['U'], n_loBo_int=['F3'],
    #                                   YM     = YM_pref_con['E'],
    #                                   YM_abs = YM_pref_con['E_abs'],
    #                                   strain_offset=strain_osd[i],
    #                                   use_rd =True, rise_det=[True,2], 
    #                                   n_yield=i, ywhere='n')
    #         VIP_messu,txt,tmp = tmp
    #         Evac.MG_strlog(Evac.str_indent(txt,3), log_mg,output_lvl)
    tmp=Evac.Yield_redet2_Multi(m_df=messu, VIP=VIP_messu,
                           strain_osd=strain_osd, 
                           strain_osdf=strain_osdf,
                           n_strain='Strain', n_stress='Stress',
                           n_loBo=['F3'], n_upBo=['U'], n_loBo_int=['F3'],
                           YM     = YM_pref_con['E'],
                           YM_abs = YM_pref_con['E_abs'],
                           use_rd =True, rise_det=[True,2], 
                           ywhere='n')
    VIP_messu, yield_df_con, txt = tmp
    Evac.MG_strlog(Evac.str_indent(txt,3), log_mg,output_lvl)
    if _opts['OPT_DIC']:
        Evac.MG_strlog("\n  Determination of yield strain-optical:",log_mg,output_lvl)
      # for i in strain_osd.keys():
      #   if i=='YK':
      #       VIP_dicu[i] = VIP_dicu['F4']
      #       txt="YK set on F4 (end of Keuherleber approach)"
      #       Evac.MG_strlog(Evac.str_indent(txt,3), log_mg,output_lvl)
      #   else:
      #       tmp = Evac.Yield_redet2(m_df=messu, VIP=VIP_dicu,
      #                                 n_strain=dic_used_Strain, n_stress='Stress',
      #                                 n_loBo=['F3'], n_upBo=['U'], n_loBo_int=['F3'],
      #                                 YM     = YM_pref_opt['E'],
      #                                 YM_abs = YM_pref_opt['E_abs'],
      #                                 strain_offset=strain_osd[i],
      #                                 use_rd =True,rise_det=[True,2], 
      #                                 n_yield=i, ywhere='n')
      #       VIP_dicu,txt,tmp = tmp
      #       Evac.MG_strlog(Evac.str_indent(txt,3), log_mg,output_lvl)
        tmp=Evac.Yield_redet2_Multi(m_df=messu, VIP=VIP_dicu,
                               strain_osd=strain_osd, 
                               strain_osdf=strain_osdf,
                               n_strain=dic_used_Strain, n_stress='Stress',
                               n_loBo=['F3'], n_upBo=['U'], n_loBo_int=['F3'],
                               YM     = YM_pref_opt['E'],
                               YM_abs = YM_pref_opt['E_abs'],
                               use_rd =True, rise_det=[True,2], 
                               ywhere='n')
        VIP_dicu, yield_df_opt, txt = tmp
        Evac.MG_strlog(Evac.str_indent(txt,3), log_mg,output_lvl)
        
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, 
                                  sharex=False, sharey=False, 
                                  figsize = (6.3,2*3.54))
    fig.suptitle('%s - Stress vs. strain curve - yield point determination'%plt_name)
    ax1.set_title('Conventional strain')
    ax1.set_xlabel('Strain / -')
    ax1.set_ylabel('Stress / MPa')
    ax1.plot(messu.loc[:VIP_messu['B']]['Strain'], messu.loc[:VIP_messu['B']]['Stress'], 'r--',label='con')
    tmp=ax1.get_xlim(),ax1.get_ylim()
    for i in strain_osd.keys():
        ax1.plot(Evac.strain_linfit([0,1000],
                                    YM_pref_con['E'], 
                                    YM_pref_con['E_abs'], 
                                    strain_offset=strain_osd[i]),[0,1000],
                 '-',label='$E_{off-%.3fp}$'%(strain_osd[i]*100))
    ax1.set_xlim(tmp[0])
    ax1.set_ylim(tmp[1])
    a, b=messu.Strain[VIP_messu[:'B']],messu.Stress[VIP_messu[:'B']]
    j=np.int64(-1)
    ax1.plot(a, b, 'bx')
    for x in VIP_messu[:'B'].index:
        j+=1
        if j%2: c=(6,-6)
        else:   c=(-6,6)
        ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                      xytext=c, ha="center", va="center", textcoords='offset points')
    ax1.legend()
    ax2.set_title('Optical strain')
    if _opts['OPT_DIC']:
        ax2.set_xlabel('Strain / -')
        ax2.set_ylabel('Stress / MPa')
        ax2.plot(messu.loc[:VIP_dicu['B']][dic_used_Strain], messu.loc[:VIP_dicu['B']]['Stress'], 
                 'r--',label='opt')
        tmp=ax2.get_xlim(),ax1.get_ylim()
        for i in strain_osd.keys():
            ax2.plot(Evac.strain_linfit([0,1000],
                                        YM_pref_opt['E'], 
                                        YM_pref_opt['E_abs'], 
                                        strain_offset=strain_osd[i]),[0,1000],
                     '-',label='$E_{off-%.3fp}$'%(strain_osd[i]*100))
        ax2.set_xlim(tmp[0])
        ax2.set_ylim(tmp[1])
        a, b=messu[dic_used_Strain][VIP_dicu[:'B']],messu.Stress[VIP_dicu[:'B']]
        j=np.int64(-1)
        ax2.plot(a, b, 'bx')
        for x in VIP_dicu[:'B'].index:
            j+=1
            if j%2: c=(6,-6)
            else:   c=(-6,6)
            ax2.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                          xytext=c, ha="center", va="center", textcoords='offset points')
        ax2.legend()
    Evac.plt_handle_suffix(fig,path=out_full+"-sigeps_yielddet",**plt_Fig_dict)
    
    # ============================================================================
    #%%% 6.6 Determine final curve
    Evac.MG_strlog("\n "+"-"*100,log_mg,output_lvl,printopt=False)
    Evac.MG_strlog("\n ### -6.6 Determine final stress-strain curve ###",log_mg,output_lvl,printopt=False)
    timings.loc[6.6]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)
    FinSSC = True # finale kurve für werteermittlung auf Ym linearisieren an anfang
    if FinSSC:
        messu_FP,linstrainos_con=Evac.DetFinSSC(mdf=messu, YM=YM_pref_con, 
                                     iS=VIP_messu['F3'], iLE=None,
                                     StressN='Stress', StrainN='Strain', 
                                     addzero=True, izero=VIP_messu['S'])
        yield_df_con['Strain']=yield_df_con['Strain']-linstrainos_con
        yield_df_con['Force']=yield_df_con['Stress']/Bend.stress_perF(0.0,func_I,func_t,Bo_Le,Bo_Ri)
        yield_df_con['Way']=(yield_df_con['Strain']*Length**2)/(6*prot_ser['thickness_2'])
        Evac.MG_strlog("\n   Strain offset (conventional) about %.5f"%(linstrainos_con),log_mg,output_lvl,printopt=False)
        if _opts['OPT_DIC']:
            tmp,     linstrainos_opt=Evac.DetFinSSC(mdf=messu, YM=YM_pref_opt, 
                                         iS=VIP_dicu['F3'], iLE=None,
                                         StressN='Stress', StrainN=dic_used_Strain, 
                                         addzero=True, izero=VIP_messu['S'])
            messu_FP=messu_FP.join(tmp, how='outer', lsuffix='', rsuffix='_opt')
            messu_FP['Stress'].fillna(messu_FP['Stress_opt'],inplace=True)
            yield_df_opt[dic_used_Strain]=yield_df_opt[dic_used_Strain]-linstrainos_opt
            del messu_FP['Stress_opt']
            yield_df_opt['Force']=yield_df_opt['Stress']/Bend.stress_perF(0.0,func_I,func_t,Bo_Le,Bo_Ri)
            yield_df_opt[dic_used_Disp]=(yield_df_opt[dic_used_Strain]*Length**2)/(6*prot_ser['thickness_2'])
        Evac.MG_strlog("\n   Strain offset (optical) about %.5f"%(linstrainos_opt),log_mg,output_lvl,printopt=False)
    
        messu_FP['Force']=messu_FP['Stress']/Bend.stress_perF(0.0,func_I,func_t,Bo_Le,Bo_Ri)
        messu_FP['Way']=(messu_FP['Strain']*Length**2)/(6*prot_ser['thickness_2'])
        if _opts['OPT_DIC']: messu_FP[dic_used_Disp]=(messu_FP[dic_used_Strain]*Length**2)/(6*prot_ser['thickness_2'])
    else:
        messu_FP =  messu
        linstrainos_con = 0
        linstrainos_opt = 0
        Evac.MG_strlog("\n   No linear start of final stress-strain-curve",log_mg,output_lvl,printopt=False)
        
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
    out_tab['geo_MoI_max'] = func_I(xlin).max()
    out_tab['geo_MoI_min'] = func_I(xlin).min()
    out_tab['geo_MoI_mid'] = func_I(0.0)
    if _opts['OPT_DIC']:
        out_tab['geo_curve_max'] = geo_d2_max
        out_tab['geo_curve_min'] = geo_d2_min
        out_tab['geo_curve_mid'] = geo_d2_mid
        
        t=bl['w_I']['d0'](np.array([Bo_Le,Bo_Ri]),
                          Pre_fit_df.loc(axis=1)['Fit_params_dict'])
        steps = Evac.pd_combine_index(messu.loc[VIP_messu['S']:VIP_messu['U']],
                                      Pre_fit_df)[[0,-1]]
        
        # tl, tr = t.loc[[VIP_messu['S']+1,VIP_messu['U']]].diff().iloc[-1].values
        tl, tr = t.loc[steps].diff().iloc[-1].values
        out_tab['ind_U_l'],       out_tab['ind_U_r'] = tl, tr
        tl, tr = t.loc[sf_eva_con[[0,-1]]].diff().iloc[-1].values
        out_tab['ind_Fcon_l'], out_tab['ind_Fcon_r'] = tl, tr
        tl, tr = t.loc[sf_eva_dic[[0,-1]]].diff().iloc[-1].values
        out_tab['ind_Fopt_l'], out_tab['ind_Fopt_r'] = tl, tr
        tl, tr = t.loc[sr_eva_con[[0,-1]]].diff().iloc[-1].values
        out_tab['ind_Rcon_l'], out_tab['ind_Rcon_r'] = tl, tr
        tl, tr = t.loc[sr_eva_dic[[0,-1]]].diff().iloc[-1].values
        out_tab['ind_Ropt_l'], out_tab['ind_Ropt_r'] = tl, tr
    
    out_tab['leos_con']   = linstrainos_con
    if _opts['OPT_DIC']: out_tab['leos_opt']   = linstrainos_opt

    relVS = [*np.sort([*strain_osd.keys()]),'U','B']
    tmp=Evac.Otvalgetter_Multi(messu_FP, Vs=relVS,
                          datasep=['con','opt'], 
                          VIPs={'con':VIP_messu,'opt':VIP_dicu}, 
                          exacts={'con':yield_df_con,'opt':yield_df_opt}, 
                          orders={'con':['f','e','U'],'opt':['e','U']}, 
                          add_esufs={'con':'_con','opt':'_opt'},
                          n_strains={'con':'Strain','opt':dic_used_Strain}, 
                          n_stresss={'con':'Stress','opt':'Stress'},
                          use_exacts=True)
    tmp.name=prot_ser.name
    out_tab=out_tab.append(tmp)
    tmp=Evac.Otvalgetter_Multi(messu_FP, Vs=relVS,
                          datasep=['con','opt'], 
                          VIPs={'con':VIP_messu,'opt':VIP_dicu}, 
                          exacts={'con':yield_df_con,'opt':yield_df_opt}, 
                          orders={'con':['F','s','W'],'opt':['s','W']}, 
                          add_esufs={'con':'_con','opt':'_opt'},
                          n_strains={'con':'Way','opt': dic_used_Disp}, 
                          n_stresss={'con':'Force','opt':'Force'},
                          use_exacts=True)
    tmp.name=prot_ser.name
    out_tab=out_tab.append(tmp)
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
        # ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
        #              xytext=(5, -5), textcoords='offset points')
    ax1.tick_params(axis='y', labelcolor=color1)
    #ax1.xticks([x for x in range(100) if x%10==0])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.grid(False)
    color2 = 'tab:blue'
    ax2.set_ylabel('Rise / curve /  (N/s)', color=color2)
    ax2.plot(messu.Time, messu.driF, 'b:', label='Force-rise')
    ax2.plot(messu.Time, messu.dcuF, 'g:', label='Force-curve')
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.legend(loc='lower right', bbox_to_anchor=(0.85, 0.15))
    Evac.plt_handle_suffix(fig,path=out_full+"-Fdricu",**plt_Fig_dict)
    
    if _opts['OPT_DIC']:
        fig, ax1 = plt.subplots()
        ax1.set_title('%s - Stress vs. strain curve of different opt. strain calculations'%plt_name)
        ax1.set_xlabel('Strain / -')
        ax1.set_ylabel('Stress / MPa')
        ax1.plot(messu.loc[:VIP_messu['B']]['Strain_opt_d_A'], messu.loc[:VIP_messu['B']]['Stress'], 'r:',label='displacement A')
        ax1.plot(messu.loc[:VIP_messu['B']]['Strain_opt_d_S'], messu.loc[:VIP_messu['B']]['Stress'], 'b:',label='displacement S')
        ax1.plot(messu.loc[:VIP_messu['B']]['Strain_opt_d_M'], messu.loc[:VIP_messu['B']]['Stress'], 'g:',label='displacement M')
        ax1.plot(messu.loc[:VIP_messu['B']]['Strain_opt_c_A'], messu.loc[:VIP_messu['B']]['Stress'], 'r--',label='curvature A')
        ax1.plot(messu.loc[:VIP_messu['B']]['Strain_opt_c_S'], messu.loc[:VIP_messu['B']]['Stress'], 'b--',label='curvature S')
        ax1.plot(messu.loc[:VIP_messu['B']]['Strain_opt_c_M'], messu.loc[:VIP_messu['B']]['Stress'], 'g--',label='curvature M')
        fig.legend(loc=[0.45, 0.135])
        # ftxt=('$E_{curve}$ = % 8.3f MPa '%(ED2_vgl1),
        #       '$E_{fork-C}$ = % 8.3f MPa '%(ED2_vgl2),
        #       '$E_{fork-P}$ = % 8.3f MPa '%(ED2_vgl3),
        #       '$E_{curve-P4O}$ = % 8.3f MPa '%(ED2_vgl4))
        # fig.text(0.97,0.15,'\n'.join(ftxt),
        #          ha='right',va='bottom', bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
        Evac.plt_handle_suffix(fig,path=out_full+"-sigeps_dicvgl",**plt_Fig_dict)
    
            # Pre-Fit -----------------------------------------------------------------
        plot_range_dic=messu.loc[VIP_dicu[(VIP_dicu[['F1','F3']]).idxmin()]:VIP_dicu[(VIP_dicu[['F2','F4']]).idxmax()]].index
        cbtick=VIP_dicu.to_dict()
        Bend.Plotter.colplt_funcs_all(x=xlin, func_cohort=bl['w_A'],
                                    params=Pre_fit_df.loc(axis=1)['Fit_params_dict'],
                                    step_range=plot_range_dic, 
                                    title=('%s - Fit-full - evaluation range'%plt_name),
                                    xlabel='x / mm',
                                    Point_df=P_xcoord_ydisp_meas,
                                    savefig=True,
                                    savefigname=out_full+'-DIC_fit-A-eva',
                                    cblabel='VIP', cbtick=cbtick)
        # Bending-Fit -------------------------------------------------------------
        Bend.Plotter.colplt_funcs_all(x=xlin, func_cohort=bl['w_M'],
                                    params=Bend_fit_df.loc(axis=1)['Fit_params_dict'],
                                    step_range=plot_range_dic, 
                                    title=('%s - Fit-Bending - evaluation range'%plt_name),
                                    xlabel='x / mm',
                                    Point_df=P_xcoord_ydisp_meas_M,
                                    savefig=True,
                                    savefigname=out_full+'-DIC_fit-M-eva',
                                    cblabel='VIP', cbtick=cbtick)
        # Pre-Fit -----------------------------------------------------------------
        Bend.Plotter.colplt_funcs_all(x=xlin, func_cohort=bl['w_A'],
                                    params=Pre_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                    step_range=plot_range_dic, 
                                    title=('%s - Incremental Fit-full - evaluation range'%plt_name),
                                    xlabel='x / mm',
                                    Point_df=P_xcoord_ydiff_meas,
                                    savefig=True,
                                    savefigname=out_full+'-INC_fit-A-eva',
                                         cblabel='VIP', cbtick=cbtick)
        # Bending-Fit -------------------------------------------------------------
        Bend.Plotter.colplt_funcs_all(x=xlin, func_cohort=bl['w_M'],
                                    params=Bend_inc_fit_df.loc(axis=1)['Fit_params_dict'],
                                    step_range=plot_range_dic, 
                                    title=('%s - Incremental Fit-Bending - evaluation range'%plt_name),
                                    xlabel='x / mm',
                                    Point_df=P_xcoord_ydiff_meas_M,
                                    savefig=True,
                                    savefigname=out_full+'-INC_fit-M-eva',
                                     cblabel='VIP', cbtick=cbtick)
    
    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Stress vs. strain curve with labels'%plt_name)
    ax1.set_xlabel('Strain / -')
    ax1.set_ylabel('Stress / MPa')
    ax1.plot(messu.loc[:VIP_messu['B']]['Strain'], messu.loc[:VIP_messu['B']]['Stress'], 'r--',label='con')
    if _opts['OPT_DIC']:
        ax1.plot(messu.loc[:VIP_messu['B']][dic_used_Strain], messu.loc[:VIP_messu['B']]['Stress'], 'm--',label='opt')
    t=messu.Strain.loc[[VIP_messu['F3'] , VIP_messu['F4']]].values
    t=np.array([max(0,t[0]-0.1*(t[1]-t[0])),min(messu.Strain.max(),t[1]+0.1*(t[1]-t[0]))])
    # ax1.plot(t, np.polyval(E2_pf_tmp[0][:],t), 'g-',label='$E_{con}$')
    # ax1.plot(t, E2_fit.eval(x=t), 'g-',label='$E_{con}$')
    ax1.plot(t, YM_pref_con['E']*t+YM_pref_con['E_abs'],
             'g-',label='$E_{con}$')
    a, b=messu.Strain[VIP_messu[:'B']],messu.Stress[VIP_messu[:'B']]
    j=np.int64(-1)
    ax1.plot(a, b, 'bx')
    for x in VIP_messu[:'B'].index:
        j+=1
        if j%2: c=(6,-6)
        else:   c=(-6,6)
        ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                      xytext=c, ha="center", va="center", textcoords='offset points')
    if _opts['OPT_DIC']:
        a,b=messu.loc[VIP_dicu[:'B'],dic_used_Strain],messu.Stress[VIP_dicu[:'B']]
        j=np.int64(-1)
        ax1.plot(a, b, 'yx')
        t=messu.loc[[VIP_dicu['F3'] , VIP_dicu['F4']],dic_used_Strain].values
        t=np.array([max(0,t[0]-0.1*(t[1]-t[0])),min(messu.loc[:,dic_used_Strain].max(),t[1]+0.1*(t[1]-t[0]))])
        # ax1.plot(t, np.polyval(ED2_pf_tmp[0][:],t), 'b-',label='$E_{opt}$')
        # ax1.plot(t, ED2_fit.eval(x=t), 'b-',label='$E_{opt}$')
        ax1.plot(t, YM_pref_opt['E']*t+YM_pref_opt['E_abs'],
             'b-',label='$E_{opt}$')
    ftxt=('$f_{y}$ = % 6.3f MPa ($\epsilon_{y}$ = %4.3f %%)'%(out_tab['fy'],out_tab['ey_opt']*100),
          '$f_{u}$ = % 6.3f MPa ($\epsilon_{u}$ = %4.3f %%)'%(out_tab['fu'],out_tab['eu_opt']*100),
          '$E_{con}$ = % 8.3f MPa ($R²$ = %4.3f)'%(*YM_pref_con[['E','Rquad']],),
          '$E_{opt}$ = % 8.3f MPa ($R²$ = %4.3f)'%(*YM_pref_opt[['E','Rquad']],))
    fig.text(0.95,0.15,'\n'.join(ftxt),
              ha='right',va='bottom', bbox=dict(boxstyle='round', edgecolor='0.8', facecolor='white', alpha=0.8))
    fig.legend(loc='upper left', bbox_to_anchor=(0.10, 0.91))
    Evac.plt_handle_suffix(fig,path=out_full+"-sigeps_wl",**plt_Fig_dict)
    
    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Stress vs. strain curve, final part, with labels'%plt_name)
    ax1.set_xlabel('Strain [-]')
    ax1.set_ylabel('Stress [MPa]')
    ax1.plot(messu_FP['Strain'], messu_FP['Stress'], 'r--', label='meas. curve')
    if _opts['OPT_DIC']:
        ax1.plot(messu_FP.loc[:VIP_messu['B']][dic_used_Strain], messu_FP.loc[:VIP_messu['B']]['Stress'], 'm--',label='opt')
    a,b = Evac.stress_linfit_plt(messu_FP['Strain'], VIP_messu[['F3','F4']],
                                 YM_pref_con['E'],0)
    ax1.plot(a, b, 'g-',label='$E_{con}$')
    tmp=VIP_messu[['S','F3','F4','Y','Y0','Y1','U']].sort_values()
    if 'B' in VIP_messu.index: tmp=tmp.append(VIP_messu[['B']])
    tmp=tmp.append(VIP_messu[['E']])
    tmp=tmp.sort_values()
    a, b=messu_FP.Strain[tmp],messu_FP.Stress[tmp]
    j=np.int64(-1)
    ax1.plot(a, b, 'bx')
    for x in tmp.index:
        j+=1
        if j%2: c=(6,-6)
        else:   c=(-6,6)
        ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data', 
                     xytext=c, ha="center", va="center", textcoords='offset points')
    if _opts['OPT_DIC']:
        tmp=VIP_dicu[['S','F3',*np.sort([*strain_osd.keys()]),'U']].sort_values()
        if 'B' in VIP_dicu.index: tmp=tmp.append(VIP_dicu[['B']])
        tmp=tmp.append(VIP_dicu[['E']])
        tmp=tmp.sort_values()
        a,b=messu_FP.loc[tmp,dic_used_Strain],messu_FP.Stress[tmp]
        j=np.int64(-1)
        ax1.plot(a, b, 'yx')
        a,b = Evac.stress_linfit_plt(messu_FP[dic_used_Strain], VIP_messu[['F3','F4']],
                                     YM_pref_opt['E'],0)
        ax1.plot(a, b, 'b-',label='$E_{opt}$')
    fig.legend()
    Evac.plt_handle_suffix(fig,path=out_full+"-sigeps_fin",**plt_Fig_dict)
    
    mun_tmp = messu.loc[:VIP_messu['U']]
    fig, ax1 = plt.subplots()
    ax1.set_title('%s - Stress vs. strain curve comparison to selected YM-Methods'%plt_name)
    ax1.set_xlabel('Strain / -')
    ax1.set_ylabel('Stress / MPa')
    ax1.set_xlim(0,mun_tmp['Strain'].max())
    ax1.plot(mun_tmp['Strain'], mun_tmp['Stress'], 'r--',label='con')
    a, b=messu.Strain[VIP_messu[:'U']],messu.Stress[VIP_messu[:'U']]
    j=np.int64(-1)
    ax1.plot(a, b, 'bx')
    for x in VIP_messu[:'U'].index:
        j+=1
        if j%2: c=(6,-6)
        else:   c=(-6,6)
        ax1.annotate('%s' % x, xy=(a.iloc[j],b.iloc[j]), xycoords='data',
                      xytext=c, ha="center", va="center", textcoords='offset points')
    ax1.plot(mun_tmp[dic_used_Strain], mun_tmp['Stress'],'m--',
             label='opt %s'%_opts['OPT_YM_Determination_refinement'][3])
    a,b=messu.loc[VIP_dicu[:'U'],dic_used_Strain],messu.Stress[VIP_dicu[:'U']]
    ax1.plot(a, b, 'yx')    
    t = mun_tmp.Stress / E_Methods_df.loc[mun_tmp.index[1:]]['A0Al']
    tra = Evac.Rquad(mun_tmp.Strain, t)
    trf = Evac.Rquad(mun_tmp.Strain.loc[sf_eva_con], t.loc[sf_eva_con])
    trr = Evac.Rquad(mun_tmp.Strain.loc[sr_eva_con], t.loc[sr_eva_con])
    ax1.plot(t, mun_tmp.loc[t.index]['Stress'], 'r:',
             label='A0Al %1.4f|%1.4f|%1.4f'%(tra,trf,trr))
    t = mun_tmp.Stress / E_Methods_df.loc[mun_tmp.index[1:]]['B2Ml']
    tra = Evac.Rquad(mun_tmp[dic_used_Strain], t)
    trf = Evac.Rquad(mun_tmp[dic_used_Strain].loc[sf_eva_dic], t.loc[sf_eva_dic])
    trr = Evac.Rquad(mun_tmp[dic_used_Strain].loc[sr_eva_dic], t.loc[sr_eva_dic])
    ax1.plot(t, mun_tmp.loc[t.index]['Stress'], 'y:',
             label='B2Ml %1.4f|%1.4f|%1.4f'%(tra,trf,trr))
    t = mun_tmp.Stress / E_Methods_df.loc[mun_tmp.index[1:]]['E4Mg']
    tra = Evac.Rquad(mun_tmp[dic_used_Strain], t)
    trf = Evac.Rquad(mun_tmp[dic_used_Strain].loc[sf_eva_dic], t.loc[sf_eva_dic])
    trr = Evac.Rquad(mun_tmp[dic_used_Strain].loc[sr_eva_dic], t.loc[sr_eva_dic])
    ax1.plot(t, mun_tmp.loc[t.index]['Stress'], 'm:',
             label='E4Mg %1.4f|%1.4f|%1.4f'%(tra,trf,trr))
    t = mun_tmp.Stress / E_Methods_df.loc[mun_tmp.index[1:]]['G3Mg']
    tra = Evac.Rquad(mun_tmp[dic_used_Strain], t)
    trf = Evac.Rquad(mun_tmp[dic_used_Strain].loc[sf_eva_dic], t.loc[sf_eva_dic])
    trr = Evac.Rquad(mun_tmp[dic_used_Strain].loc[sr_eva_dic], t.loc[sr_eva_dic])
    ax1.plot(t, mun_tmp.loc[t.index]['Stress'], 'b:',
             label='G3Mg %1.4f|%1.4f|%1.4f'%(tra,trf,trr))
    t = mun_tmp.Stress / E_Methods_df.loc[mun_tmp.index[1:]]['D2Mgwt']
    tra = Evac.Rquad(mun_tmp[dic_used_Strain], t)
    trf = Evac.Rquad(mun_tmp[dic_used_Strain].loc[sf_eva_dic], t.loc[sf_eva_dic])
    trr = Evac.Rquad(mun_tmp[dic_used_Strain].loc[sr_eva_dic], t.loc[sr_eva_dic])
    ax1.plot(t, mun_tmp.loc[t.index]['Stress'], 'g:',
             label='D2Mgwt %1.4f|%1.4f|%1.4f'%(tra,trf,trr))
    ax1.legend()
    Evac.plt_handle_suffix(fig,path=out_full+"-sigeps_YMcomp",**plt_Fig_dict)

    # =============================================================================
    #%%% 7.3 Generate outputs
    timings.loc[7.3]=time.perf_counter()
    Evac.MG_strlog("\n   Timing %f: %.5f s"%(timings.index[-1],
                                       timings.iloc[-1]-timings.iloc[0]),
                   log_mg,output_lvl,printopt=False)    
    
    
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
    
    t = pd.concat([Check_to_D['AtoD_g'],Check_to_D['StoD_g'],
                   Check_to_D['MtoD_g']],axis=1).mean()
    t = t.add_prefix('Check_g_')
    t.name=prot_ser.name
    t2 = pd.concat([Check_to_D['AtoD_x'],Check_to_D['StoD_x'],
                    Check_to_D['MtoD_x']],axis=1).mean()
    t2 = t2.add_prefix('Check_l_')
    t2.name=prot_ser.name
    t = pd.concat([t,t2])
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
    HDFst['Measurement_FP'] = messu_FP
    HDFst['Material_Parameters'] = out_tab
    HDFst['Geometrie_functions'] = pd.Series({'func_t': func_t,'func_w': func_w, 'func_A': func_A, 'func_I': func_I})
    HDFst['Timings'] = timings
    
    # HDFst['Bending_Legion'] = bl
    HDFst['Opt_Geo_Fit'] = geo_fit
    HDFst['Opt_Pre_Fit'] = Pre_fit_df
    HDFst['Opt_Bend_Fit'] = Bend_fit_df
    HDFst['Inc_Pre_Fit'] = Pre_inc_fit_df
    HDFst['Inc_Bend_Fit'] = Bend_inc_fit_df
    
    HDFst['Points_transformed'] = Points_T_stacked
    HDFst['Points_lot_transformed'] = Points_L_T_stacked
    HDFst['Points_A_disp'] = P_xcoord_ydisp
    HDFst['Points_S_disp'] = P_xcoord_ydisp_meas_S
    HDFst['Points_M_disp'] = P_xcoord_ydisp_meas_M
    HDFst['Points_A_diff'] = P_xcoord_ydiff
    HDFst['Points_S_diff'] = P_xcoord_ydiff_meas_S
    HDFst['Points_M_diff'] = P_xcoord_ydiff_meas_M
    
    HDFst['VIP'] = pd.concat([VIP_messu,VIP_dicu],axis=1)
    
    HDFst['Exact_VIP_con']=yield_df_con
    HDFst['Exact_VIP_opt']=yield_df_opt
    
    HDFst['DQcon'] = DQcon
    HDFst['DQopt'] = DQopt
    
    HDFst['E_lsq'] = E_lsq
    HDFst['E_inc_df'] = E_Methods_df
    HDFst['E_inc'] = pd.concat([E_inc_F_comp.add_prefix('E_inc_F_'),
                                E_inc_R_comp.add_prefix('E_inc_R_')],axis=1)
    HDFst['Check_to_D'] = Check_to_D
    
    HDFst.close()

    
    timings.loc[10.0]=time.perf_counter()
    if output_lvl>=1: log_mg.close()
    
    return timings
    
#%% 9 Main
def TBT_series(paths, no_stats_fc, var_suffix):
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
                timings = TBT_single(prot_ser = prot.loc[eva],
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
       
    option = 'single'
    # option = 'series'
    # option = 'complete'
    # option = 'pack-complete'
    # option = 'pack-complete-all'
    
    no_stats_fc = ['A01.1','A01.2','A01.3', 'A02.3',
                   'B01.1','B01.2','B01.3', 'B02.3',
                   'C01.1','C01.2','C01.3', 'C02.3',
                   'D01.1','D01.2','D01.3', 'D02.3',
                   'F01.1','F01.2','F01.3', 'F02.3']
                   # 'G01.1','G01.2','G01.3', 'G02.3']
    # var_suffix = ["A","B","C","D"] #Suffix of variants of measurements (p.E. diffferent moistures)
    var_suffix = [""] #Suffix of variants of measurements (p.E. diffferent moistures)
        
    protpaths = pd.DataFrame([],dtype='string')
    combpaths = pd.DataFrame([],dtype='string')
    # path_main="F:/Messung/003-190822-Becken1-DBV/"
    # name_protokoll="190822_Becken1_DBV_Protokoll.xlsx"
    # path_main="F:/Messung/004-191129-Schulter-DBV/"
    # name_protokoll="191129_Schulter_DBV_Protokoll.xlsx"
    # path_main="D:/Gebhardt/Messungen/200226_Test_FU_Holz_A/"
    # name_protokoll="200226_Test_FU_Holz_Protokoll.xlsx"
    # path_main="D:/Gebhardt/Messungen/200401_Test_FU_Weide/"
    # name_protokoll="200401_Test_FU_Weide_Protokoll_new.xlsx"
    # protpaths.loc['B2','path_main'] = "F:/Messung/004-200515-Becken2-DBV/"
    # protpaths.loc['B2','name_prot'] = "200515_Becken2_DBV_Protokoll_new.xlsx"
    protpaths.loc['B3','path_main'] = "F:/Messung/005-200724_Becken3-DBV/"
    protpaths.loc['B3','name_prot'] = "200724_Becken3-DBV_Protokoll_new.xlsx"
    protpaths.loc['B4','path_main'] = "F:/Messung/006-200917_Becken4-DBV/"
    protpaths.loc['B4','name_prot'] = "200917_Becken4-DBV_Protokoll_new.xlsx"
    protpaths.loc['B5','path_main'] = "F:/Messung/007-201014_Becken5-DBV/"
    protpaths.loc['B5','name_prot'] = "201014_Becken5-DBV_Protokoll_new.xlsx"
    protpaths.loc['B6','path_main'] = "F:/Messung/008-201125_Becken6-DBV/"
    protpaths.loc['B6','name_prot'] = "201125_Becken6-DBV_Protokoll_new.xlsx"
    protpaths.loc['B7','path_main'] = "F:/Messung/009-210120_Becken7-DBV/"
    protpaths.loc['B7','name_prot'] = "210120_Becken7-DBV_Protokoll_new.xlsx"

    protpaths.loc[:,'path_con']     = "Messdaten/Messkurven/"
    protpaths.loc[:,'path_dic']     = "Messdaten/DIC/"
    protpaths.loc[:,'path_eva1']    = "Auswertung/"
    # protpaths.loc[:,'path_eva2']    = "Test_py/"
    protpaths.loc[:,'path_eva2']    = "ExMechEva/"
    
    # t=protpaths[['path_main','path_eva1','name_prot']].stack()
    # t.groupby(level=0).apply(lambda x: '{0}{1}{2}'.format(*x))
    combpaths['prot'] = protpaths['path_main']+protpaths['path_eva1']+protpaths['name_prot']
    combpaths['meas'] = protpaths['path_main']+protpaths['path_con']
    combpaths['dic']  = protpaths['path_main']+protpaths['path_dic']
    combpaths['out']  = protpaths['path_main']+protpaths['path_eva1']+protpaths['path_eva2']
    
    # mfile_add="" # suffix für Feuchte
    
    
    if option == 'single':
        ser='B7'
        des='cl11a'
        mfile_add = var_suffix[0] #Suffix of variants of measurements (p.E. diffferent moistures)
        
        prot=pd.read_excel(combpaths.loc[ser,'prot'],
                           header=11, skiprows=range(12,13),
                           index_col=0)
        _=TBT_single(prot_ser=prot[prot.Designation==des].iloc[0], 
                     paths=combpaths.loc[ser],
                     mfile_add = mfile_add)
        
    elif option == 'series':
        ser='B7'
        TBT_series(paths = combpaths.loc[ser],
                   no_stats_fc = no_stats_fc,
                   var_suffix = var_suffix)
        
    elif option == 'complete':
        for ser in combpaths.index:
            TBT_series(paths = combpaths.loc[ser],
                       no_stats_fc = no_stats_fc,
                       var_suffix = var_suffix)
            
    elif option == 'pack-complete':        
        out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/XXX/TBT/B3-B7_TBT-Summary"
        # out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/230919/TBT/B3-B7_TBT-Summary"
        # out_path="D:/Gebhardt/Spezial/DBV/Methodenvgl/211217/B3-B7_TBT-Summary"
        packpaths = combpaths[['prot','out']]
        packpaths.columns=packpaths.columns.str.replace('out','hdf')
        Evac.pack_hdf(in_paths=packpaths, out_path = out_path,
                      hdf_naming = 'Designation', var_suffix = var_suffix,
                      h5_conc = 'Material_Parameters', h5_data = 'Measurement',
                      opt_pd_out = False, opt_hdf_save = True)
        
    elif option == 'pack-complete-all':        
        out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/XXX/TBT/B3-B7_TBT-Summary-all"
        # out_path="D:/Gebhardt/Projekte/001_PARAFEMM/Auswertung/211206/TBT/B3-B7_TBT-Summary"
        # out_path="D:/Gebhardt/Spezial/DBV/Methodenvgl/220905/B3-B7_TBT-Summary_new2"
        packpaths = combpaths[['prot','out']]
        packpaths.columns=packpaths.columns.str.replace('out','hdf')
        Evac.pack_hdf_mul(in_paths=packpaths, out_path = out_path,
                          hdf_naming = 'Designation', var_suffix = var_suffix,
                          h5_conc = 'Material_Parameters',
                          opt_pd_out = False, opt_hdf_save = True)
        
    else:
        raise NotImplementedError('%s not implemented!'%option)
        

if __name__ == "__main__":
    main()
