# -*- coding: utf-8 -*-
"""
Testing elastic modulus determination by generic three-point-bending test data.

@author: MarcGebhardt
"""
#%% Import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
nwd = Path.cwd().resolve().parent
os.chdir(nwd)

import exmecheva.common as emec
import exmecheva.bending as emeb

#%% Input and selection
length = 20 # span between supports in mm
poisson = 0.3 # poisson ratio
elastic_modulus = 5000 # elastic modulus
force = pd.Series([0,1,25,35,40,50]) # force series

n_of_ps=11 # number of points in generic displ. data
det_n_disp=100 # number of points for displ. data generation (x must match n_of_ps)
det_n_metD=100 #n umber of points (numerical integration steps) for Method D
xlin = np.linspace(-length/2,length/2,det_n_disp+1)

sample=4 # change sample number to try other geometry type
if sample == 1:
    # uniform type
    thick_l=[2, 2, 2]
    width_l=[2, 2, 2]
elif sample == 2:
    # trapezoid type 1 (thickness variation)
    thick_l=[1, 2, 3]
    width_l=[2, 2, 2]
elif sample == 3:
    # trapezoid type 2 (width variation)
    thick_l=[2, 2, 2]
    width_l=[1, 2, 3]
elif sample == 4:
    # trapezoid type 3  (thickness and width variation (synchronous))
    thick_l=[1, 2, 3]
    width_l=[2, 4, 6]
elif sample == 5:
    # trapezoid type 4  (thickness and width variation (asynchronous))
    thick_l=[1, 2, 3]
    width_l=[6, 4, 2]
elif sample == 6:
    # parabol type 1 (symetric)
    thick_l=[1, 3, 1]
    width_l=[2, 2, 2]
elif sample == 7:
    # parabol type 2 (asymetric thickness)
    thick_l=[1, 3, 2]
    width_l=[2, 2, 2]

#%% Geometry evaluation
Bo_Le = -length/2
Bo_Ri = length/2
geo = pd.DataFrame({'x': [Bo_Le, 0, Bo_Ri],
                    't': thick_l,
                    'w': width_l})
func_t = np.poly1d(np.polyfit(geo.loc[:,'x'], geo.loc[:,'t'],2), variable='x') # Polynom 2'ter Ordnung für Dicke über x-Koordinate
func_w = np.poly1d(np.polyfit(geo.loc[:,'x'], geo.loc[:,'w'],2), variable='x') # Polynom 2'ter Ordnung für Breite über x-Koordinate
func_A = func_t * func_w # Polynom für Querschnitt über x-Koordinate
func_I = func_t**3*func_w/12 # Polynom für Flächenträgheitsmoment 2'ten Grades über x-Koordinate
if True:
    fig, (ax1,ax3) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, 
                                  figsize = np.multiply(
                                      [16.0/2.54, 9.0/2.54],[1,2]
                                      ))
    ax1.set_title('Width and Thickness')
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
    ax3.set_title('Moment of Inertia')
    ax3.set_xlabel('x / mm')
    ax3.set_ylabel('MoI / mm^4')
    ax3.plot(xlin, func_I(xlin), 'g-', label = 'MoI-fit')
    ax3.plot(geo.loc[:,'x'],geo.loc[:,'t']**3*geo.loc[:,'w']/12, 
             'gh', label = 'MoI')
    ax3.legend()
    emec.plotting.plt_handle_suffix(fig,None)
    
print('Geometry values in midspan:')
print('  - t = {:.3f}'.format(func_t(0)))
print('  - w = {:.3f}'.format(func_w(0)))
print('  - A = {:.3f}'.format(func_A(0)))
print('  - I = {:.3f}'.format(func_I(0)))

# standard displacment in midspan (uniform cs)
w_ms=-force[1]*length**3/(48*elastic_modulus*func_I(0))

#%% generic optical displacment data
d_1_f=emeb.evaluation.YM_eva_method_D_bend_df(
    Length=length, I_func=func_I, n=det_n_disp, E=elastic_modulus,F=1
    )
    
d_1=d_1_f.loc[d_1_f.index.to_series().apply(
    lambda x: x in np.linspace(-length/2,length/2,n_of_ps)
    )]
d_1=d_1['w'].reset_index()
d_1.columns=['x','y']
d_1.index=d_1.index.to_series().apply(lambda x: 'P{}'.format(x+1))
d_1 = d_1.stack()
name_ms=d_1.loc[:,'x'].loc[d_1.loc[:,'x']==0].index[0]
print("Displacement in midspan w(x=0):")
print("  - Standard:                        {:.5f}".format(w_ms))
print("  - Theoretical elastic curve (mid): {:.5f}".format(d_1[name_ms,'y']))
print("  -> Deviation: {:.5f}".format((d_1[name_ms,'y']-w_ms)/w_ms))
print("Displacement absolute maximum:       {:.5f}".format(d_1_f.w.min()))
if True:
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize = [16.0/2.54, 9.0/2.54])
    ax1.set_title('Generic elastic curve')
    ax1.set_xlabel('x / mm')
    ax1.set_ylabel('Displacement / mm')
    ax1.plot(d_1_f.index, d_1_f['w'], 'b-', label='Curve')
    ax1.plot(d_1[:,'x'], d_1[:,'y'], 'ko', label='Points')
    d_1.unstack().apply(lambda x: ax1.annotate(
        x.name, xy=[x['x'],x['y']], xycoords='data', xytext=(0,-10), 
        ha="center", va="center", textcoords='offset points'
        ), axis=1)
    d_1_min=d_1_f['w'].min(),d_1_f['w'].idxmin()
    ax1.annotate(
        '{0:.5f} (x={1:.3f})'.format(*d_1_min), 
        xy=(d_1_min[1], d_1_min[0]), xytext=(0, 80),
        ha="center", textcoords='offset points',
        arrowprops=dict(facecolor='red', shrink=0.05)
        )
    ax1.grid(True)
    ax1.legend()
    emec.plotting.plt_handle_suffix(fig,None)

xc_yd=pd.DataFrame([d_1,]*len(force.index), index=force.index)
yd=emeb.opt_mps.Point_df_idx(xc_yd,coords='y').mul(force,axis=0)
xc_yd= emeb.opt_mps.Point_df_combine(xc_yd,yd,coords_set='y')
xc_yd_diff=xc_yd.copy()
xc_yd_diff.loc(axis=1)[:,'y'] = xc_yd_diff.loc(axis=1)[:,'y'].diff()

#%% generic measurement data
mess = pd.DataFrame({'force':force})
mess['disp']=emeb.opt_mps.Point_df_idx(xc_yd,points=name_ms,coords='y')*(-1)
mess['stress_ms']=emeb.evaluation.stress_perF(
    0.0, func_I, func_t, Bo_Le, Bo_Ri
    )*mess.force
mess['strain_ms']=6*thick_l[1]*mess.disp/length**2
mess['force_d']=mess['force'].diff()
mess['stress_ms_d']=mess['stress_ms'].diff()
mess['strain_ms_d']=mess['strain_ms'].diff()

#%% Elastic modulus determination methods
#%%% Method A
A0Al_ser = emeb.evaluation.YM_eva_method_A(
    stress_mid_ser=mess['stress_ms_d'].iloc[1:],
    strain_mid_ser=mess['strain_ms_d'].iloc[1:],
    comp=True,
    name='A0Al', 
    det_opt='incremental'
    )
print('Elastic modulus determination methods:')
print('  - A0Al (incremental):   {:.3f}'.format(A0Al_ser.mean()))

E_lsq_F_A0Al = emeb.evaluation.YM_eva_method_A(
    stress_mid_ser=mess['stress_ms'],
    strain_mid_ser=mess['strain_ms'],
    comp=True,
    name='E_lsq_F_A0Al', 
    det_opt='leastsq'
    )
E_lsq_F_A0Al = pd.Series(
    E_lsq_F_A0Al, index=['E','E_abs','Rquad','Fit_result'], 
    name='E_lsq_F_A0Al'
    )
print('  - A0Al (least-square):  {:.3f}'.format(E_lsq_F_A0Al[0]))

#%%% Method B
# Getting point names for Method B
nms_s, nms_i='',''
for i in name_ms:
    try:
        _=int(i)
    except ValueError:
        nms_s+=i
    else:
        nms_i+=i
name_fp=[nms_s+str(int(nms_i)-1),name_ms,nms_s+str(int(nms_i)+1)]
B1Ml_ser, B1Ml_strain_ser = emeb.evaluation.YM_eva_method_B(
    option="Points",
    stress_mid_ser=mess['stress_ms_d'],
    thickness = func_t(0.0), 
    Length = length,
    P_df = xc_yd_diff, 
    P_fork_names = name_fp,
    comp=True,
    name='B1Ml'
    )
print('  - B1Ml (incremental):   {:.3f}'.format(B1Ml_ser.mean()))

E_lsq_F_B1Ml = emeb.evaluation.YM_eva_method_B(
    option="Points",
    stress_mid_ser=mess['stress_ms'],
    thickness = func_t(0.0),
    Length = length,
    P_df = xc_yd,
    P_fork_names = name_fp,
    comp=True,
    name='E_lsq_F_B1Ml',
    det_opt='leastsq'
    )
print('  - B1Ml (least-square):  {:.3f}'.format(E_lsq_F_B1Ml[0]))

#%%% Method D
D1Mguw,D1Mguw_df=emeb.evaluation.YM_eva_method_D(
    P_df=xc_yd_diff, Force_ser=mess['force_d'],
    rel_steps = mess.index[1:],
    Length=length, func_I=func_I, n=det_n_metD,
    weighted=True, weight_func=emeb.evaluation.Weight_func,
    wkwargs={'option':'Cut',
             'xmin':Bo_Le,'xmax':Bo_Ri},
    pb_b=False, name='D1Mguw'
    )
print('  - D1Mguw (incremental): {:.3f}'.format(D1Mguw.mean()))