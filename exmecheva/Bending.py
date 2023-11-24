# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:01:47 2021

@author: mgebhard
Changelog:
    - 22-06-13: colplt_funcs_ax: RangeIndex Keyerror umgangen mittels Evac.pd_limit
                -> auch bei colplt_df_ax ändern!
    - 22-10-26 - Methode E - unload (negative force increment) added
"""
import copy
import warnings
import numpy as np
import pandas as pd
import sympy as sy
import lmfit
import scipy.integrate as scint
import vg # entfernen!

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

import dill
from tqdm import tqdm

# import Eva_common
import exmecheva.Eva_common as Evac #import Eva_common relative?

#%%Global Variables
# set Parameter types, which are not marked as free for fit:
_param_types_not_free   = ['independent','expr','fixed','post']
_param_types_fit_or_set = ['free','post']

_significant_digits_fpd=12

#%% Measured Points
def v_length(v1):
    """
    Computes the length of an vector
    """
    length=np.sqrt(np.sum(v1**2))
    return length

def v_Ctrans (v,TM,P0):
    """
    Transform a set of vectors(v=ixN3) with a translationmatrix (TM) and a new origin (P0)
    """
    if type(v).__name__ == 'ndarray':
        vtrans=np.ndarray(v.shape)
        for i in range(v.shape[1]):
            vtrans[:,i]=np.dot(TM,(v[:,i]-P0))
    elif type(v).__name__ == 'DataFrame':
        vtrans=pd.DataFrame(index=v.index,columns=v.columns)
        for i in v.columns:
            vtrans.loc[:,i]=np.dot(TM,(v.loc[:,i]-P0))
    return vtrans

def Point_df_transform(Pmeas, Pspec, Pmeas_Sdev, Pspec_Sdev, 
                       dic_P_name_org1, dic_P_name_org2,
                       output_lvl=0, log_mg=''):
    """
    Calculate in plane transformation of given measured 3D-points.

    Parameters
    ----------
    Pmeas : pandas.Dataframe([],index=['x','y','z'],columns=['P1','P2',...],dtype=float64)
        Cartesian coordinates of measured points, which should be fitted.
    Pspec : pandas.Dataframe([],index=['x','y','z'],columns=['S1','S2',...],dtype=float64)
        Cartesian coordinates of special points which should used for coordinate transformation.
    Pmeas_Sdev : pandas.Dataframe([],index=['Sx','Sy','Sz'],columns=['P1','P2',...],dtype=float64)
        Standard deviation of measured points, which should be fitted.
    Pspec_Sdev : pandas.Dataframe([],index=['Sx','Sy','Sz'],columns=['S1','S2',...],dtype=float64)
        Standard deviation of special points which should used for coordinate transformation.
    dic_P_name_org1 : str
        Name of point, which should be on the negativ x-axis. Origin is in the middle between dic_P_name_org1 and dic_P_name_org2.
    dic_P_name_org2 : str
        Name of point, which should be on the positiv x-axis. Origin is in the middle between dic_P_name_org1 and dic_P_name_org2.


    Returns
    -------
     Points_T
         Transformed coordinates and standard deviation of given points.
     Points_L_T
         In plane transformed coordinates and standard deviation of given points.

    """
    # Plane-Fitting
    A = np.column_stack([Pspec.loc['x'], Pspec.loc['y'], np.ones(len(Pspec.loc['x']))]) # Ebene auf Punkte Versuchseinrichtung gelegt
    b = Pspec.loc['z']
    P_fit, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
    fitnorm=vg.normalize(np.array([P_fit[0],P_fit[1],-1]))
#    rquad=1 - res / (b.size * b.var()) # nur bei Pmeas als Eingang
    
    Pmeas_dist=(P_fit[0] * Pmeas.loc['x'] + P_fit[1] * Pmeas.loc['y'] - Pmeas.loc['z'] + P_fit[2])/v_length(np.array([P_fit[0],P_fit[1],-1]))
    Pspec_dist=(P_fit[0] * Pspec.loc['x'] + P_fit[1] * Pspec.loc['y'] - Pspec.loc['z'] + P_fit[2])/v_length(np.array([P_fit[0],P_fit[1],-1]))
    
    if output_lvl>=1:
        log_mg.write("\n    Fitting solution: %f x + %f y + %f = z" % (P_fit[0], P_fit[1], P_fit[2]))
#        print("R² = %.3f" %rquad) # nur bei Pmeas als Eingang
        log_mg.write("\n    Mean distance from meas. to spec. points = %.3e" %Pmeas_dist.mean())
    
    #Lotfußpunkte
    Pmeas_L=pd.DataFrame(data=np.zeros(Pmeas.shape),index=Pmeas.index,columns=Pmeas.columns)
    for i in range(Pmeas.shape[1]):
        Pmeas_L.iloc[:,i]=Pmeas.iloc[:,i]-Pmeas_dist[i]*fitnorm
    Pspec_L=pd.DataFrame(data=np.zeros(Pspec.shape),index=Pspec.index,columns=Pspec.columns)
    for i in range(Pspec.shape[1]):
        Pspec_L.iloc[:,i]=Pspec.iloc[:,i]-Pspec_dist[i]*fitnorm
    
    if output_lvl>=2:    
        # plot raw data
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(Pmeas.loc['x'], Pmeas.loc['y'],Pmeas.loc['z'], color='b',label='meas. points')
        ax.scatter(Pspec.loc['x'], Pspec.loc['y'],Pspec.loc['z'], color='m',label='spec. points')
        #plot Lotfußpunkte
        ax.scatter(Pmeas_L.loc['x'], Pmeas_L.loc['y'],Pmeas_L.loc['z'], color='c', marker="+",label='meas. lot points')
        ax.scatter(Pspec_L.loc['x'], Pspec_L.loc['y'],Pspec_L.loc['z'], color='r', marker="+",label='spec. lot points')
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([Pmeas_L.loc['x'].max()-Pmeas_L.loc['x'].min(), Pmeas.loc['y'].max()-Pmeas.loc['y'].min(), Pmeas.loc['z'].max()-Pmeas.loc['z'].min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(Pmeas_L.loc['x'].max()+Pmeas_L.loc['x'].min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Pmeas.loc['y'].max()+Pmeas.loc['y'].min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Pmeas.loc['z'].max()+Pmeas.loc['z'].min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        # plot plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
          np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
           for c in range(X.shape[1]):
               Z[r,c] = P_fit[0] * X[r,c] + P_fit[1] * Y[r,c] + P_fit[2]
        ax.plot_wireframe(X,Y,Z, color='g',label='fit plane')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        
    # Koordinatentransformation
    KT_Pzero=(Pspec_L.loc[:,dic_P_name_org1]+Pspec_L.loc[:,dic_P_name_org2])/2 # Koordinatenursprung auf Mitte org1 und org2 (Standard S1 und S2)
    KT_Pxdir=Pspec_L.loc[:,dic_P_name_org2] # Richtung x-Achse in org2 (Standard S2)
    KT_Vxdir=vg.normalize(KT_Pxdir-KT_Pzero)
    KT_Vzdir=fitnorm # z-Achse in Normalen-Richtung des Ebenen-fits 
    KT_Vydir=vg.perpendicular(KT_Vxdir,KT_Vzdir, normalized=True)
    KT_MTrans=np.array([KT_Vxdir,KT_Vydir,KT_Vzdir]) # Transformationsmatrix
        
    Pmeas_T     =v_Ctrans(Pmeas,    KT_MTrans,KT_Pzero)
    Pmeas_L_T   =v_Ctrans(Pmeas_L,  KT_MTrans,KT_Pzero)
    Pspec_T     =v_Ctrans(Pspec,    KT_MTrans,KT_Pzero)
    Pspec_L_T   =v_Ctrans(Pspec_L,  KT_MTrans,KT_Pzero)
    
    Pmeas_Sdev_T     =v_Ctrans(Pmeas_Sdev,    KT_MTrans,np.zeros(3))
    Pspec_Sdev_T     =v_Ctrans(Pspec_Sdev,    KT_MTrans,np.zeros(3))
    
    Points_T=pd.concat([pd.concat([Pmeas_T,Pmeas_Sdev_T],axis=0,sort=True),pd.concat([Pspec_T,Pspec_Sdev_T],axis=0,sort=True)],axis=1,sort=True)
    Points_L_T=pd.concat([pd.concat([Pmeas_L_T,Pmeas_Sdev_T],axis=0,sort=True),pd.concat([Pspec_L_T,Pspec_Sdev_T],axis=0,sort=True)],axis=1,sort=True)
    
    if output_lvl>=2:    
        # plot transformed data
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(Pmeas_T.loc['x'], Pmeas_T.loc['y'], Pmeas_T.loc['z'], color='b',label='meas. points')
        ax.scatter(Pspec_T.loc['x'], Pspec_T.loc['y'], Pspec_T.loc['z'], color='r',label='spec. points')
        # plot Lotfußpunkte
        ax.scatter(Pmeas_L_T.loc['x'], Pmeas_L_T.loc['y'], Pmeas_L_T.loc['z'], color='c', marker="+",label='meas. lot points')
        ax.scatter(Pspec_L_T.loc['x'], Pspec_L_T.loc['y'], Pspec_L_T.loc['z'], color='y', marker="+",label='spec. lot points')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    if output_lvl>=2:    
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('x / mm')
        ax1.set_ylabel('y / mm')
        plt.axis('equal')
        ax1.plot(Pmeas_L_T.loc['x'], Pmeas_L_T.loc['y'], 'r+',label='meas. points')
        ax1.grid()
        fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        fig.clf()
        plt.close(fig)
        
    return (Points_T, Points_L_T)

def Point_df_idx(df, steps=None, points=None, coords=None,
                 deepcopy=True, option=None):
    """Indexing a points dataframe by step (index), point-names and coordinates."""
    if deepcopy:
        dfs=df.copy(deep=True)
    else:
        dfs=df

    if option=='Regex':
        # if (type(points) is not str) or (type(coords) is not str):
        if not (isinstance(points,str) or isinstance(coords,str)):
            raise ValueError("Input is not a regex-string!")
            
    def nonchecker(vcheck):
        if vcheck is None:
            vcbool = False
        #elif (len(vcheck)>1) and not (type(vcheck) is str):
        #    vcbool = (vcheck!=None).all()
        else:
        #    vcbool = vcheck!=None
            vcbool = True
        return vcbool

    stepsbool=nonchecker(steps)
    pointsbool=nonchecker(points)
    coordsbool=nonchecker(coords)

    if stepsbool:
        dfs=dfs.loc(axis=0)[steps]

    if pointsbool:
        if type(dfs) is pd.core.series.Series:
            if option=='Regex':
                #points=dfs.droplevel(level=1).str.contains(pat=points,na=False,regex=True)
                points=dfs.index.droplevel(level=1).str.contains(pat=points,na=False,regex=True)
            dfs=dfs.loc[points,:]
        else:
            if option=='Regex':
                points=dfs.columns.droplevel(level=1).str.contains(pat=points,na=False,regex=True)
            dfs=dfs.loc(axis=1)[points,:]

    if coordsbool:
        if type(dfs) is pd.core.series.Series:
            if option=='Regex':
                coords=dfs.index.droplevel(level=0).str.contains(pat=coords,na=False,regex=True)
            #     dfs=dfs.loc[:,coords]
            # else:
            #     dfs=dfs.loc[:,[coords]]
            # if type(coords) is list:
            if isinstance(coords,(list,np.ndarray)):
                dfs=dfs.loc[:,coords]
            else:
                dfs=dfs.loc[:,[coords]]
        else:
            if option=='Regex':
                coords=dfs.columns.droplevel(level=0).str.contains(pat=coords,na=False,regex=True)
            #     dfs=dfs.loc(axis=1)[:,coords]
            # else:
            #     dfs=dfs.loc(axis=1)[:,[coords]]
            # if type(coords) is list:
            if isinstance(coords,(list,np.ndarray)):
                dfs=dfs.loc(axis=1)[:,coords]
            else:
                dfs=dfs.loc(axis=1)[:,[coords]]

    return dfs

def Points_dif_step(df, dif_step=0,
                    steps=None, points=None, coords=None,
                    deepcopy=True, option=None):
    """Substracts a Points df from an other."""
    dfs    = Point_df_idx(df,steps,points,coords,deepcopy,option)
    # Fehler bei Points_df_idx (Ausgabe Serie, muss var enthalten)
    df_dif = Point_df_idx(df,dif_step,points,coords,deepcopy,option)
    # df_dif = df.loc(axis=1)[:,coords].loc[dif_step]
    dfs    = dfs - df_dif
    return dfs

def Points_add_step(df, add_step=0, add_df=None,
                    steps=None, points=None, coords=None,
                    deepcopy=True, option=None):
    """Adding a Points df to an other."""
    dfs    = Point_df_idx(df,steps,points,coords,deepcopy,option)
    if add_df is None:
        df_add = Point_df_idx(df,add_step,points,coords,deepcopy,option)
    else:
        df_add = Point_df_idx(add_df,add_step,points,coords,deepcopy,option)
    dfs    = dfs + df_add
    return dfs

def Points_diff(df, diff_coord='y', first_val=0,
                steps=None, points=None, coords=None,
                deepcopy=True, option=None, 
                nan_policy = 'corr_nan'):
    """Returns differential of specified coordinates of Points."""
    dfs    = Point_df_idx(df,steps,points,coords,deepcopy,option)
    df_diff = Point_df_idx(df,steps,points,diff_coord,deepcopy,option).diff()
    if not np.isnan(first_val):
        df_diff.iloc[0]=0
    dfs.loc(axis=1)[:,diff_coord] = df_diff
    if nan_policy == 'corr_nan':
        sdf = dfs.loc(axis=1)[:,diff_coord].isna().droplevel(1,axis=1)
        dfs[sdf] = np.nan
    return dfs

def Points_eval_func(func, fit_params, pointdf, steps,
                     in_coord='x', out_coord='y', points=None,
                     deepcopy=True, option=None):
    """Evaluates a dataframe or series by a function."""
    if type(fit_params) is not pd.core.series.Series:
        raise ValueError('Fit Parameters are not pandas series!')
    
    dfs = Point_df_idx(df=pointdf, steps=steps,
                       points=points, coords=in_coord,
                       deepcopy=deepcopy, option=option)
    if type(dfs) is pd.core.frame.DataFrame:
        dfs = dfs.apply(lambda b: func(b,**fit_params.loc[b.name]),axis=1)
    else:
        dfs = dfs.apply(lambda b: func(b,**fit_params.loc[b.name]))
    dfs = dfs.rename(columns={in_coord: out_coord})

    return dfs

def Point_df_from_lin(x, steps, coords=['x','y'], coords_set='x',
                      col_type='str', Point_prefix='L'):
    """Prepare a points dataframe with index of step and values of x."""
    if col_type == 'str':
        xt  = Point_prefix + pd.Series(x).index.map(str)
    elif col_type == 'val':
        xt  = x
    else:
        raise NotImplementedError("Column index type %s not implemented!"%col_type)
    if coords is None:
        # round float point division
        mit = pd.Index(xt).map(lambda x: Evac.round_to_sigdig(x,_significant_digits_fpd))
        dft = pd.DataFrame([],index=steps,columns=mit)
        if not coords_set is None: dft.loc(axis=1)[:]=x
    else:        
        mit = pd.MultiIndex.from_product([xt,coords], names=['Points','Vars'])
        dft = pd.DataFrame([],index=steps,columns=mit)
        if not coords_set is None: dft.loc(axis=1)[:,coords_set]=x
    return dft

def Point_df_combine(df1, df2, coords_set='y',
                    deepcopy=True, option=None):
    """Combines a deepcopy of one points dataframe to an other."""
    dfs=df1.copy(deep=deepcopy)
    dfs.loc(axis=1)[:,coords_set]=df2.loc(axis=1)[:,coords_set]
    return dfs

#%% Bending line function collection

#%%% General Functions
def triangle_func_d0(x, xmin,xmax, f_0):
    """Standard triangle function with maximum in middle between min. and max. x coordinate."""
    eq=f_0*(1-2*abs(x)/(xmax-xmin))
    return eq
def triangle_func_d1(x, xmin,xmax, f_0):
    """First derivate of standard triangle function."""
    eq=f_0*(-2*np.sign(x)/(xmax-xmin))
    return eq
def triangle_func_d2(x, xmin,xmax, f_0):
    """Second derivate of standard triangle function (equal 0)."""
    eq=0
    return eq

def Shear_area(Area, CS_type='Rectangle', kappa=None):
    """
    Returns the shear area of a cross section.

    Parameters
    ----------
    Area : float or function or 1
        Cross section area.
    CS_type : string, optional
        Cross section type. The default is 'Rectangle'.
    kappa : float, optional
         Correction factor shear area. Depending to CS_type. The default is None.

    Returns
    -------
    AreaS : same type as Area
        DESCRIPTION.

    """
    if  CS_type=='Rectangle': kappa = 5/6
    AreaS = Area * kappa
    return AreaS

def gamma_V_det(poisson, t_mean, Length, CS_type='Rectangle', kappa=None):
    """
    Returns the ratio between shear to entire deformation in mid of bending beam.

    Parameters
    ----------
    poisson : float
        Poisson's ratio.
    t_mean : float
        Mean thickness.
    Length : float
        Distance between load bearings.
    CS_type : string, optional
        Cross section type. The default is 'Rectangle'.
    kappa : float, optional
        Correction factor shear area. Depending to CS_type. The default is None.

    Returns
    -------
    gamma_V : float
        Ratio between shear to entire deformation in mid of bending beam.

    """
    kappa = Shear_area(1, CS_type, kappa)
    
    gamma_V = 2 / kappa * (1 + poisson) * t_mean**2 / Length**2
    return gamma_V
#%%% Fourier series specific functions
def FSE_4sin_wlin_d0(x, xmin,xmax,FP, b1,b2,b3,b4,c,d, f_V_0=None):
    """
    Fourier series expansion with four sine elements and additional linear and constant element.
    Defined as:
 
    .. math::
     
        f(x; b1,b2,b3,b4,b5,c,d) = b1 \sin (\pi / (xmax-xmin) (x-xmin)) + b2 \sin (2 \pi / (xmax-xmin) (x-xmin)) + b3 \sin (3 \pi / (xmax-xmin) (x-xmin)) + b4 \sin (4 \pi / (xmax-xmin) (x-xmin)) +c (x-xmin) + d

    with `sine-parameters` for :math:`b1,b2,b3,b4`, `slope` for :math:`c` and `intercept` for :math:`d`.

    Parameters
    ----------
    x : array of float
        Array of cartesian coordinates in x-direction.
    xmin : float
        Cartesian coordinate in x-direction of start of function (curvature=0, p.e. left bearing).
        Defines length of Fourier series expansion with xmax-xmin.
    xmax : float
        Cartesian coordinate in x-direction of end of function (curvature=0, p.e. right bearing).
        Defines length of Fourier series expansion with xmax-xmin.
    b1 : float
        Parameter of first sine element.
    b2 : float
        Parameter of second sine element.
    b3 : float
        Parameter of third sine element.
    b4 : float
        Parameter of fourth sine element.
    c : float
        Parameter of slope.
    d : float
        Parameter of intercept.

    Returns
    -------
    eq : Function
        Fourier series expansion with four sine elements and additional linear and constant element.

    """
    eq=b1*np.sin(FP*(x-xmin))+b2*np.sin(2*FP*(x-xmin))+b3*np.sin(3*FP*(x-xmin))+b4*np.sin(4*FP*(x-xmin))+c*(x-xmin)+d
    return eq
def FSE_4sin_wlin_d1(x, xmin,xmax,FP, b1,b2,b3,b4,c,d=None, f_V_0=None):
    """First derivate of fourier series expansion with four sine elements and 
    additional linear and constant element. For more information see FSE_4sin_wlin_d0."""
    eq=FP*(b1*np.cos(FP*(x-xmin))+2*b2*np.cos(2*FP*(x-xmin))+3*b3*np.cos(3*FP*(x-xmin))+4*b4*np.cos(4*FP*(x-xmin)))+c
    return eq
def FSE_4sin_wlin_d2(x, xmin,xmax,FP, b1,b2,b3,b4,c=None,d=None, f_V_0=None):
    """Second derivate of fourier series expansion with four sine elements and 
    additional linear and constant element. For more information see FSE_4sin_wlin_d0."""
    eq=(-FP**2)*(b1*np.sin(FP*(x-xmin))+4*b2*np.sin(2*FP*(x-xmin))+9*b3*np.sin(3*FP*(x-xmin))+16*b4*np.sin(4*FP*(x-xmin)))
    return eq

def FSE_4sin_lin_func_d0(x, c,d, 
                         xmin=None,xmax=None,FP=None, b1=None,b2=None,b3=None,b4=None,
                         f_V_0=None):
    """Linear part of fourier series expansion with four sine elements and 
    additional linear and constant element. For more information see FSE_4sin_wlin_d0."""
    eq=c*(x-xmin)+d
    return eq
def FSE_4sin_lin_func_d1(x, c,d=None, 
                         xmin=None,xmax=None,FP=None, b1=None,b2=None,b3=None,b4=None,
                         f_V_0=None):
    """First deriavate of linear part of fourier series expansion with four sine elements and 
    additional linear and constant element. For more information see FSE_4sin_wlin_d0."""
    eq=c
    return eq
def FSE_4sin_lin_func_d2(x, c=None,d=None, 
                         xmin=None,xmax=None,FP=None, b1=None,b2=None,b3=None,b4=None,
                         f_V_0=None):
    """Second deriavate of linear part of fourier series expansion with four sine elements and 
    additional linear and constant element. For more information see FSE_4sin_wlin_d0. (Equal 0)"""
    eq=0
    return eq


def FSE_4sin_d0(x, xmin,xmax,FP, b1,b2,b3,b4,c=None,d=None, f_V_0=None):
    """Fourier series expansion with four sine elements without additional linear 
    and constant element. For more information see FSE_4sin_wlin_d0."""
    eq=b1*np.sin(FP*(x-xmin))+b2*np.sin(2*FP*(x-xmin))+b3*np.sin(3*FP*(x-xmin))+b4*np.sin(4*FP*(x-xmin))
    return eq
def FSE_4sin_d1(x, xmin,xmax,FP, b1,b2,b3,b4,c=None,d=None, f_V_0=None):
    """First deriavate of Fourier series expansion with four sine elements 
    without additional linear and constant element. For more information see FSE_4sin_wlin_d0."""
    eq=(FP)*(b1*np.cos(FP*(x-xmin))+2*b2*np.cos(2*FP*(x-xmin))+3*b3*np.cos(3*FP*(x-xmin))+4*b4*np.cos(4*FP*(x-xmin)))
    return eq       
def FSE_4sin_d2(x, xmin,xmax,FP, b1,b2,b3,b4,c=None,d=None, f_V_0=None):
    """Second deriavate of Fourier series expansion with four sine elements 
    without additional linear and constant element. For more information see FSE_4sin_wlin_d0."""
    eq=(-FP**2)*(b1*np.sin(FP*(x-xmin))+4*b2*np.sin(2*FP*(x-xmin))+9*b3*np.sin(3*FP*(x-xmin))+16*b4*np.sin(4*FP*(x-xmin)))
    return eq

def FSE_SF_func_d0(x, xmin,xmax, f_V_0,
                   FP=None, b1=None,b2=None,b3=None,b4=None,c=None,d=None, opt=None):
    """Shear deformation part of ourier series expansion with four sine elements 
    and additional linear and constant element. Children of triangle_func_d0.
    For more information see FSE_4sin_wlin_d0."""
    eq=triangle_func_d0(x=x, xmin=xmin,xmax=xmax, f_0=f_V_0)
    return eq
def FSE_SF_func_d1(x, xmin,xmax, f_V_0,
                   FP=None, b1=None,b2=None,b3=None,b4=None,c=None,d=None, opt=None):
    """First derivate of shear deformation part of ourier series expansion with
    four sine elements and additional linear and constant element. 
    Children of triangle_func_d1. For more information see FSE_4sin_wlin_d0."""
    eq=triangle_func_d1(x=x, xmin=xmin,xmax=xmax, f_0=f_V_0)
    return eq
def FSE_SF_func_d2(x, xmin,xmax, f_V_0,
                   FP=None, b1=None,b2=None,b3=None,b4=None,c=None,d=None, opt=None):
    """Second derivate of shear deformation part of ourier series expansion with
    four sine elements and additional linear and constant element. 
    Children of triangle_func_d1. For more information see FSE_4sin_wlin_d0."""
    eq=triangle_func_d2(x=x, xmin=xmin,xmax=xmax, f_0=f_V_0)
    return eq


#%%% Classes
class Bend_func_sub:
    """Lowest element which provides necessary function informations."""
    def __init__(self, func, sy_expr, sy_string,
                 independent_vars=None, param_names=None, param_types=None, name=None, **kws):
        """
        Subfunction takes callable function and additional informations.

        Parameters
        ----------
        func : callable
            Function.
        sy_expr : sympy expression
            Sympy expression.
        sy_string : string expression
            Expression string.
        independent_vars : array of string | string, optional
            Arguments to `func` that are independent variables. The default is None.
        param_names : array of string, optional
            Names of arguments to `func` that are to be made into
            parameters. The default is None.
        name : string, optional
            Name of function. The default is None.
        **kws : TYPE
            Additional keyword arguments to pass.

        Returns
        -------
        None.

        """
        self.func=func
        self.sy_expr=sy_expr
        self.sy_string=sy_string
        self.independent_vars=independent_vars
        self.param_names=param_names
        # self.param_types=param_types
        # self.param_types=Bend_func_sub._set_free_type(self)
        _param_types=Bend_func_sub._set_free_type(param_names,param_types)
        self.param_types=_param_types
        self.name=name
        self.opts = kws
    
    def __call__(self, x, kws, **opts):
        """
        Call and evaluate underlayed function.
        If x is an array of float and kws is a OrderedDict, result will be an array of float.
        If x is a Dataframe or float and kws is a Series, evaluates each index in kws.

        Parameters
        ----------
        x : array of float | Dataframe of type Points
            Independet variables to evaluate function.
        **kws : OrderedDict | Series of OrderedDicts
            Additional keyword arguments to pass to func
            (p.e. parameter value dictionary).

        Returns
        -------
        array of float
            Evaluation values of function to x and kws.

        """
        if (type(x) is pd.core.frame.DataFrame) and (type(kws) is pd.core.series.Series):
            return Points_eval_func(self.func, fit_params=kws, pointdf=x, steps=kws.index)
        if (type(x) is np.ndarray) and (type(kws) is pd.core.series.Series):
            # xt='L'+pd.Series(x).index.map(str)
            # mit=pd.MultiIndex.from_product([xt,['y']],names=['Points','Vars'])
            # # dft=pd.DataFrame([],index=kws.index,columns=mit)
            # # dft.loc[:]=x
            # # return Points_eval_func(self.func, fit_params=kws, pointdf=dft, steps=kws.index)
            dfs=kws.apply(lambda b: self.func(x,**b))
            dft=pd.DataFrame(item for item in dfs)
            dft.index=kws.index
            if not 'coords'       in opts: opts.update({'coords':      ['y']})
            if not 'coords_set'   in opts: opts.update({'coords_set':   None})
            if not 'col_type'     in opts: opts.update({'col_type':    'str'})
            if not 'Point_prefix' in opts: opts.update({'Point_prefix':'L'  })
            mit=Point_df_from_lin(x=x, steps=[0], **opts).columns
            dft.columns=mit
            return dft
        if (type(x) is pd.core.indexes.numeric.Float64Index) and (type(kws) is pd.core.series.Series):
            dfs=kws.apply(lambda b: self.func(x.to_numpy(),**b))
            dft=pd.DataFrame(item for item in dfs)
            dft.index=kws.index
            dft.columns=x
            return dft
        if type(x) is int: x=float(x)
        if (type(x) is float) and (type(kws) is pd.core.series.Series):
            dfs=kws.apply(lambda b: self.func(x,**b))
            dfs.name=self.name+'>'+str(x)
            return dfs
        return self.func(x,**kws)
    
    def __getitem__(self,obj):
        return getattr(self, obj)
    
    def __str__(self):
        return self.sy_string
    
    # def _set_free_type(self):
    #     """Set all remaining parameter types, not specified in types to 'free'."""
    #     self.param_types = {i: (self.param_types[i] if i in self.param_types.keys() else 'free') 
    #                         for i in self.param_names}
    def _set_free_type(pnames,ptypes):
        """Set all remaining parameter types, not specified in types to 'free'."""
        param_types = {i: (ptypes[i] if i in ptypes.keys() else 'free') 
                            for i in pnames}
        return param_types
    
# # Test für bessere zuordnung    
# class Bend_func_sub_FSE_4sin_wlin_d0(Bend_func_sub):
#     def __init__(self, name=None, **kws):
#         def FSE_4sin_wlin_d0(x, xmin,xmax,FP, b1,b2,b3,b4,c,d, f_V_0=None):
#             eq=b1*np.sin(FP*(x-xmin))+b2*np.sin(2*FP*(x-xmin))+b3*np.sin(3*FP*(x-xmin))+b4*np.sin(4*FP*(x-xmin))+c*(x-xmin)+d
#             return eq
#         f  = FSE_4sin_wlin_d0
#         ss = 'b1*sin(FP*(x-xmin))+b2*sin(2*FP*(x-xmin))+b3*sin(3*FP*(x-xmin))+b4*sin(4*FP*(x-xmin))+c*(x-xmin)+d'
#         se = sy.parse_expr(ss)
#         param_names = np.array(['x','xmin','xmax','FP','b1','b2','b3','b4','c','d','f_V_0'])
#         super().__init__(func=f, sy_expr=se, sy_string=ss,
#                          independent_vars='x', param_names=param_names,
#                          name=name, **kws)


class Bend_func_cohort(object):
    """Function collection of Subfunctions (Bend_func_sub)"""
    
    # @property
    # def _constructor(self) -> type[Bend_func_cohort]:
    #     return Bend_func_cohort
    
    def __init__(self, d0=None, d1=None, d2=None,
                 name=None, **kws):
        # self.d0=Bend_func_sub.d0
        # self.d1=Bend_func_sub.d1
        # self.d2=Bend_func_sub.d2
        self.name=name
        self.opts=kws
            
    def __getitem__(self,obj):
        return getattr(self, obj)

    def Init_fandds(self, expr_d0, var_names, var='x', var_types=None,
                    option='d0_str_to_all',
                    expr_d1=None, expr_d2=None,
                    func_d0=None, func_d1=None, func_d2=None):
        """
        Return Sympy expressions and associated lambdified functions, depending on option-string.
    
        Parameters
        ----------
        expr_d0 : string or sympy.expression
            String or expression, which determine function.
        var_names : TYPE
            Variable names for parsing.
        var : TYPE, optional
            Variable name for derivation. The default is 'x'.
        option : string, optional
            Option of function generation, in form of which(d0/each)_type(str/expr)_to_what(d0/all). The default is 'd0_str_to_all'.
        expr_d1 : string or sympy.expression, optional
            String or expression, which determine 1st derivate of function. The default is None.
        expr_d2 : string or sympy.expression, optional
            String or expression, which determine 2nd derivate of function. The default is None.
    
        Raises
        ------
        NotImplementedError
            Error for not implemented option.
    
        Returns
        -------
        func_package : Bend_func_cohort
            A function collection of all generated sympy expressions
            and associated lambdified functions and their 1st and 2nd derivates.
    
        """
        if option == 'func_to_all':
            self.d0=Bend_func_sub(func=func_d0, sy_expr=sy.parse_expr(expr_d0), sy_string=expr_d0,
                                  name=self.name+'_d0', independent_vars=var,
                                  param_names=var_names, param_types=var_types)
            self.d1=Bend_func_sub(func=func_d1, sy_expr=sy.parse_expr(expr_d1), sy_string=expr_d1,
                                  name=self.name+'_d1', independent_vars=var,
                                  param_names=var_names, param_types=var_types)
            self.d2=Bend_func_sub(func=func_d2, sy_expr=sy.parse_expr(expr_d2), sy_string=expr_d2,
                                  name=self.name+'_d2', independent_vars=var,
                                  param_names=var_names, param_types=var_types)
        else:            
            if option=='d0_str_to_all':
                    d0 = sy.parse_expr(expr_d0)
                    d1 = sy.simplify(d0.diff(var))
                    d2 = sy.simplify(d1.diff(var))
            elif option=='d0_expr_to_all':
                    d0 = expr_d0
                    d1 = sy.simplify(d0.diff(var))
                    d2 = sy.simplify(d1.diff(var))
            elif option== 'each_str_to_all':
                    d0 = sy.parse_expr(expr_d0)
                    d1 = sy.parse_expr(expr_d1)
                    d2 = sy.parse_expr(expr_d2)
            elif option== 'each_expr_to_all':
                    d0 = expr_d0
                    d1 = expr_d1
                    d2 = expr_d2
            elif option== 'd0_str_to_d0':
                    d0 = sy.parse_expr(expr_d0)
            elif option== 'd0_expr_to_d0':
                    d0 = expr_d0
            else:
                raise NotImplementedError("Option not implemented!")
                
            
            if ('to_all') in option:
                d0_func = sy.lambdify(var_names, d0)
                d1_func = sy.lambdify(var_names, d1)
                d2_func = sy.lambdify(var_names, d2)
                self.d0=Bend_func_sub(func=d0_func, sy_expr=d0, sy_string=expr_d0.__str__(),
                                             name=self.name+'_d0', independent_vars=var,
                                             param_names=var_names, param_types=var_types)
                self.d1=Bend_func_sub(func=d1_func, sy_expr=d1, sy_string=d1.__str__(),
                                             name=self.name+'_d1', independent_vars=var,
                                             param_names=var_names, param_types=var_types)
                self.d2=Bend_func_sub(func=d2_func, sy_expr=d2, sy_string=d2.__str__(),
                                             name=self.name+'_d2', independent_vars=var,
                                             param_names=var_names, param_types=var_types)
    
            elif ('to_d0') in option:
                d0_func = sy.lambdify(var_names, d0)
                self.d0=Bend_func_sub(func=d0_func, sy_expr=d0, sy_string=expr_d0,
                                             name='_d0', independent_vars=var,
                                             param_names=var_names, param_types=var_types)
                
            else:
                raise NotImplementedError("Option not implemented!")

class Bend_func_legion(object):
    """Bend line describing functions and their 1st and 2nd derivates
       Example:
        bl=Bend_func_legion(name='FSE fit')
        bl.Builder(option='FSE')"""
    # @property
    # def _constructor(self) -> type[Bend_func_cohort]:
    #     return Bend_func_cohort
    
    def __init__(self, name=None, description=None, **kws):
        """
        Inital properties setter.

        Parameters
        ----------
        name : string, optional
            Name of bend func legion. The default is None.
        description : TYPE, optional
            Description of bend func legion. The default is None.
        **kws : dict
            Dictionary of keyword arguments.

        Returns
        -------
        None.

        """
        # self.d0=Bend_func_sub.d0
        # self.d1=Bend_func_sub.d1
        # self.d2=Bend_func_sub.d2
        self.name=name
        self.description=description
        self.opts=kws
        
    def __getitem__(self,obj):
        return getattr(self, obj)
        
    def Builder(self, option='FSE'):
        """
        Build an instance of Bend per option to prepare fit of measured bend line.

        Parameters
        ----------
        option : string, optional
            Kind of . The default is 'FSE'.

        Raises
        ------
        NotImplementedError
            Building option not implemented.

        Returns
        -------
        None.

        """
        if option=='FSE':
            self.description='Bend line as lambdified Fourier series expansion'
    
            w_A_d0_str='b1*sin(FP*(x-xmin))+b2*sin(2*FP*(x-xmin))+b3*sin(3*FP*(x-xmin))+b4*sin(4*FP*(x-xmin))+c*(x-xmin)+d'
            # w_variables_names=np.array(['x','xmin','xmax','FP','b1','b2','b3','b4','c','d'])
            w_variables_names=np.array(['x','xmin','xmax','FP','b1','b2','b3','b4','c','d','f_V_0'])
            self.w_A=Bend_func_cohort(name='A')
            Bend_func_cohort.Init_fandds(self.w_A, expr_d0=w_A_d0_str,
                                         var_names=w_variables_names,
                                         option='d0_str_to_all')
            
            w_I_d0_str = 'c*(x-xmin)+d'
            self.w_I=Bend_func_cohort(name='I')
            Bend_func_cohort.Init_fandds(self.w_I, expr_d0=w_I_d0_str,
                                     var_names=w_variables_names,
                                     option='d0_str_to_all')
            
            w_S_d0_exp = self.w_A.d0.sy_expr - self.w_I.d0.sy_expr
            self.w_S=Bend_func_cohort(name='S')
            Bend_func_cohort.Init_fandds(self.w_S, expr_d0=w_S_d0_exp,
                                     var_names=w_variables_names,
                                     option='d0_expr_to_all')
            
            # w_variables_names_V=np.append(w_variables_names,'f_V_0')
            w_V_d0_str = 'f_V_0*(1-2*abs(x)/(xmax-xmin))'
            w_V_d1_str = 'f_V_0*(-2*sign(x)/(xmax-xmin))'
            w_V_d2_str = '0'
            self.w_V=Bend_func_cohort(name='V')
            Bend_func_cohort.Init_fandds(self.w_V, expr_d0=w_V_d0_str,
                                     expr_d1=w_V_d1_str, expr_d2=w_V_d2_str,
                                     # var_names=w_variables_names_V,
                                     var_names=w_variables_names,
                                     option='each_str_to_all')
    
            w_Mima_d0_exp = self.w_S.d0.sy_expr - self.w_V.d0.sy_expr
            w_Mima_d1_exp = self.w_S.d1.sy_expr - self.w_V.d1.sy_expr
            w_Mima_d2_exp = self.w_S.d2.sy_expr - self.w_V.d2.sy_expr
            self.w_M_ima=Bend_func_cohort(name='M imaginary')
            Bend_func_cohort.Init_fandds(self.w_M_ima, expr_d0=w_Mima_d0_exp,
                                     expr_d1=w_Mima_d1_exp, expr_d2=w_Mima_d2_exp,
                                     # var_names=w_variables_names_V,
                                     var_names=w_variables_names,
                                     option='each_expr_to_all')
            
            w_M_d0_str='b1*sin(FP*(x-xmin))+b2*sin(2*FP*(x-xmin))+b3*sin(3*FP*(x-xmin))+b4*sin(4*FP*(x-xmin))'
            w_variables_names_M=np.array(['x','xmin','xmax','FP','b1','b2','b3','b4']) 
            self.w_M=Bend_func_cohort(name='M')
            Bend_func_cohort.Init_fandds(self.w_M, expr_d0=w_M_d0_str,
                                         var_names=w_variables_names_M,
                                         # var_names=w_variables_names,
                                         option='d0_str_to_all')
        elif option=='FSE_fixed':
            self.description='Bend line as Fourier series expansion'
    
            w_A_d0_str='b1*sin(FP*(x-xmin))+b2*sin(2*FP*(x-xmin))+b3*sin(3*FP*(x-xmin))+b4*sin(4*FP*(x-xmin))+c*(x-xmin)+d'
            w_A_d1_str='FP*b1*cos(FP*(x - xmin)) + 2*FP*b2*cos(2*FP*(x - xmin)) + 3*FP*b3*cos(3*FP*(x - xmin)) + 4*FP*b4*cos(4*FP*(x - xmin)) + c'
            w_A_d2_str='-FP**2*(b1*sin(FP*(x - xmin)) + 4*b2*sin(2*FP*(x - xmin)) + 9*b3*sin(3*FP*(x - xmin)) + 16*b4*sin(4*FP*(x - xmin)))'
            w_A_d0_func=FSE_4sin_wlin_d0
            w_A_d1_func=FSE_4sin_wlin_d1
            w_A_d2_func=FSE_4sin_wlin_d2
            w_variables_names=np.array(['x','xmin','xmax','FP',
                                        'b1','b2','b3','b4','c','d',
                                        'f_V_0'])
            w_variables_types=dict({'x':'independent',
                                    'xmin':'fixed','xmax':'fixed',
                                    'FP':'expr','f_V_0':'post'})
            
            self.w_A=Bend_func_cohort(name='A')
            Bend_func_cohort.Init_fandds(self.w_A, 
                                         expr_d0=w_A_d0_str, func_d0=w_A_d0_func,
                                         expr_d1=w_A_d1_str, func_d1=w_A_d1_func,
                                         expr_d2=w_A_d2_str, func_d2=w_A_d2_func,
                                         var_names=w_variables_names,
                                         var_types=w_variables_types,
                                         option='func_to_all')
            
            w_I_d0_str = 'c*(x-xmin)+d'
            w_I_d1_str = 'c'
            w_I_d2_str = '0'
            w_I_d0_func=FSE_4sin_lin_func_d0
            w_I_d1_func=FSE_4sin_lin_func_d1
            w_I_d2_func=FSE_4sin_lin_func_d2
            self.w_I=Bend_func_cohort(name='I')
            Bend_func_cohort.Init_fandds(self.w_I,
                                         expr_d0=w_I_d0_str, func_d0=w_I_d0_func,
                                         expr_d1=w_I_d1_str, func_d1=w_I_d1_func,
                                         expr_d2=w_I_d2_str, func_d2=w_I_d2_func,
                                         var_names=w_variables_names,
                                         var_types=w_variables_types,
                                         option='func_to_all')
            
            w_S_d0_str = 'b1*sin(FP*(x - xmin)) + b2*sin(2*FP*(x - xmin)) + b3*sin(3*FP*(x - xmin)) + b4*sin(4*FP*(x - xmin))'
            w_S_d1_str = 'FP*(b1*cos(FP*(x - xmin)) + 2*b2*cos(2*FP*(x - xmin)) + 3*b3*cos(3*FP*(x - xmin)) + 4*b4*cos(4*FP*(x - xmin)))'
            w_S_d2_str = '-FP**2*(b1*sin(FP*(x - xmin)) + 4*b2*sin(2*FP*(x - xmin)) + 9*b3*sin(3*FP*(x - xmin)) + 16*b4*sin(4*FP*(x - xmin)))'
            w_S_d0_func=FSE_4sin_d0
            w_S_d1_func=FSE_4sin_d1
            w_S_d2_func=FSE_4sin_d2
            self.w_S=Bend_func_cohort(name='S')
            Bend_func_cohort.Init_fandds(self.w_S,
                                         expr_d0=w_S_d0_str, func_d0=w_S_d0_func,
                                         expr_d1=w_S_d1_str, func_d1=w_S_d1_func,
                                         expr_d2=w_S_d2_str, func_d2=w_S_d2_func,
                                         var_names=w_variables_names,
                                         var_types=w_variables_types,
                                         option='func_to_all')
            
            w_V_d0_str = 'f_V_0*(1-2*abs(x)/(xmax-xmin))'
            w_V_d1_str = 'f_V_0*(-2*sign(x)/(xmax-xmin))'
            w_V_d2_str = '0'
            w_V_d0_func=FSE_SF_func_d0
            w_V_d1_func=FSE_SF_func_d1
            w_V_d2_func=FSE_SF_func_d2
            self.w_V=Bend_func_cohort(name='V')
            Bend_func_cohort.Init_fandds(self.w_V,
                                         expr_d0=w_V_d0_str, func_d0=w_V_d0_func,
                                         expr_d1=w_V_d1_str, func_d1=w_V_d1_func,
                                         expr_d2=w_V_d2_str, func_d2=w_V_d2_func,
                                         var_names=w_variables_names,
                                         var_types=w_variables_types,
                                         option='func_to_all')
    
            w_Mima_d0_exp = self.w_S.d0.sy_expr - self.w_V.d0.sy_expr
            w_Mima_d1_exp = self.w_S.d1.sy_expr - self.w_V.d1.sy_expr
            w_Mima_d2_exp = self.w_S.d2.sy_expr - self.w_V.d2.sy_expr
            self.w_M_ima=Bend_func_cohort(name='M imaginary')
            Bend_func_cohort.Init_fandds(self.w_M_ima, expr_d0=w_Mima_d0_exp,
                                     expr_d1=w_Mima_d1_exp, expr_d2=w_Mima_d2_exp,
                                     var_names=w_variables_names,
                                     var_types=w_variables_types,
                                     option='each_expr_to_all')
            
            w_M_d0_str ='b1*sin(FP*(x-xmin))+b2*sin(2*FP*(x-xmin))+b3*sin(3*FP*(x-xmin))+b4*sin(4*FP*(x-xmin))'
            w_M_d1_str = 'FP*(b1*cos(FP*(x - xmin)) + 2*b2*cos(2*FP*(x - xmin)) + 3*b3*cos(3*FP*(x - xmin)) + 4*b4*cos(4*FP*(x - xmin)))'
            w_M_d2_str = '-FP**2*(b1*sin(FP*(x - xmin)) + 4*b2*sin(2*FP*(x - xmin)) + 9*b3*sin(3*FP*(x - xmin)) + 16*b4*sin(4*FP*(x - xmin)))'
            w_M_d0_func=FSE_4sin_d0
            w_M_d1_func=FSE_4sin_d1
            w_M_d2_func=FSE_4sin_d2
            w_variables_names_M=np.array(['x','xmin','xmax','FP','b1','b2','b3','b4'])
            w_variables_types_M=dict({'x':'independent',
                                      'xmin':'fixed','xmax':'fixed',
                                      'FP':'expr'})
            self.w_M=Bend_func_cohort(name='M')
            Bend_func_cohort.Init_fandds(self.w_M,
                                         expr_d0=w_M_d0_str, func_d0=w_M_d0_func,
                                         expr_d1=w_M_d1_str, func_d1=w_M_d1_func,
                                         expr_d2=w_M_d2_str, func_d2=w_M_d2_func,
                                         var_names=w_variables_names_M,
                                         var_types=w_variables_types_M,
                                         option='func_to_all')
        elif option=='P4O':
            self.description='Bend line as 4th order polynom'
            
            w_A_d0_str='a*x**4+b*x**3+c*x**2+d*x+e'
            w_variables_names=np.array(['x','xmin','xmax','a','b','c','d','e','f_V_0'])
            self.w_A=Bend_func_cohort(name='A')
            Bend_func_cohort.Init_fandds(self.w_A, expr_d0=w_A_d0_str,
                                     var_names=w_variables_names,
                                     option='d0_str_to_all')
            
            w_I_d0_str = '((a*(xmax)**4+b*(xmax)**3+c*(xmax)**2+d*(xmax)+e)-(a*(xmin)**4+b*(xmin)**3+c*(xmin)**2+d*(xmin)+e))/(xmax-xmin)*x+((a*(xmax)**4+b*(xmax)**3+c*(xmax)**2+d*(xmax)+e)+(a*(xmin)**4+b*(xmin)**3+c*(xmin)**2+d*(xmin)+e))/2'
            self.w_I=Bend_func_cohort(name='I')
            Bend_func_cohort.Init_fandds(self.w_I, expr_d0=w_I_d0_str,
                                     var_names=w_variables_names,
                                     option='d0_str_to_all')
            
            w_S_d0_exp = self.w_A.d0.sy_expr - self.w_I.d0.sy_expr
            self.w_S=Bend_func_cohort(name='S')
            Bend_func_cohort.Init_fandds(self.w_S, expr_d0=w_S_d0_exp,
                                     var_names=w_variables_names,
                                     option='d0_expr_to_all')
            
            # w_variables_names_V=np.append(w_variables_names,'f_V_0')
            w_V_d0_str = 'f_V_0*(1-2*abs(x)/(xmax-xmin))'
            w_V_d1_str = 'f_V_0*(-2*sign(x)/(xmax-xmin))'
            w_V_d2_str = '0'
            self.w_V=Bend_func_cohort(name='V')
            Bend_func_cohort.Init_fandds(self.w_V, expr_d0=w_V_d0_str,
                                     expr_d1=w_V_d1_str, expr_d2=w_V_d2_str,
                                     var_names=w_variables_names,
                                     option='each_str_to_all')
    
            w_Mima_d0_exp = self.w_S.d0.sy_expr - self.w_V.d0.sy_expr
            w_Mima_d1_exp = self.w_S.d1.sy_expr - self.w_V.d1.sy_expr
            w_Mima_d2_exp = self.w_S.d2.sy_expr - self.w_V.d2.sy_expr
            self.w_M_ima=Bend_func_cohort(name='M imaginary')
            Bend_func_cohort.Init_fandds(self.w_M_ima, expr_d0=w_Mima_d0_exp,
                                     expr_d1=w_Mima_d1_exp, expr_d2=w_Mima_d2_exp,
                                     var_names=w_variables_names,
                                     option='each_expr_to_all')
            
            w_M_d0_str = 'a*x**4+b*x**3+c*x**2+d*x+e'
            w_variables_names_M = np.array(['x','xmin','xmax','a','b','c','d','e'])
            self.w_M=Bend_func_cohort(name='M')
            Bend_func_cohort.Init_fandds(self.w_M, expr_d0=w_M_d0_str,
                                     var_names=w_variables_names_M,
                                     option='d0_str_to_all')
        
        else:
            raise NotImplementedError("Option %s not implemented!"%option)
            
    def save_dill(self,filename):
        """
        Save builded Bend_func legion
        Does not work!!!
        
        """
        dill.dump(self, open(filename, "wb"))
        raise NotImplementedError("Not implemented successfully!")
        
    def load_dill(self,filename):
        """
        Load builded Bend_func legion
        Does not work!!!
        
        """
        dill.load(self, open(filename, "rb"))
        raise NotImplementedError("Not implemented successfully!")

#%% Fitting

def lmfit_modelize(Bend_func_sub, option='init'):
    """Returns a lmfit.model defined by Bend function sub."""
    mo=lmfit.Model(Bend_func_sub.func, independent_vars=Bend_func_sub.independent_vars)
    mo.name=Bend_func_sub.name+'_fit'
    if option=='init':
        par = mo.make_params()
    else:
        raise NotImplementedError("Option %s not implemented!"%option)
    return mo,par


def lmfit_param_adder(par_df):
    """Returns lmfit Parameterset defined by pandas Dataframe."""
    params = lmfit.Parameters()
    for i in par_df.index:
        if par_df.loc[i,'typ'] == 'independent':
            pass
            # params.add(i, value = None, vary = False)
        elif par_df.loc[i,'typ'] == 'fixed':
            params.add(i, value = par_df.loc[i,'val'], vary = False)
        elif par_df.loc[i,'typ'] == 'expr':
            params.add(i, expr = par_df.loc[i,'val'], vary = False)
        elif par_df.loc[i,'typ'] == 'post':
            params.add(i, value = None, vary = False)
        elif par_df.loc[i,'typ'] == 'free':
            params.add(i, value = par_df.loc[i,'val'], vary = True,
                       min = par_df.loc[i,'min'], max = par_df.loc[i,'max'])
        else:
            raise NotImplementedError("Variable type not implemented!")
    return params
            
def lmfit_free_val_setter(Bend_func_sub, param_val={}, default_val=-1.0):
    """Preset free values/parameters for lmfit."""
    pts=pd.Series(Bend_func_sub.param_types)
    pval_guess={i: (param_val[i] if i in param_val.keys() else default_val)
                for i in pts.loc[pts=='free'].index}
    return pval_guess

def lmfit_param_key_checker(Bend_func_sub, param_dict):
    """Check parameters of lmfit with Bend_func_sub."""
    param_dict=copy.deepcopy(param_dict)
    t=[]
    for i in param_dict.keys():
        if not i in list(Bend_func_sub.param_types.keys()):
            t.append(i)
    for i in t:        
        param_dict.pop(i,None)
    return param_dict

def lmfit_param_prep(option,
                     param_name, param_val=None, param_type=None, 
                     param_min=None, param_max=None):
    """Prepare fit parameters for fitting."""
    if option=='1st_build':
        # w_variables_df=pd.DataFrame(np.full(len(w_variables_names),4,None),
        #                             index=w_variables_names,
        #                             columns=['type','val','min','max'])
        par_typ = pd.Series(param_type,index=param_name,dtype='string',name='typ')
        par_val = pd.Series(param_val,index=param_name,dtype='O',name='val')
        par_min = pd.Series(param_min,index=param_name,dtype='O',name='min')
        par_max = pd.Series(param_max,index=param_name,dtype='O',name='max')
        par_df = pd.DataFrame([par_typ,par_val,par_min,par_max]).T
        params = lmfit_param_adder(par_df)
        
    elif option=='1st_build_Bfs':
        par_df = pd.DataFrame(np.full((len(param_name.param_names),4),None),
                              index=param_name.param_names,
                              columns=['typ','val','min','max'])
        if param_type is None:
            par_typ = param_name.param_types
        else:
            par_typ = param_type
        if param_val is None:
            # par_val = param_name.param_vals
            raise ValueError("Set at least fixed and expression parameter values!")
        else:
            par_val = param_val
            par_val = lmfit_param_key_checker(param_name, par_val)
            par_val.update(lmfit_free_val_setter(param_name,par_val))
        if param_min is None:
            # par_min = param_name.param_mins
            par_min = {i: -np.inf for i in param_name.param_names}
        else:
            par_min = param_min
            par_min = lmfit_param_key_checker(param_name, par_min)
        if param_max is None:
            # par_max = param_name.param_maxs
            par_max = {i: np.inf for i in param_name.param_names}
        else:
            par_max = param_max
            par_max = lmfit_param_key_checker(param_name, par_max)
            
        par_df.loc[par_typ.keys(),'typ']=par_typ
        par_df.typ.loc[np.invert(par_df.typ.isin(_param_types_not_free))]='free'    
        par_df.loc[par_val.keys(),'val']=par_val
        par_df.loc[par_min.keys(),'min']=par_min
        par_df.loc[par_max.keys(),'max']=par_max
        params = lmfit_param_adder(par_df)
        
    elif option == 'old_result':
        params = param_name
    
    elif option == 'old_dict':
        params = lmfit.Parameters(param_name)
    else:
        raise NotImplementedError("Option not implemented!")
        
    return params


def shaped_array_fill_fandl(ShapeAr, fElemV, lElemV):
    """
    Returns an 1D numpy array with shape of input array and filled with ones.
    First and last element replaced with input data.

    Parameters
    ----------
    ShapeAr : numpy.array
        Input array wich determine shape of output array.
    fElemV : float64
        Value of first element passed to output array.
    lElemV : float64
        Value of last element passed to output array.

    Returns
    -------
    OutAr : numpy array
        1D numpy array with shape of input array and filled with ones.
        First and last element replaced with input data.

    """
    OutAr     = 0 * np.ones(shape=(len(ShapeAr),))
    OutAr[0]  = fElemV
    OutAr[-1] = lElemV
    return OutAr

def res_multi_const_weighted(params, x, func, func_d2,
                             x_lB, x_rB, func_err_weight=[1,10,100,100], 
                             load_dir='-y', data=None):
    """
    Returns weighted error sum according to different constraints between input values and function values.
    Constraints:
        0. error between data and function value
        1. error between function value of left and right bound to zero
        2. error between negative/positive value of second derivate of function and zero (depends on load direction)
        3. error between value of second derivate of function of left and right bound to zero
    

    Parameters
    ----------
    params : OrderedDict / dict / array
        Parameters to be used in functions.
    x : array of float
        Cartesian coordinate in x-direction.
    func : lamopdified function
        Input function.
    func_d2 : lamopdified function
        Second derivate (curvature) of input function.
    x_lB : float
        Cartesian coordinate in x-direction of start of function (curvature=0, p.e. left bearing).
    x_rB : float
        Cartesian coordinate in x-direction of end of function (curvature=0, p.e. right bearing).
    func_err_weight : array of float, optional
        Error weights assigned to constraints. 
        Enter [1,0,0,0] for standard residual fit on displacement.
        The default is [1,10,100,100].
    load_dir : string, optional
        Direction of displacement application.
        Possible are:
            - "-y": application in negative y-direction (standard, curvature positve(err2))
            - "+y": application in negative y-direction (curvature negative(err2))
    data : array of float, optional
        Cartesian coordinate in y-direction. The default is None.

    Returns
    -------
    err: numpy array
        Weighted multi constraint error sum.

    """
    if data is None:
        return func(x,**params)
    if func_err_weight is None:
        func_err_weight=[1,1,1,1]
        
    ew=func_err_weight/np.sum(func_err_weight)
    
    # Error on displacement
    err0=(func(x,**params) - data)
    if func_err_weight == [1,0,0,0]:
        return err0
    # Error on displacement on left end right border in aspect to 0
    err1=(func(shaped_array_fill_fandl(x,x_lB,x_rB),**params)-0)*shaped_array_fill_fandl(x,1,1)
    # Error on curvature (have to be positive/negative for loading in -y/+y)
    if load_dir == '-y':
        err2=np.where(func_d2(x,**params)<0.0, -func_d2(x,**params), 0)
    elif load_dir == '+y':
        err2=np.where(func_d2(x,**params)>0.0, -func_d2(x,**params), 0)
    else:
        raise NotImplementedError("Loading direction %s not implemented!"%load_dir)
    # Error on curvature on left end right border in aspect to 0
    err3=(func_d2(shaped_array_fill_fandl(x,x_lB,x_rB),**params)-0)*shaped_array_fill_fandl(x,1,1)
    
    err=((err0*ew[0])**2+(err1*ew[1])**2+(err2*ew[2])**2+(err3*ew[3])**2)**0.5
    return err

def lmfit_bound_checker(Fit_Result, BFs, param_check_types = ['free'],level=1):
    """Check if lmfit Fit Result excides or hits bounds."""
    msg=''
    ub_bool = False
    lb_bool = False
    for p in Fit_Result.params:
        if BFs.param_types[p] in param_check_types:
            tv  = Fit_Result.params[p].value
            tub = Fit_Result.params[p].max
            tlb = Fit_Result.params[p].min
            if (tv >= tub) or (tv <= tlb):
                if tv >= tub:
                    ub_bool=True
                    msg+=('\nParameter %s with value %e excides maximum of %e'%(p,tv,tub))
                if tv <= tlb:
                    lb_bool=True
                    msg+=('\nParameter %s with value %e excides minimum of %e'%(p,tv,tlb))
            else:
                msg+=('\nParameter %s with value %e is inside bounds [%e,%e]'%(p,tv,tlb,tub))
    bound_bool=ub_bool|lb_bool
    return bound_bool, ub_bool, lb_bool, msg

def Multi_minimize(x, data, params, func_d0, func_d2, 
                   max_nfev, nan_policy, err_weights, x_lB, x_rB, load_dir='-y'):
    """
    Performs a weighted multiconstraint least-square-fit, based on lmfit.minimize.
    Returns lmfit-result, parameter-dict, Coefficient of determination for multi-errors and only displacement as well.

    Parameters
    ----------
    x : array of float
        Cartesian coordinate in x-direction.
    data : array of float, optional
        Cartesian coordinate in y-direction. The default is None.
    params : OrderedDict / dict / array
        Parameters to be used in functions.
    func_d0 : lamopdified function
        Input function.
    func_d2 : lamopdified function
        Second derivate (curvature) of input function.
    max_nfev : int
        Maximum number of function evaluations.
    err_weights :  array of float, optional
        Error weights assigned to constraints. 
        Enter [1,0,0,0] for standard residual fit on displacement.
        The default is [1,10,100,100].
    x_lB : float
        Cartesian coordinate in x-direction of start of function (curvature=0, p.e. left bearing).
    x_rB : float
        Cartesian coordinate in x-direction of end of function (curvature=0, p.e. right bearing).
    load_dir : string, optional
        Direction of displacement application.
        Possible are:
            - "-y": application in negative y-direction (standard, curvature positve(err2))
            - "+y": application in negative y-direction (curvature negative(err2))

    Returns
    -------
    MG_multi_minimize_Dict : dict
        Fit result dictionary (lmfit-result, parameter-dict, Coefficient of determination for multi-errors and only displacement as well.).

    """
    fit_Result = lmfit.minimize(res_multi_const_weighted, params,
                                args=(x,), kws={'func':func_d0,'func_d2':func_d2,
                                                'x_lB':x_lB, 'x_rB':x_rB,
                                                'func_err_weight': err_weights, 
                                                'load_dir': load_dir,
                                                'data': data},
                                scale_covar=True, max_nfev=max_nfev,
                                nan_policy=nan_policy)
    fit_params_dict = fit_Result.params.valuesdict()
    # Rquad_multi = 1 - fit_Result.residual.var() / np.var(data)
    Rquad_multi = 1 - fit_Result.residual.var() / np.nanvar(data)
    res_disp = res_multi_const_weighted(params=fit_params_dict, x=x,
                                        func=func_d0, func_d2=func_d2,
                                        x_lB=None, x_rB=None,
                                        func_err_weight=[1,0,0,0], 
                                        load_dir=load_dir, data=data)
    # Rquad_disp = 1 - res_disp.var() / np.var(data)
    Rquad_disp = 1 - np.nanvar(res_disp) / np.nanvar(data)
    Multi_minimize_Dict = dict({'Fit_Result': fit_Result, 'Fit_params_dict': fit_params_dict,
                                'Rquad_multi': Rquad_multi, 'Rquad_disp': Rquad_disp})
    return Multi_minimize_Dict

def Perform_Fit(BFL, Fit_func_key, P_df,
                lB, rB, s_range, 
                # Shear_func_key=None, t_mean=None, poisson=0.3, 
                Shear_func_key=None, gamma_V=None,
                err_weights=[ 1, 10, 1000, 100], max_nfev=500, nan_policy='raise',
                option='Pre', ldoption='fixed-y', ldoptionadd=None,
                pb_b=True,**pwargs):
    """
    Performs a weighted multiconstraint least-square-fit, based on lmfit.minimize.

    Parameters
    ----------
    BFL : Bend_func_legion
        DESCRIPTION.
    Fit_func_key : TYPE
        DESCRIPTION.
    P_df : TYPE
        DESCRIPTION.
    lB : TYPE
        DESCRIPTION.
    rB : TYPE
        DESCRIPTION.
    s_range : TYPE
        DESCRIPTION.
    # Shear_func_key : TYPE, optional
        DESCRIPTION. The default is None.
    t_mean : TYPE, optional
        DESCRIPTION. The default is None.
    poisson : TYPE, optional
        DESCRIPTION. The default is 0.3.
    Shear_func_key : TYPE, optional
        DESCRIPTION. The default is None.
    gamma_V : TYPE, optional
        DESCRIPTION. The default is None.
    err_weights : TYPE, optional
        DESCRIPTION. The default is [ 1, 10, 1000, 100].
    max_nfev : TYPE, optional
        DESCRIPTION. The default is 500.
    nan_policy : TYPE, optional
        DESCRIPTION. The default is 'raise'.
    option : TYPE, optional
        DESCRIPTION. The default is 'Pre'.
    ldoption : string, optional
        Load direction automatism.
        possible are:
            - 'fixed-y': Load application in -y-direction (default).
            - 'fixed+y': Load application in +y-direction.
            - 'auto-dispser': Load application direction automaticly determined by a series of displacements (applied as ldoptionadd, index have to match with s_range/P_df).
            - 'auto-Pcoorddisp': Load application direction automaticly determined by a Points dataframe and specified point name and coordinate (applied as ldoptionadd, index have to match with s_range/P_df)
        The default is 'fixed-y'.
    ldoptionadd : Series or array of [Dataframe, string, string], optional
        Addendum for ldoption.
        Have to match to ldoption:
            - 'auto-dispser': Series of displacements (index have to match with s_range/P_df).
            - 'auto-Pcoorddisp': Points dataframe and specified point name and coordinate ([Points as Dataframe, point name as string, coordinate as string],index have to match with s_range/P_df)            
        The default is None.
    pb_b : bool, optional
        Switch for showing progressbar. The default is True.
    **pwargs : dict or pandas.Series
        Parameter keyword arguments for function.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    Fit_res_df : pandas.DatFrame
        Data of fit results.

    """
    
    if not option in ['Pre','Bend']:
                raise NotImplementedError("Option %s not implemented!"%option)
    if not ldoption in ['fixed-y','fixed+y','auto-dispser','auto-Pcoorddisp']:
                raise NotImplementedError("Option for load direction %s not implemented!"%ldoption)
        
    if pb_b: pb = tqdm(s_range, desc =option+" fit: ", unit=' steps', ncols=100)

    Fit_model, Fit_params = lmfit_modelize(BFL[Fit_func_key]['d0'],
                                           option='init')
    Fit_res_df = pd.DataFrame([],index=s_range,columns=['Fit_Result','Fit_params_dict',
                                                        'Rquad_multi','Rquad_disp'],
                              dtype='float64')
    
    for step in s_range:
        if step == s_range[0]:
            # Fit_params.add('xmin', value=lB, vary = False)
            # Fit_params.add('xmax', value=rB, vary = False)
            # if BFL.name=='FSE fit':
            #     Fit_params.add('FP', expr='pi/(xmax-xmin)', vary = False)
            #     Fit_params.add('b1', value=1.0)
            #     Fit_params.add('b2', value=1.0)
            #     Fit_params.add('b3', value=1.0)
            #     Fit_params.add('b4', value=1.0)
            #     if option == 'Pre':
            #         Fit_params.add('c',  value=1.0)
            #         Fit_params.add('d',  value=1.0)
            # elif BFL.name=='P4O fit':
            #     Fit_params.add('a', value=1.0)
            #     Fit_params.add('b', value=1.0)
            #     Fit_params.add('c', value=1.0)
            #     Fit_params.add('d', value=1.0)
            #     Fit_params.add('e', value=1.0)
            # else:
            #     raise NotImplementedError("Fit type not implemented!")
            # if option == 'Pre': Fit_params.add('f_V_0',  value=None, vary = False)
            if BFL.name=='FSE fit':
                # if 'pval' in pwargs.keys():
                #     pval=pwargs['pval']
                # else:
                #     pval = {'xmin':lB,'xmax':rB,'FP':'pi/(xmax-xmin)'}
                # Fit_params=lmfit_param_prep('1st_build_Bfs',
                #                             BFL[Fit_func_key]['d0'], pval)
                if 'param_val' not in pwargs.keys():
                    pwargs.update({'param_val':{'xmin':lB,'xmax':rB,'FP':'pi/(xmax-xmin)'}})
                Fit_params=lmfit_param_prep('1st_build_Bfs',
                                            BFL[Fit_func_key]['d0'], **pwargs)
            else:
                if BFL.name=='P4O fit':
                    Fit_params.add('xmin', value=lB, vary = False)
                    Fit_params.add('xmax', value=rB, vary = False)
                    Fit_params.add('a', value=1.0)
                    Fit_params.add('b', value=1.0)
                    Fit_params.add('c', value=1.0)
                    Fit_params.add('d', value=1.0)
                    Fit_params.add('e', value=1.0)
                else:
                    raise NotImplementedError("Fit type not implemented!")
                if option == 'Pre': Fit_params.add('f_V_0',  value=None, vary = False)
        else:
            # changed 21-09-07 (doesn't work with gaps in s_range)
            step_bf = s_range[s_range.get_indexer_for([step])[0]-1]
            # step_bf = Evac.pd_slice_index(s_range,step-1,option='list')
            # Fit_params = Fit_res_df.loc[step_bf,'Fit_Result'].params
            Fit_params=lmfit_param_prep('old_result',
                                        Fit_res_df.loc[step_bf,'Fit_Result'].params)
        
        if ldoption == 'fixed-y':
            load_dir = '-y'
        elif ldoption == 'fixed+y':
            load_dir = '+y'
        else:
            if ldoption == 'auto-dispser':
                load_dir_set = ldoptionadd.loc[step]
            elif ldoption == 'auto-Pcoorddisp':
                load_dir_set = ldoptionadd[0].loc[step].loc[ldoptionadd[1],ldoptionadd[2]]
            load_dir = '-y' if load_dir_set<=0 else '+y'
        
        x_data = P_df.loc[step].loc[:,'x'].values
        y_data = P_df.loc[step].loc[:,'y'].values
        f_tmp = Multi_minimize(x=x_data, data=y_data, params=Fit_params,
                               func_d0=BFL[Fit_func_key]['d0'].func,
                               func_d2=BFL[Fit_func_key]['d2'].func,
                               max_nfev=max_nfev,  nan_policy=nan_policy,
                               err_weights=err_weights,
                               x_lB=lB, x_rB=rB, load_dir=load_dir)
        Fit_res_df.loc[step] = f_tmp
        # Check boundaries for free paramters
        bccheck = lmfit_bound_checker(f_tmp['Fit_Result'],BFL[Fit_func_key]['d0'])
        if bccheck[0]: warnings.warn(UserWarning(bccheck[-1]))
        if pb_b: pb.update()
    if pb_b: pb.close()
    
    if option =='Pre':    
    # Shear-force-deformation ratio
        f_S_0 = BFL['w_S']['d0'](0.0,Fit_res_df.loc[:,'Fit_params_dict'])
        # Length = lB-rB
        # gamma_V = 2.4*(1+poisson)*t_mean**2/Length**2
        f_V_0 = f_S_0*gamma_V/(1+gamma_V)
        for step in s_range:
            Fit_res_df.loc[step,'Fit_params_dict'].update({'f_V_0':f_V_0.loc[step]})
            
    return Fit_res_df


def params_dict_sub(params_dict, step, dif_step, BFs,
                    param_types_dif=_param_types_fit_or_set):
    """Performs a substraction of parameter values
    with specified type in a parameter dictionary.
    Warning: Seems to be not accurate!"""
    a = params_dict.loc[dif_step]
    b = params_dict.loc[step]
    params_dict_res = copy.deepcopy(b)
    for key in b:
        # if key in ['b1','b2','b3','b4','c','d','f_V_0']:
        if BFs.param_types[key] in param_types_dif:
            params_dict_res[key]=b[key]-a[key]
    return params_dict_res

def params_dict_diff(params_dict_series, step_range, BFs,
                    param_types_dif=_param_types_fit_or_set,
                    option='Fill_NaN'):
    """Performs a differencation of a series of parameter values
    with specified type in a parameter dictionary.
    Warning: Seems to be not accurate!"""
    params_dict_res=pd.Series([],dtype='O')
    for step in step_range:        
        step_bf = step_range[step_range.get_indexer_for([step])[0]-1]
        if step_bf == (step-1):
            params_dict_res.loc[step] = params_dict_sub(params_dict_series, step, step_bf,
                                                        BFs, param_types_dif)
        elif option == 'Next' and step_bf < step:
            params_dict_res.loc[step] = params_dict_sub(params_dict_series, step, step_bf,
                                                        BFs, param_types_dif)
        else:
            params_dict_res.loc[step] = np.nan
    return params_dict_res
#%% Evaluation functions
# def strain_from_curve(x,step,func_curve,params_curve,func_thick,
#                       CS_type='Rectangle',evopt=1/2):
#     if CS_type == 'Rectangle':
#         func_strain = func_curve(x,params_curve.loc[step])*func_thick(x)*evopt
#     else:
#         raise NotImplementedError("Cross-section-type %s not implemented!"%CS_type)
#     return func_strain
# def strain_from_curve2(func_curve,func_thick,
#                       CS_type='Rectangle',evopt=1/2):
#     if CS_type == 'Rectangle':
#         func_strain = lambda x,**kws: func_curve(x,**kws)*func_thick(x)*evopt
#     else:
#         raise NotImplementedError("Cross-section-type %s not implemented!"%CS_type)
#     return func_strain
def straindf_from_curve(x, func_curve, params_curve, func_thick, evopt=1/2):
    """Evaluates the combination of a curvature and thickness function to x
    on a thickness ratio. (Use only for bending)
    Returns a dataframe with steps as index and x as columns."""
    df_strain = func_curve(x,params_curve,
                             coords=None,coords_set=None,col_type='val')*func_thick(x)*evopt
    return df_strain

def Moment_perF_func(x, xmin,xmax, Test_type='TPB'):
    """Returns a by force scaleable moment function."""
    if Test_type == 'TPB':
        MpF_0=(xmax-xmin)/4
        eq=triangle_func_d0(x=x, xmin=xmin,xmax=xmax, f_0=MpF_0)
    else:
        raise NotImplementedError("Test-type %s not implemented!"%Test_type)
    return eq
    
def stress_perF(x, func_MoI, func_thick, xmin,xmax, evopt=1/2, Test_type='TPB'):
    """Returns a by force scaleable bending stress function."""
    stress_pF_data = (lambda b: Moment_perF_func(b, xmin,xmax, Test_type)*func_thick(b)*evopt/func_MoI(b))(x)
    if isinstance(x,float):
        return stress_pF_data
    else:
        # round float point division
        ind=pd.Index(x).map(lambda x: Evac.round_to_sigdig(x,_significant_digits_fpd))
        stress_pF = pd.Series(stress_pF_data,index=ind)
        return stress_pF

def stress_df_from_lin(F, x, func_MoI, func_thick, xmin,xmax, evopt=1/2, Test_type='TPB'):
    """Returns bending stress values according given x coordinates."""
    stress_pF = stress_perF(x, func_MoI, func_thick, xmin,xmax, evopt, Test_type)
    df_stress = pd.DataFrame(stress_pF.values * F.values[:, None],
                             index=F.index, columns=stress_pF.index)
    return df_stress

def Weight_func(x, option='Triangle', c_func=None, **kwargs):
    """
    Returns a weighting function by given options and parameters.

    Parameters
    ----------
    x : float
        Coordinate in x direction.
    option : string, optional
        Choosen option. 
        Possible are:
            - 'Cut': Excluding values outside range of xmin to xmax (weight equal 0).
            - 'Triangle': Weighing to triangle function with maximum in the middle between xmin and xmax.
            - 'Triangle_cut': Mixture of 'Triangle' and 'Cut'.
            - 'Custom': Weighing to custom function (p.e. displacement funtion). 
            - 'Custom_cut': Mixture of 'Custom' and 'Cut'.
        The default is 'Triangle'.
    c_func : function, optional
        Custom function for weighing (p.e. displacement funtion). The default is None.
    **kwargs : dict
        Keyword arguments for custom function (p.e. displacement funtion parameters).

    Raises
    ------
    NotImplementedError
        Option not implemented.

    Returns
    -------
    eq : float
        Weights.

    """
    # if option == 'Triangle':
    #     f = 2/(kwargs['xmax']-kwargs['xmin'])
    #     f = 1
    #     eq = triangle_func_d0(x=x, xmin=kwargs['xmin'], xmax=kwargs['xmax'], f_0=f)
    if option == 'Cut':
        eq = np.where((x>=kwargs['xmin'])&(x<=kwargs['xmax']),1.0,0.0)
    elif option in ['Triangle','Triangle_cut']:
        # f = 2/(kwargs['xmax']-kwargs['xmin'])
        f = 1
        eq = triangle_func_d0(x=x, xmin=kwargs['xmin'], xmax=kwargs['xmax'], f_0=f)
        if option == 'Triangle_cut':
            eq = np.where((x>=kwargs['xmin'])&(x<=kwargs['xmax']),eq,0.0)
    elif option in ['Custom','Custom_cut']:
        eq = c_func(x, kwargs)
        if option == 'Custom_cut':
            eq = np.where((x>=kwargs['xmin'])&(x<=kwargs['xmax']),eq,0.0)
    else:
        raise NotImplementedError("Option %s not implemented!"%option)
    return eq

# def YM_eva_method_A(stress_mid_ser,
#                     strain_mid_ser,
#                     comp=True, name='B', 
#                     det_opt='incremental',**kws):
#     step_range=Evac.pd_combine_index(stress_mid_ser, strain_mid_ser)
#     stress_mid_ser = stress_mid_ser.loc[step_range]
#     strain_mid_ser = strain_mid_ser.loc[step_range]
#     if det_opt=="incremental":
#         YM_ser = stress_mid_ser / strain_mid_ser
#         YM_ser = pd.Series(YM_ser, name=name)
#         return YM_ser
#     elif det_opt=="leastsq":
#         YM, YM_abs, YM_Rquad, YM_fit = Evac.YM_sigeps_lin(stress_mid_ser,
#                                                                 strain_mid_ser, **kws)
#         return YM, YM_abs, YM_Rquad, YM_fit
def YM_eva_method_A(stress_mid_ser,
                    strain_mid_ser,
                    comp=True, name='A', 
                    det_opt='incremental',**kws):
    """
    Calculates Young's Modulus by rise of stress to strain in midspan 
    over defined range with definable method.
    Children of Evac.YM_eva_com_sel.

    Parameters
    ----------
    stress_mid_ser : pd.Series
        Series with stress values in midspan corresponding strain_ser.
    strain_mid_ser : pd.Series
        Series with strain values in midspan corresponding stress_ser.
    comp : boolean, optional
        Compression mode. The default is True.
    name : string, optional
        Name of operation. The default is 'A'.
    det_opt : TYPE, optional
        Definable method for determination.
        Ether incremental or leastsq. The default is 'incremental'.
    **kws : dict
        Keyword dict for least-square determination.

    Returns
    -------
    det_opt == "incremental":
        YM_ser : pd.Series
            Series of Young's Moduli.
    or
    det_opt == "leastsq":
        YM : float
            Youngs Modulus (corresponds to slope of linear fit).
        YM_abs : float
            Stress value on strain origin (corresponds to interception of linear fit).
        YM_Rquad : float
            Coefficient of determination.
        YM_fit : lmfit.model.ModelResult
            Fitting result from lmfit (use with fit.fit_report() for report).
    """
    YM = Evac.YM_eva_com_sel(stress_ser=stress_mid_ser,
                                   strain_ser=strain_mid_ser,
                                   comp=comp, name=name,
                                   det_opt=det_opt, **kws)
    return YM

def YM_eva_method_B(stress_mid_ser,
                    thickness, Length, option="Points", 
                    P_df=None, P_fork_names=None,
                    w_func=None, w_params=None, Length_det=None,
                    comp=True, name='B', det_opt='incremental',**kws):
    """
    Calculates Young's Modulus by rise of stress to strain, 
    calculated by optical counterpart to traditional fork transducer,
    over defined range with definable method.

    Parameters
    ----------
    stress_mid_ser : pandas.Series
        Series with stress values in midspan corresponding strain_ser.
    thickness : np.poly1d function
        Function of thickness to span.
    Length : float
        Length of span.
    option : string, optional
        Input option, ether "Points", for a Dataframe of three points, or
        "Fit", for a fitted bend line.
        The default is "Points".
    P_df : pandas.DataFrame, optional
        Dataframe of Points. The default is None.
    P_fork_names : array of string, optional
        Names of points in P_df in form of [left, mid, right]. The default is None.
    w_func : function, optional
        Function of bending line. The default is None.
    w_params : array or pandas.Series, optional
        Parameters of function of bending line per step. The default is None.
    Length_det : float, optional
        Determination length between left and right point to calculate. The default is None.
    comp : boolean, optional
        Compression mode.The default is True.
    name : string, optional
        Name of operation. The default is 'B'.
    det_opt : TYPE, optional
        Definable method for determination.
        Ether incremental or leastsq. The default is 'incremental'.
    **kws : dict
        Keyword dict for least-square determination.

    Raises
    ------
    ValueError
        Error in combination of option and inputs.
    NotImplementedError
        Error if option not implemented.
        
    Returns
    -------
    det_opt == "incremental":
        YM_ser : pd.Series
            Series of Young's Moduli.
    or
    det_opt == "leastsq":
        YM : float
            Youngs Modulus (corresponds to slope of linear fit).
        YM_abs : float
            Stress value on strain origin (corresponds to interception of linear fit).
        YM_Rquad : float
            Coefficient of determination.
        YM_fit : lmfit.model.ModelResult
            Fitting result from lmfit (use with fit.fit_report() for report).
    """
    if option == "Points":
        if (P_df is None) or (P_fork_names is None) or len(P_fork_names)!=3:
            raise ValueError("Please insert correct values for P_df and P_fork_names!")
        # step_range=Evac.pd_combine_index(stress_mid_ser, P_df)
        step_range=Evac.pd_combine_index(stress_mid_ser, P_df.index)
        f_lr = Point_df_idx(P_df, steps=step_range,
                            points=[P_fork_names[0],P_fork_names[-1]],
                               coords=['y']).mean(axis=1)
        f_c  = Point_df_idx(P_df, steps=step_range,
                            points=[P_fork_names[1]],
                            coords=['y']).mean(axis=1)
        x_lr = Point_df_idx(P_df, steps=step_range,
                            points=[P_fork_names[0],P_fork_names[-1]],
                            coords=['x'])
        Length_det = (x_lr[P_fork_names[-1]]-x_lr[P_fork_names[0]])['x']
    elif option == "Fit":
        if (w_func is None) or (w_params is None) or (Length_det is None):
            raise ValueError("Please insert correct values w_func, w_params and Length_det!")
        step_range=Evac.pd_combine_index(stress_mid_ser, w_params)
        f_lr = w_func(np.array([-Length_det/2, Length_det/2]),
                      w_params.loc[step_range]).mean(axis=1)
        f_c = w_func(0.0, w_params.loc[step_range])
    else:
        raise NotImplementedError("Option %s not implemented!"%option)
        
    f = (f_c - f_lr) * (-1 if comp else 1)
    stress_mid_ser = stress_mid_ser.loc[step_range]
    strain_ser = (12 * f * thickness * Length) / (3 * Length * Length_det**2 - Length_det**3)
    strain_ser = pd.Series(strain_ser, name=name)
    if det_opt=="incremental":
        YM_ser = stress_mid_ser / strain_ser
        YM_ser = pd.Series(YM_ser, name=name)
        return YM_ser, strain_ser
    elif det_opt=="leastsq":
        if stress_mid_ser.isna().all() or strain_ser.isna().all():
            YM, YM_abs, YM_Rquad, YM_fit = np.nan,np.nan,np.nan,np.nan
        else:
            YM, YM_abs, YM_Rquad, YM_fit = Evac.YM_sigeps_lin(stress_mid_ser, strain_ser, **kws)
        return YM, YM_abs, YM_Rquad, YM_fit, strain_ser
        

def YM_eva_method_E(Force_ser, length,
                    func_curve, params_curve,
                    func_MoI, func_thick, evopt=1/2,
                    opt_det='all', n=100,
                    opt_g='length', opt_g_lim=0.5,
                    weight_func=Weight_func,
                    wargs=[], wkwargs={}, name='E'):
    """
    Calculates Young's Modulus by rise of stress to strain, 
    calculated by curvature (2nd derivate) of bending line,
    per step with definable method.

    Parameters
    ----------
    Force_ser : pandas.Series
        Series of force increments.
    length : float
        Length of span.
    func_curve :  function
        Function of curvature (2nd derivate) of bending line. The default is None.
    params_curve : array or pandas.Series of dictionionaries
        Parameters of function of curvature (2nd derivate) of bending line per step..
    func_MoI : np.poly1d function
        Function of Moment of inertia to position on span (x-direction).
    func_thick : np.poly1d function
        Function of thickness to position on span (x-direction).
    evopt : float, optional
        Position according thickness for strain calculation. The default is 1/2.
    opt_det : string, optional
        Determination option.
        possible are:
            - 'all': All options evaluated.
            - 'stress' : Local evaluation on maximum stress.
            - 'strain' : Local evaluation on maximum strain.
            - 'range' : Global evaluation over range defined by opt_g.
        The default is 'all'.
    n : integer, optional
        Division of the length for calculation. The default is 100.
    opt_g : string, optional
        Option for global determination.
        possible are:
            - 'length': Averaging by range (opt_g_lim*length) around midspan.
            - 'strain': Averaging by range (strain<=opt_g_lim*strain_max).
            - 'Moment_weighted': Weighted averaging by moment function.
            - 'Custom_weighted': Weighted averaging by custom function.
        The default is 'length'.
    opt_g_lim : float, optional
        Limiter for opt_g. The default is 0.5.
    weight_func : function, optional
        Weighting function for global averaging. The default is Weight_func.
    wargs : array, optional
        Arguments for weighting function for global averaging. The default is [].
    wkwargs : dictionary, optional
        Keyword arguments for weighting function for global averaging. The default is {}.
    name : string, optional
        Name of operation. The default is 'E'.

    Raises
    ------
    NotImplementedError
        Raise error if option not implemented.

    Returns
    -------
    E_df : pandas.DataFrame
        Dataframe of determined Young's Moduli.
    sig_eps_df : pandas.DataFrame
        Dataframe of stress and strain values used in determination.
    E_to_x : TYPE
        Dataframe of determined Young's Moduli to span position (x-direction).
    stress_df : pandas.DataFrame
        Dataframe of stress values.
    strain_df : pandas.DataFrame
        Dataframe of strain values.
    E_to_x_g : pandas.DataFrame
        Dataframe of determined Young's Moduli to span position (x-direction) for global averaging.

    """
            
    # 26.10.22 - unload (negative force increment) added
    def MaxMinboSer(po,oths,k='Max',axis=1):
        gax = Evac.pd_axischange(axis)
        # po_sign   = po.apply(np.sign)
        oths_sign = oths.apply(np.sign)
        po_osign = po.mul(oths_sign,axis=gax)
        if k == 'max':
            out=po_osign.max(axis=axis).mul(oths_sign,axis=gax)
        elif k == 'idxmax':
            out=po_osign.idxmax(axis=axis)
        elif k == 'min':
            out=po_osign.min(axis=axis).mul(oths_sign,axis=gax)
        elif k == 'idxmin':
            out=po_osign.idxmin(axis=axis)
        else:
            raise NotImplementedError("Option %s not implemented!"%k)
        return out
    
    if not opt_det in ['all', 'stress', 'strain', 'range']:
        raise NotImplementedError("Option %s type not implemented!"%opt_det)
        
    step_range=Evac.pd_combine_index(Force_ser, params_curve)
    xlin = np.linspace(-length/2,length/2,n+1)
    
    stress_df = stress_df_from_lin(F=Force_ser, x=xlin,
                                   func_MoI=func_MoI, func_thick=func_thick,
                                   xmin=-length/2, xmax=length/2)
    stress_df = stress_df.loc[step_range]
    strain_df = straindf_from_curve(xlin, func_curve, params_curve, func_thick, evopt)
    strain_df = strain_df.loc[step_range]
    
    E_to_x = stress_df / strain_df
    
    if opt_det in ['all', 'stress', 'strain']:
                  
        # stress_max   = pd.Series(stress_df.max(axis=1), name='stress_max')
        # stress_max_x = pd.Series(stress_df.idxmax(axis=1), name='stress_max_x')
        # 26.10.22 - unload (negative force increment) added
        stress_max   = pd.Series(MaxMinboSer(stress_df,Force_ser,'max',1),
                                 name='stress_max')
        stress_max_x = pd.Series(MaxMinboSer(stress_df,Force_ser,'idxmax',1),
                                 name='stress_max_x')
        
        # Krücke, muss besser über Direktindizierung gehen
        E_stress_max = pd.Series([],dtype='float64', name='E_stress_max')
        stress_max_strain = pd.Series([],dtype='float64', name='stress_max_strain')
        for step in step_range: #Krücke da Force ab 'S' 
            E_stress_max.loc[step]      = E_to_x.loc[step,stress_max_x.loc[step]]
            stress_max_strain.loc[step] = strain_df.loc[step,stress_max_x.loc[step]]
        # E_stress_max = pd.Series(E_stress_max, name='E_stress_max')
        E_stress_max = pd.Series(E_stress_max, name=name+'l')
            
    if opt_det in ['all', 'strain']:
    
        # strain_max   = pd.Series(strain_df.max(axis=1), name='strain_max')
        # strain_max_x = pd.Series(strain_df.idxmax(axis=1), name='strain_max_x')
        # 26.10.22 - unload (negative force increment) added
        strain_max   = pd.Series(MaxMinboSer(strain_df,Force_ser,'max',1),
                                 name='strain_max')
        strain_max_x = pd.Series(MaxMinboSer(strain_df,Force_ser,'idxmax',1),
                                 name='strain_max_x')
        
        # Krücke, muss besser über Direktindizierung gehen
        E_strain_max = pd.Series([],dtype='float64', name='E_strain_max')
        strain_max_stress = pd.Series([],dtype='float64', name='strain_max_stress')
        for step in step_range: #Krücke da Force ab 'S' 
            E_strain_max.loc[step]      = E_to_x.loc[step,strain_max_x.loc[step]]
            strain_max_stress.loc[step] = stress_df.loc[step,strain_max_x.loc[step]]
        # E_strain_max = pd.Series(E_strain_max, name='E_strain_max')
        E_strain_max = pd.Series(E_strain_max, name=name+'e')

    if opt_det in ['all', 'range']:
        if opt_g == 'length':
            cols = Evac.pd_slice_index(E_to_x.columns,
                                             [xlin.min()*opt_g_lim, 
                                              xlin.max()*opt_g_lim])
            E_to_x_g = E_to_x.loc(axis=1)[cols]
            E_to_x_g_mean = E_to_x.loc[step_range,cols].mean(axis=1)
        elif opt_g == 'strain':
            test=pd.DataFrame([],columns=strain_df.columns)
            for step in strain_df.index:
                # test.loc[step]=strain_df.loc[step] >= opt_g_lim*strain_df.max(axis=1).loc[step]
                # 26.10.22 - unload (negative force increment) added
                if Force_ser[step]>=0:
                    test.loc[step]=strain_df.loc[step] >= opt_g_lim*strain_df.max(axis=1).loc[step]
                else:
                    test.loc[step]=strain_df.loc[step] <= opt_g_lim*strain_df.min(axis=1).loc[step]
            E_to_x_g = E_to_x.copy()[test]
            E_to_x_g_mean = E_to_x_g.mean(axis=1)
        elif opt_g == 'Moment_weighted':
            cols = Evac.pd_slice_index(E_to_x.columns,
                                             [xlin.min()*opt_g_lim,
                                              xlin.max()*opt_g_lim])
            E_to_x_g = E_to_x.loc(axis=1)[cols]
            kws={'xmin':cols.min(),'xmax':cols.max()}
            mask=np.isnan(E_to_x_g)
            weights=np.array([Weight_func(x=cols,option='Triangle',**kws), ] * E_to_x_g.shape[0], dtype=np.float64)
            mw = np.ma.MaskedArray(weights, mask=mask)
            ma = np.ma.MaskedArray(E_to_x_g, mask=mask)
            E_to_x_g_mean = np.ma.average(a=ma,axis=1,weights=mw).filled(np.nan)
            # E_to_x_g_mean = np.average(a=E_to_x_g, axis=1,
            #                            weights=Weight_func(x=cols,option='Triangle',**kws))
            E_to_x_g_mean = pd.Series(E_to_x_g_mean, index=E_to_x.index)
        elif opt_g == 'Custom_weighted':
            cols = Evac.pd_slice_index(E_to_x.columns,
                                             [xlin.min()*opt_g_lim,
                                              xlin.max()*opt_g_lim])
            E_to_x_g = E_to_x.loc(axis=1)[cols]
            mask=np.isnan(E_to_x_g)
            if (type(wkwargs) is pd.core.series.Series):
                weights=weight_func(E_to_x_g.columns,wkwargs.loc[step_range])
            elif (type(wkwargs) is dict):
                weights=np.array([weight_func(E_to_x_g.columns,**wkwargs), ] * E_to_x_g.shape[0], dtype=np.float64)
            else:
                raise NotImplementedError("Type %s of wkwargs not implemented!"%type(wkwargs))
            mw = np.ma.MaskedArray(weights, mask=mask)
            ma = np.ma.MaskedArray(E_to_x_g, mask=mask)
            E_to_x_g_mean = np.ma.average(a=ma,axis=1,weights=mw).filled(np.nan)
            E_to_x_g_mean = pd.Series(E_to_x_g_mean, index=E_to_x.index)
        else:
            raise NotImplementedError("Option-global %s type not implemented!"%opt_g)
        # E_to_x_g_mean = pd.Series(E_to_x_g_mean, name='E_global_' + opt_g)
        # E_to_x_g_mean = pd.Series(E_to_x_g_mean, name='E_global')
        E_to_x_g_mean = pd.Series(E_to_x_g_mean, name=name+'g')
    else:
        E_to_x_g=[]    
        
    if opt_det == 'all':   
        E_df = pd.DataFrame([E_stress_max,E_strain_max,E_to_x_g_mean]).T
        sig_eps_df = pd.DataFrame([stress_max,stress_max_x,stress_max_strain,
                                   strain_max,strain_max_x,strain_max_stress]).T
    elif opt_det == 'stress':   
        E_df = pd.DataFrame([E_stress_max]).T
        sig_eps_df = pd.DataFrame([stress_max,stress_max_x,stress_max_strain]).T 
    elif opt_det == 'strain':  
        E_df = pd.DataFrame([E_strain_max]).T
        sig_eps_df = pd.DataFrame([strain_max,strain_max_x,strain_max_stress]).T
    elif opt_det == 'range':   
        E_df = pd.DataFrame([E_to_x_g_mean]).T
        sig_eps_df = pd.DataFrame([]).T
        
    return E_df, sig_eps_df, E_to_x, stress_df, strain_df, E_to_x_g
    
def YM_eva_method_G(Force_ser,
                     w_func_f_0, w_params,
                     c_func, r_func,
                     c_params, r_params,
                     length, I_func, A_func=None,
                     CS_type='Rectangle', kappa=None, poisson=0.3,
                     comp=True, option="M+V", name="G"):
    """
    Calculates Young's Moduli via the approach of equality of external work 
    and deformation energy.

    Parameters
    ----------
    Force_ser : pandas.Series
        Series of force increments.
    w_func_f_0 : function
        Function of bending line for external work calculation.
    w_params : pandas.Series of dictionionaries
        Parameters of function of bending line per step for external work calculation.
    c_func : function
        Function of curvature (2nd derivate) of bending line.
    r_func : function
        Function of rise (1st derivate) of bending line.
    c_params : pandas.Series of dictionionaries
        Parameters of function of curvature (2nd derivate) of bending line per step.
    r_params : pandas.Series of dictionionaries
        Parameters of function of rise (1st derivate) of bending line per step.
    length : float
        Length of span.
    I_func : np.poly1d function
        Function of Moment of inertia to position on span (x-direction).
    A_func : np.poly1d function, optional
        Function of Area to position on span (x-direction). The default is None.
    CS_type : string, optional
        Cross-section type. The default is 'Rectangle'.
    kappa : float, optional
         Correction factor shear area. Depending to CS_type. The default is None.
    poisson : float, optional
        Poisson's ratio. The default is 0.3.
    comp : boolean, optional
        Compression mode.The default is True.
    option : string, optional
        Determiantion option. Possible are 'ignore_V' and 'M+V'. The default is "M+V".
    name : string, optional
        Name of operation. The default is "G".

    Returns
    -------
    pandas.Series
        Series of determined Young's Moduli.

    """
    step_range = Evac.pd_combine_index(Force_ser, w_params)
    
    def W_ext(Force_ser, f_0, comp=True):
        W_ext = (-1 if comp else 1) * 1/2 * Force_ser * f_0
        return W_ext
    f_0 = w_func_f_0(0.0, w_params.loc[step_range])
    Wext = W_ext(Force_ser.loc[step_range], f_0, comp)

    def W_int_integrant(w_func, w_params, CS_func):
        Wint_I_func = lambda s,x: w_func(x,w_params.loc[s])**2*CS_func(x)
        return Wint_I_func
    
    if option == "ignore_V":
        Wint_I = W_int_integrant(c_func, c_params, I_func)
        Wint = pd.Series([],dtype='float64',name=name)
        for step in step_range:
            Wint.loc[step]=1/2 * scint.quad(lambda x: Wint_I(x=x,s=step),-length/2,length/2)[0]
    elif option == "M+V":
        Wint_I_M = W_int_integrant(c_func, c_params, I_func)
        Wint_M = pd.Series([],dtype='float64',name=name)
        func_AS = Shear_area(A_func, CS_type, kappa)
        Wint_I_V = W_int_integrant(r_func, r_params, func_AS)
        Wint_V = pd.Series([],dtype='float64',name=name)
        for step in step_range:
            Wint_M.loc[step]=1/2 * scint.quad(lambda x: Wint_I_M(x=x,s=step),-length/2,length/2)[0]
            Wint_V.loc[step]=1/2 * 1/(2*(1+poisson))*scint.quad(lambda x: Wint_I_V(x=x,s=step),-length/2,length/2)[0]
        Wint=Wint_M + Wint_V
    E_ser = pd.Series(Wext/Wint,name=name)
    return E_ser

def YM_eva_method_F(c_func, c_params,
                    Force_ser,
                    Length, func_I,
                    weighted=True, weight_func=Weight_func,
                    wargs=[], wkwargs={},
                    xr_dict = {'fu':1/1, 'ha':1/2, 'th':1/3},
                    pb_b=True, name='F', n=100):
    """
    Calculates the Young's modulus via the local application of 
    the differential equation of the bending line.

    Parameters
    ----------
    c_func : function
        Function of curvature (2nd derivate) of bending line.
    c_params : pandas.Series of dictionionaries
        Parameters of function of curvature (2nd derivate) of bending line per step.
    Force_ser : pandas.Series
        Series of force increments.
    Length : float
        Length of span.
    func_I : Tnp.poly1d function
        Function of Moment of inertia to position on span (x-direction).
    weighted : boolean, optional
        Switch for weighted averaging. The default is True.
    weight_func : function, optional
        Weighting function for global averaging. The default is Weight_func.
    wargs : array, optional
        Arguments for weighting function for global averaging. The default is [].
    wkwargs : dictionary, optional
        Keyword arguments for weighting function for global averaging. The default is {}.
    xr_dict : dictionary, optional
        Dictionary of name to range of length around midspan for determination.
        The default is {'fu':1/1, 'ha':1/2, 'th':1/3}.
    pb_b : boolean, optional
        Switch of progressbar. The default is True.
    name : string, optional
        Name of operation. The default is 'F'.
    n : integer, optional
        Division of the length for calculation. The default is 100.

    Returns
    -------
    YM_df : pandas.Series
        Series of determined Young's Moduli.

    """
    def E_YM(c_func, c_params, I_func, Length):
        YM = lambda s,x: Moment_perF_func(x, -Length/2, Length/2)*Force_ser.loc[s]/(c_func(x,c_params.loc[s])*I_func(x))
        return YM 
    
    step_range = Evac.pd_combine_index(Force_ser, c_params)    
    if pb_b: pb = tqdm(step_range, desc =name+": ", unit=' steps', ncols=100)
    E_YM_func = E_YM(c_func, c_params, func_I, Length)
    
    YM_df = pd.DataFrame([],columns=xr_dict.keys(),dtype='float64')
    xlin  = pd.DataFrame([],columns=xr_dict.keys(),dtype='float64')
    for k in xr_dict:
        xlin[k] = np.linspace(-Length*xr_dict[k]/2,Length*xr_dict[k]/2,n+1)
    for step in step_range:
        YM_wei = pd.DataFrame([],columns=xr_dict.keys(),dtype='float64')
        YM_tmp = pd.DataFrame([],columns=xr_dict.keys(),dtype='float64')
        for k in xr_dict: 
            if weighted:
                if (type(wkwargs) is pd.core.series.Series):
                    YM_wei[k] = weight_func(xlin[k],*wargs,**wkwargs.loc[step])
                else:
                    YM_wei[k] = weight_func(xlin[k],*wargs,**wkwargs)
            else:
                # YM_wei[k] = None
                YM_wei[k] = np.array([1,]*xlin[k].shape[0])
            YM_tmp[k] = E_YM_func(s=step, x=xlin[k])
            ind_E       = np.where(np.logical_not(np.isnan(YM_tmp[k])))[0]
            YM_df.at[step,k] = np.average(YM_tmp.loc[ind_E,k],
                                          weights=YM_wei.loc[ind_E,k])
        if pb_b: pb.update()
    if pb_b: pb.close()
    # YM_df.columns=np.array([name+'_'+k for k in xr_dict.keys()])
    YM_df.columns=np.array([name+k for k in xr_dict.keys()])
    return YM_df

def YM_eva_method_C(Force_ser, w_func, w_params,
                     length, I_func, A_func=None,
                     CS_type='Rectangle', kappa=None, poisson=0.3,
                     comp=True, option="M+V", name="C"):
    """
    Calculates the Young's modulus from the deformation energy as a
    function of the internal forces (pure geometric fitting function).

    Parameters
    ----------
    Force_ser : pandas.Series
        Series of force increments.
    w_func : function
        Function of bending line.
    w_params : pandas.Series of dictionionaries
        Parameters of function  of bending line per step.
    length : float
        Length of span.
    I_func : np.poly1d function
        Function of Moment of inertia to position on span (x-direction).
    A_func : np.poly1d function, optional
        Function of Area to position on span (x-direction)..
    CS_type : string, optional
        Cross-section type. The default is 'Rectangle'.
    kappa : float, optional
         Correction factor shear area. Depending to CS_type. The default is None.
    poisson : float, optional
        Poisson's ratio. The default is 0.3.
    comp : boolean, optional
        Compression mode.The default is True.
    option : string, optional
        Determiantion option. Possible are 'M' and 'M+V'. The default is "M+V".
    name : string, optional
        Name of operation. The default is "F".

    Returns
    -------
    E_ser : pandas.Series
        Series of determined Young's Moduli.

    """
    step_range = Evac.pd_combine_index(Force_ser, w_params)
    func_AS = Shear_area(A_func, CS_type, kappa)
    F_Alpha = length**2/8*scint.quad(lambda x: (1-abs(x*2/length))**2/I_func(x),-length/2,length/2)[0]
    if option == "M+V":
        F_Beta  = (1+poisson)*scint.quad(lambda x: 1/func_AS(x),-length/2,length/2)[0]
    else:
        F_Beta=0
    f_0   = (-1 if comp else 1)*(w_func(0.0,w_params.loc[step_range]))
    F_gamma = Force_ser.loc[step_range] / (2 * f_0)
    E_ser = pd.Series(F_gamma * (F_Alpha + F_Beta), name=name)
    return E_ser

def YM_eva_method_D_bend_df(Length, I_func, n=100, E=1, F=1):
    """
    Calculates the deflection values via the determined integral of the bending line.

    Parameters
    ----------
    Length :  float
        Length of span.
    I_func : np.poly1d function
        Function of Moment of inertia to position on span (x-direction).
    n : integer, optional
        Division of the length for calculation. The default is 100.
    E : float, optional
        Young's Modulus. The default is 1.
    F : float, optional
        Force. The default is 1.

    Returns
    -------
    m : pandas.DataFrame
        Dataframe of deflections and parts of partwise integration to x-positions.

    """
    def sidewise_df(x, Length, F, E , I_func, side='l'):
        if side == 'l':
            m_df=pd.DataFrame(data=None,index=-x)
        elif side == 'r':
            m_df=pd.DataFrame(data=None,index= x)
        else:
            raise NotImplementedError("Side %s not implemented, only l and r allowed!"%side)
        m_df['M'] = -F * Moment_perF_func(x=m_df.index, xmin=-Length/2,xmax=Length/2, Test_type='TPB')
        m_df['I'] = I_func(m_df.index)
        m_df['Quotient'] = m_df.loc[:,'M'] / (E * m_df.loc[:,'I'])
        j = 0
        for i in m_df.index:
            if i==0:
                m_df.loc[i,'Integral_1'] = 0
                m_df.loc[i,'Integral_2'] = 0
                m_df.loc[i,'Omega']      = 0
                j = i
            else:
                # m_df.loc[i,'Integral_1'] = m_df.loc[j,'Integral_1'] + m_df.loc[[i,j],'Quotient'].mean() * Length/2/(n/2)
                # m_df.loc[i,'Integral_2'] = m_df.loc[j,'Integral_2'] + m_df.loc[[i,j],'Integral_1'].mean() * Length/2/(n/2)
                m_df.loc[i,'Integral_1'] = m_df.loc[j,'Integral_1'] + m_df.loc[[i,j],'Quotient'].mean() * abs((i-j))
                m_df.loc[i,'Integral_2'] = m_df.loc[j,'Integral_2'] + m_df.loc[[i,j],'Integral_1'].mean() * abs((i-j))
                m_df.loc[i,'Omega']      = -m_df.loc[i,'Integral_2']
                j = i
        return m_df
    
    x = pd.Float64Index(np.linspace(0,Length/2,int(n/2)+1),name='x')
    _significant_digits_fpd=12
    x=x.map(lambda x: Evac.round_to_sigdig(x,_significant_digits_fpd))
    m_l = sidewise_df(x, Length, F, E, I_func, side='l')
    m_r = sidewise_df(x, Length, F, E, I_func, side='r')
    
    RHS     = -np.array([m_l.iloc[-1].loc['Omega'], m_r.iloc[-1].loc['Omega']])        
    Kmat    = np.array([[1,-Length/2],[1,Length/2]])
    Kmatinv = np.linalg.inv(Kmat)
    mid     = Kmatinv.dot(RHS)
    
    m = m_l.iloc[1:].append(m_r).sort_index()
    m['w']   = mid[0] + mid[1] * m.index + m.loc[:,'Omega']
    m['wi']  = mid[1] - m.loc[:,'Integral_1']*np.sign(m.index)
    m['wii'] = -m.loc[:,'Quotient']
    return m

def YM_eva_method_D_bend_df_add(points_x, m_df, Length, I_func, E=1, F=1):
    """
    Adds additional points to the calculated deflection values via the
    determined integral of the bending line.

    Parameters
    ----------
    points_x : pd.Series
        X-coordinates of additional points.
    m_df : pandas.DataFrame
        Dataframe of deflections and parts of partwise integration to x-positions.
    Length : float
        Length of span.
    I_func : np.poly1d function
        Function of Moment of inertia to position on span (x-direction).
    E : float, optional
        Young's Modulus. The default is 1.
    F : float, optional
        Force. The default is 1.

    Returns
    -------
    p_df : pandas.DataFrame
        Dataframe of deflections and parts of partwise integration to x-positions.

    """
    x = pd.Float64Index(points_x,name='x')
    p_df=pd.DataFrame(data=None,index=x)
    
    p_df['M'] = -F * Moment_perF_func(x=x, xmin=-Length/2,xmax=Length/2, Test_type='TPB')
    p_df['I'] = I_func(x)
    p_df['Quotient'] = p_df.loc[:,'M'] / (E * p_df.loc[:,'I'])
    for i in points_x:
        if i<0.0:
            method='bfill'
        else:
            method='ffill'
        j=m_df.index[m_df.index.get_loc(i,method=method)]
        if i==0.0:
            p_df.loc[i,'Integral_1'] = 0
            p_df.loc[i,'Integral_2'] = 0
            p_df.loc[i,'Omega']      = 0
        else:
            p_df.loc[i,'Integral_1'] = m_df.loc[j,'Integral_1'] + np.mean([m_df.loc[j,'Quotient'],p_df.loc[i,'Quotient']]) * abs(i-j)
            p_df.loc[i,'Integral_2'] = m_df.loc[j,'Integral_2'] + np.mean([m_df.loc[j,'Integral_1'],p_df.loc[i,'Integral_1']]) * abs(i-j)
            p_df.loc[i,'Omega']      = -p_df.loc[i,'Integral_2']
    
    p_df['w']   = m_df.loc[0.0,'w'] + m_df.loc[0.0,'wi'] * p_df.index + p_df.loc[:,'Omega']
    p_df['wi']  = m_df.loc[0.0,'wi'] - p_df.loc[:,'Integral_1']*np.sign(p_df.index)
    p_df['wii'] = -p_df.loc[:,'Quotient']
    return p_df

def YM_eva_method_D_res(E, x_data, y_data=None, weights=None, **kws):
    m_df=YM_eva_method_D_bend_df(kws['Length'], kws['I_func'], kws['n'], E, kws['F'])
    p_df=YM_eva_method_D_bend_df_add(x_data, m_df,kws['Length'], kws['I_func'], E, kws['F'])
    if y_data is None:
        err=p_df['w']
    elif weights is None:
        err=p_df['w']-y_data
    else:
        err=(p_df['w']-y_data)*weights
    return err

def YM_eva_method_D_num(P_df, Force_ser, step_range,
                         Length, func_I,
                         weighted=True, weight_func=Weight_func,
                         wargs=[], wkwargs={}, max_nfev=500,
                         pb_b=True, name='D'):
   
    D_df = pd.DataFrame([],columns=['E','Rquad'],dtype='float64')
    if pb_b: pb = tqdm(step_range, desc =name+": ", unit=' steps', ncols=100)
    for step in step_range:
        mo=lmfit.Model(YM_eva_method_D_res, independent_vars=['x_data'])
        mo.name='YM_eva_method_D'
        par = mo.make_params()
        par.add('E', value=500.0, min=0.0, max=np.inf)
        
        F=Force_ser.loc[step]
        x_data = P_df.loc[step].loc[:,'x'].values
        y_data = P_df.loc[step].loc[:,'y'].values
        if weighted:
            if (type(wkwargs) is pd.core.series.Series):
                weights = weight_func(x_data,*wargs,**wkwargs.loc[step])
            else:
                weights = weight_func(x_data,*wargs,**wkwargs)
        else:
            weights = None
        fit_Result = lmfit.minimize(YM_eva_method_D_res, par,
                                    args=(x_data,), kws={'I_func': func_I, 'n':100,
                                                         'Length': Length, 'F': F, 
                                                         'y_data': y_data, 
                                                         'weights': weights},
                                    scale_covar=True, max_nfev=max_nfev)
        D_df.loc[step,'E'] = fit_Result.params['E'].value
        D_df.loc[step,'Rquad'] = 1 - fit_Result.residual.var() / np.var(y_data)
        if pb_b: pb.update()
    if pb_b: pb.close()
    return D_df

def YM_eva_method_D(P_df, Force_ser,
                    Length, func_I, n=100, rel_steps = None,
                    weighted=False, weight_func=Weight_func,
                    wkwargs={}, wargs=[], pb_b=True, name='D'):
    """
    Calculates the modulus of elasticity by matching the theoretical displacements
    from the determined integral of the bending line with the measured ones.

    Parameters
    ----------
    P_df : pandas.DataFrame
        Dataframe of measured points to steps.
    Force_ser : pandas.Series
        Series of force increments.
    Length : float
        Length of span.
    func_I : np.poly1d function
        Function of Moment of inertia to position on span (x-direction).
    n : integer, optional
        Division of the length for calculation. The default is 100.
    rel_steps : index or array, optional
        Relevant steps for determination. The default is None.
    weighted : boolean, optional
        Switch for weighted averaging. The default is True.
    weight_func : function, optional
        Weighting function for global averaging. The default is Weight_func.
    wargs : array, optional
        Arguments for weighting function for global averaging. The default is [].
    wkwargs : dictionary, optional
        Keyword arguments for weighting function for global averaging. The default is {}.
    pb_b : boolean, optional
        Switch of progressbar. The default is True.
    name : string, optional
        Name of operation. The default is 'D'.

    Returns
    -------
    YM_ser : pandas.Series
        Series of determined Young's Moduli.
    YM_df : pandas.DataFrame
        DataFrame of determined Young's Moduli to x-position.

    """
    m = YM_eva_method_D_bend_df(Length=Length, I_func=func_I, n=n, E=1, F=1)
    import warnings
    # ignore NaN-multiply
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    
    # step_range = Evac.pd_combine_index(Force_ser, P_df)
    step_range = Evac.pd_combine_index(Force_ser, 
                                       P_df.loc[np.invert(P_df.isna().all(axis=1))].index)
    
    if not rel_steps is None: 
        step_range = Evac.pd_combine_index(rel_steps, step_range)
    
    if weighted:
        if (type(wkwargs) is pd.core.series.Series):
             step_range = Evac.pd_combine_index(step_range,
                                                      wkwargs)
        
    YM_ser = pd.Series([],dtype='float64')
    YM_df = pd.DataFrame([],columns=P_df.columns.droplevel(1).drop_duplicates(),
                         dtype='float64')
    if pb_b: pb = tqdm(step_range, desc =name+": ", unit=' steps', ncols=100)
    for step in step_range:
        F = Force_ser.loc[step]
        # x_data = P_df.loc[step].loc[:,'x'].values
        # y_data = P_df.loc[step].loc[:,'y'].values
        x_data = P_df.loc[step].loc[:,'x']
        y_data = P_df.loc[step].loc[:,'y']
        # p = YM_eva_method_D_bend_df_add(points_x=x_data, m_df=m,
        #                                  Length=Length, I_func=func_I,
        #                                  E=1, F=1)
        p = YM_eva_method_D_bend_df_add(points_x=x_data.dropna(), m_df=m,
                                         Length=Length, I_func=func_I,
                                         E=1, F=1)
        D_gamma = F/y_data
        D_alpha = (x_data/Length-0.5)*m['Omega'].iloc[0]
        D_beta  = (x_data/Length+0.5)*m['Omega'].iloc[-1]
        # YM_df.loc[step]=F/y_data*((x_data/Length-0.5)*m['Omega'].iloc[0]-(x_data/Length+0.5)*m['Omega'].iloc[-1]+p.loc[x_data,'Omega'].values)
        # YM_df.loc[step] = D_gamma*(D_alpha-D_beta+p.loc[x_data,'Omega'].values)
        D_omega = pd.Series([], dtype='float64')
        for i in x_data.index:
            if np.isnan(x_data[i]):
                D_omega[i] = np.nan
            else:
                D_omega[i] = p.loc[x_data[i],'Omega']
        YM_df.loc[step] = D_gamma*(D_alpha-D_beta+D_omega)
        if weighted:
            if (type(wkwargs) is pd.core.series.Series):
                weights = weight_func(x_data,*wargs,**wkwargs.loc[step])
            else:
                weights = weight_func(x_data,*wargs,**wkwargs)
            # ind_E = np.where(np.logical_not(np.isnan(YM_tmp)))[0]
            ind_E = np.invert((YM_df.loc[step].isna()) | (YM_df.loc[step] == 0))
            YM_ser.at[step] = np.average(YM_df.loc[step][ind_E],
                                         weights=weights[ind_E])
        else:
            weights = None
            # ind_E = np.where(np.logical_not(np.isnan(YM_tmp)))[0]
            ind_E = np.invert((YM_df.loc[step].isna()) | (YM_df.loc[step] == 0))
            YM_ser.at[step] = YM_df.loc[step][ind_E].mean()
        YM_ser =pd.Series(YM_ser,name=name)
        if pb_b: pb.update()
    if pb_b: pb.close()
    return YM_ser, YM_df

def YM_check_with_method_D(E, F, Length, I_func, w_vgl_df, pb_b=True, name='X'):
    step_range = Evac.pd_combine_index(E, w_vgl_df)
    m = YM_eva_method_D_bend_df(Length=w_vgl_df.columns.max()-w_vgl_df.columns.min(),
                                 I_func=I_func,
                                 n=len(w_vgl_df.columns)-1,
                                 E=1, F=1)
    """
    Compares the deformation of the analytical bending line, 
    scaled by Young's Modulus determined by methods, with the measured bending line.
    """
    D_gamma = F.loc[step_range]/E.loc[step_range]
    D_alpha = (m.index/Length-0.5)*m['Omega'].iloc[0]
    D_beta  = (m.index/Length+0.5)*m['Omega'].iloc[-1]
    D_sigma = (D_alpha-D_beta+m['Omega'])
    # w_D_to_E = D_gamma*(D_alpha-D_beta+m['Omega'])
    # w_D_to_E = pd.DataFrame([],index=D_gamma.index,columns=m.index,dtype='float64')
    # for step in D_gamma.index:
    #     w_D_to_E.loc[step] = D_gamma.loc[step]*D_sigma
    tmp = np.array([D_gamma.values,]*D_sigma.shape[0]).transpose()
    w_D_to_E = pd.DataFrame(tmp*D_sigma.values,
                            index=D_gamma.index,
                            columns=D_sigma.index)
    check_E = (w_D_to_E.loc[step_range]-w_vgl_df.loc[step_range])/w_vgl_df.loc[step_range]
    return check_E, w_D_to_E

def coord_df_mean(df, name='', fex=1,lex=1):
    """Return mean values."""
    ser_m=pd.Series(df.iloc(axis=1)[fex:-lex].mean(axis=1), name=name)
    return ser_m
def coord_df_depo(df, name='', pos=0.0):
    """Return values on postion (Default x=0.0)"""
    ser_m=pd.Series(df[pos], name=name)
    return ser_m

def YM_check_many_with_method_D(E_dict, F, Length, I_func,
                                 w_func, w_params, rel_steps=None, n=100,
                                 pb_b=True, name='X'):
    """
    Compares the deformation of the analytical bending line, 
    scaled by Young's Modulus determined by methods, with the measured bending line.
    Returns global mean, local (midspan) mean and complete (n-positions) deviation,
    as well as the scaled analytical bending line, for each specified step.

    Parameters
    ----------
    E_dict : dict
        Dictionary of moduli of elasticity by determination method.
    F : pandas.Series
        Series of force (or force increment).
    Length : float
        Testing length.
    I_func : TYPE
        Function of Moment of Inertia.
    w_func : Bend_func_sub
        Bending function.
    w_params : pd.Series of dict
        Parameters per step for bending function (w_func).
    rel_steps : pandas.Index or numpy array, optional
        Relevant steps. The default is None.
    n : integer, optional
        Number of determination points. The default is 100.
    pb_b : boolean, optional
        Switch progress bar output. The default is True.
    name : string, optional
        Name of executed check. The default is 'X'.

    Returns
    -------
    check_EtoD_g : pandas.Dataframe
        Global (total length) mean deviation of scaled analytical and measured deformation.
    check_EtoD_x : pandas.Dataframe
        Local (midspan) mean deviation of scaled analytical and measured deformation.
    check_E : dictionary of pandas.Dataframe
        Deviation of scaled analytical and measured deformation.
    w_D_to_E : dictionary of pandas.Dataframe
        Scaled analytical defeormation by method.

    """
    if pb_b: pb = tqdm(range(len(E_dict.keys())), desc ="check "+name+": ",
                       unit=' method', ncols=100)
    step_range = Evac.pd_combine_index(F, w_params)
    if not rel_steps is None: 
        step_range = Evac.pd_combine_index(rel_steps, step_range)
    xlin=np.linspace(-Length/2,Length/2,n+1)
    w_vgl_df = w_func(xlin, w_params.loc[step_range],
                      coords=None,coords_set=None,col_type='val')
    m = YM_eva_method_D_bend_df(Length=Length, I_func=I_func, n=n, E=1, F=1)
    
    D_alpha = (m.index/Length-0.5)*m['Omega'].iloc[0]
    D_beta  = (m.index/Length+0.5)*m['Omega'].iloc[-1]
    D_sigma = (D_alpha-D_beta+m['Omega'])
    w_D_to_E={}
    check_E={}
    check_EtoD_g = {}
    check_EtoD_x = {}
    for E in E_dict:
        D_gamma = F.loc[step_range]/E_dict[E].loc[step_range]
        tmp = np.array([D_gamma.values,]*D_sigma.shape[0]).transpose()
        w_D_to_E[E] = pd.DataFrame(tmp*D_sigma.values,
                                   index=D_gamma.index,
                                   columns=D_sigma.index)
        check_E[E] = (w_D_to_E[E].loc[step_range]-w_vgl_df.loc[step_range])/w_vgl_df.loc[step_range]
        check_EtoD_g[E] = coord_df_mean(check_E[E], E)
        check_EtoD_x[E] = coord_df_depo(check_E[E], E)
        if pb_b: pb.update()
    if pb_b: pb.close()
    check_EtoD_g = pd.DataFrame.from_dict(check_EtoD_g)
    check_EtoD_x = pd.DataFrame.from_dict(check_EtoD_x)
    return check_EtoD_g, check_EtoD_x, check_E, w_D_to_E
    
    

#%% Plot
class Plotter:
    def colplt_df_ax(df, step_range=None,
                     title='', xlabel='', ylabel='',
                     Point_df=None, ax=None,
                     cblabel='Step', cbtick=None):
        """Returns a matpltlib axis plot of a pandas Dataframe of type points in a range."""
        if step_range is None: step_range=df.index
        if ax is None: ax = plt.gca()
        # cb_map=cm.ScalarMappable(norm=colors.Normalize(vmin=0,vmax=step_range.max()-step_range.min()+1),cmap=cm.rainbow)
        cb_map=cm.ScalarMappable(norm=colors.Normalize(vmin=step_range.min(),vmax=step_range.max()),cmap=cm.rainbow)
        color=cb_map
        # color=cm.rainbow(np.linspace(0,1,step_range.max()-step_range.min()+1))
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for step in step_range:
            ax.plot(Point_df_idx(df,steps=step,coords='x'),
                    Point_df_idx(df,steps=step,coords='y'),
                    '-',c=color.to_rgba(step))
            if Point_df is not None: 
                ax.plot(Point_df_idx(Point_df,steps=step,coords='x'),
                        Point_df_idx(Point_df,steps=step,coords='y'),
                        'k+', markersize=.8)
        if cbtick is None:
            cb1=plt.colorbar(cb_map, extend='max', label=cblabel, ax=ax)
        else:
            cb1=plt.colorbar(cb_map, extend='max', label=cblabel, ax=ax,
                             ticks=list(cbtick.values()))
            cb1.ax.set_yticklabels(list(cbtick.keys()))
        cb1.ax.invert_yaxis()
        ax.grid()
        return ax
    
    def colplt_common_ax(xdata, ydata, step_range=None,
                         title='', xlabel='', ylabel='',
                         xstep=False, ystep=True,
                         Point_df=None, ax=None,
                         cblabel='Step', cbtick=None):
        """Returns a matpltlib axis plot of a pandas Dataframe of type points in a range."""
        if step_range is None:
            if ystep: step_range=ydata.index
            elif xstep: step_range=xdata.index
            else: step_range=[0]
        if ax is None: ax = plt.gca()
        # cb_map=cm.ScalarMappable(norm=colors.Normalize(vmin=0,vmax=step_range.max()-step_range.min()+1),cmap=cm.rainbow)
        cb_map=cm.ScalarMappable(norm=colors.Normalize(vmin=step_range.min(),vmax=step_range.max()),cmap=cm.rainbow)
        color=cb_map
        # color=cm.rainbow(np.linspace(0,1,step_range.max()-step_range.min()+1))
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for step in step_range:
            ax.plot(xdata.loc[step] if xstep else xdata,
                    ydata.loc[step] if ystep else ydata,
                    '-',c=color.to_rgba(step))
        if cbtick is None:
            cb1=plt.colorbar(cb_map, extend='max', label=cblabel, ax=ax)
        else:
            cb1=plt.colorbar(cb_map, extend='max', label=cblabel, ax=ax,
                             ticks=list(cbtick.values()))
            cb1.ax.set_yticklabels(list(cbtick.keys()))
        cb1.ax.invert_yaxis()
        ax.grid()
        return ax
    
    def colplt_funcs_ax(x, func, params, step_range=None,
                        title='', xlabel='', ylabel='',
                        Point_df=None, ax=None,
                        cblabel='Step', cbtick=None):
        """Returns a matpltlib axis plot of a function with defined parameters in a range."""
        if step_range is None: step_range=params.index
        df1 = Point_df_from_lin(x, step_range)
        try: #wenn nicht alle enthalten sind (bei RangeIndex)
            df2 = func(x, params.loc[step_range])
        except KeyError:
            df2 = func(x, params.indlim(step_range.min(),step_range.max()))
        df  = Point_df_combine(df1, df2)
        ax  = Plotter.colplt_df_ax(df, step_range,
                                   title, xlabel, ylabel,
                                   Point_df, ax,
                                   cblabel, cbtick)
        return ax
    
    def colplt_funcs_one(x, func, params, step_range=None,
                         title='', xlabel='', ylabel='',
                         Point_df=None,
                         cblabel='Step', cbtick=None,
                         savefig=False, savefigname=None):
        """Returns a matpltlib figure plot of a function with defined parameters in a range."""
         
        fig, ax1 = plt.subplots()
        Plotter.colplt_funcs_ax(x, func, params, step_range,
                                title, xlabel, ylabel,
                                Point_df, ax1,
                                cblabel, cbtick)
        fig.tight_layout()
        if savefig:
            fig.savefig(savefigname+'.pdf')
            fig.savefig(savefigname+'.png')
        plt.show()
        fig.clf()
        plt.close(fig)
        
    def colplt_funcs_all(x, func_cohort, params, step_range=None,
                         title='', xlabel='',
                         Point_df=None,
                         cblabel='Step', cbtick=None,
                         savefig=False, savefigname=None):
        """Returns a matpltlib axis plot of a function cohort (function and theire first and second derivate) with defined parameters in a range."""
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1,
                                          sharex=False, sharey=False,
                                          figsize = (plt.rcParams['figure.figsize'][0],plt.rcParams['figure.figsize'][1]*3))
        fig.suptitle(title)
        Plotter.colplt_funcs_ax(x=x, func=func_cohort['d0'],
                                params=params, step_range=step_range,
                                title='Displacement', 
                                xlabel=xlabel, ylabel='Displacement / mm',
                                Point_df=Point_df, ax=ax1,
                                cblabel=cblabel, cbtick=cbtick)
        Plotter.colplt_funcs_ax(x=x, func=func_cohort['d1'],
                                params=params, step_range=step_range,
                                title='Slope', 
                                xlabel=xlabel, ylabel='Slope / (mm/mm)',
                                Point_df=None, ax=ax2,
                                cblabel=cblabel, cbtick=cbtick)
        Plotter.colplt_funcs_ax(x=x, func=func_cohort['d2'],
                                params=params, step_range=step_range,
                                title='Curvature', 
                                xlabel=xlabel, ylabel='Curvature / (1/mm)',
                                Point_df=None, ax=ax3,
                                cblabel=cblabel, cbtick=cbtick)
        fig.tight_layout()
        if savefig:
            fig.savefig(savefigname+'.pdf')
            fig.savefig(savefigname+'.png')
        plt.show()
        fig.clf()
        plt.close(fig)