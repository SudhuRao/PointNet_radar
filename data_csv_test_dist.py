# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:34:57 2019

@author: uku4kor
"""

import os
import sys
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import pandas as pd
from pandas import ExcelWriter
#from matplotlib import gridspec
import h5py
import re
#import sklearn
from scipy.ndimage.filters import gaussian_filter

MIN_DATA = 20
DATA_SIZE = 1024
DISTs = [80]

count = 0
flg_train = True
flg_val = False
flg_tst = False



f = h5py.File('D:/Sudhu/Project/Data/trackList.mat')
tracklist = f['trackList']


pic_list = range(0,2861)

for dist in DISTs:
    for track in pic_list:
        
        valid = False
        #****************Annotation of target and creation of file****************  
    
        #Annotation coloumn0 = car   
        #Annotation coloumn1 = 2 wheeler  
        #Annotation coloumn2 = pedestrian
        #Annotation coloumn3 = other
        
        name = f[tracklist['type'][track,0]][()]
        name = ''.join(map(chr,name))
        
    
            
        if(bool(re.match("(.*A3.*)|(.*A6.*)|(.*Golf.*)|(.*Mokka.*)|(.*Octavia.*)|(.*Passat.*)|(.*Polo.*)",name))):
            label = 'car'
            valid = True
        
        elif(bool(re.match(".*TwoWheeler.*",name))):
            label = 'TwoWheeler'
            valid = True
            
        elif(bool(re.match(".*Pedestrian.*",name))):
            label = 'Pedestrian'
            valid = True
            
        elif(bool(re.match(".*Coke.*",name))):
            label = 'ovrride'
            valid = True
            
        else:
            label = 'Other'
            valid = False
            
        if(valid):
    
            dx = []
            dy = []
            dz = []
            rcs = []
            vel = []
        
            
            cycs = len(f[tracklist['ra5'][track,0]]['matchedLocDx'])
            
            
            for cyc in range(0,cycs-1):    
                dx_cyc = f[tracklist['ra5'][track,0]]['matchedLocDx'][cyc,]
                dx_cyc = dx_cyc[~np.isnan(dx_cyc)]
                dx_ref = f[tracklist['objAttributes'][track,0]]['dxVec'][cyc,]
                
                dy_cyc = f[tracklist['ra5'][track,0]]['matchedLocDy'][cyc,]
                dy_cyc = dy_cyc[~np.isnan(dy_cyc)]
                dy_ref = f[tracklist['objAttributes'][track,0]]['dyVec'][cyc,]
                dy_cyc = np.absolute(dy_cyc) - np.absolute(dy_ref)
                
                dz_cyc = f[tracklist['ra5'][track,0]]['matchedLocDz'][cyc,]
                dz_cyc = dz_cyc[~np.isnan(dz_cyc)]
                dz_ref = f[tracklist['objAttributes'][track,0]]['dzVec'][cyc,]
                        
                rcs_cyc = f[tracklist['ra5'][track,0]]['matchedRCS'][cyc,]
                rcs_cyc = rcs_cyc[~np.isnan(rcs_cyc)]
                
                #currentVr = feature.get_vr() + (SensorVelocityX * cosAlpha) + (SensorVelocityY * sinAlpha) 
                #SensorVelocityY = psiDt * mountingPosX
                vr_cyc = f[tracklist['ra5'][track,0]]['matchedVr'][cyc,]
                alpha = f[tracklist['ra5'][track,0]]['matchedAlpha'][cyc,]
                vr_cyc = vr_cyc[~np.isnan(vr_cyc)]
                alpha = alpha[~np.isnan(alpha)]
                senvelx = f[tracklist['vehAttributes'][track,0]]['vEgo'][cyc,]
                senvely = (f[tracklist['vehAttributes'][track,0]]['psiDtEgo'][cyc,])*(f[tracklist['vehAttributes'][track,0]]['mountOffset_dx'][cyc,])
        
                for idx in range(0,len(vr_cyc)):
                    alpha_temp = math.radians(alpha[idx])
                    vr_cyc[idx] = vr_cyc[idx] + (senvelx * math.cos(alpha_temp)) + (senvely *math.sin(alpha_temp))
                    vr_cyc[idx] = np.absolute(vr_cyc[idx])
                
#                if(len(dz_cyc)==0):
#                    dz_cyc = dz_ref
                
                invalid_idx = []
                for idx in range(0,len(dx_cyc)):
                    if(dx_cyc[idx] > dist):
                        invalid_idx.append(idx)
                        
                dx_cyc = np.delete(dx_cyc,invalid_idx)
                dy_cyc = np.delete(dy_cyc,invalid_idx)
#                rcs_cyc = np.delete(rcs_cyc,invalid_idx)
#                vr_cyc = np.delete(vr_cyc,invalid_idx)
                
#                if(len(dx_cyc)>0):
#                    if(len(dz_cyc) != len(dx_cyc)):
#                        idx = np.random.randint(0,len(dz_cyc),size=len(dx_cyc))
#                        dz_cyc = dz_cyc[idx]
                      
                for idx in range(0,len(dx_cyc)):
                    dx_cyc[idx] = dx_cyc[idx] - dx_ref + 4
                    dx.append(dx_cyc[idx])
                    dy.append(dy_cyc[idx])
                    dz.append(0)
#                    rcs.append(rcs_cyc[idx])
#                    vel.append(vr_cyc[idx])
                    
               
            #************resizing the data to data size*********
            dx = np.array(dx)
            dy = np.array(dy)
            dz = np.array(dz)
#            rcs = np.array(rcs)
#            vel = np.array(vel)
            
            if(len(dx)>MIN_DATA):
                if(len(dx)<=DATA_SIZE):
                   
                    reqd_data = DATA_SIZE - len(dx)
    
                    dx = np.concatenate((dx,np.repeat(4.0, reqd_data)),axis=0)
                    dy = np.concatenate((dy,np.repeat(0, reqd_data)),axis=0)
                    dz = np.concatenate((dz,np.repeat(0, reqd_data)),axis=0)
                else:
                    dx = dx[-DATA_SIZE:]
                    dy = dy[-DATA_SIZE:]
                    dz = dz[-DATA_SIZE:]
                    
                collabel=("Dx", "Dy","Dz")
                df = pd.DataFrame({"Dx": dx,"Dy":dy,"Dz":dz},columns= collabel)
                
           
#                df.to_csv('D:/Sudhu/Project/codes/my_pointnet/Data_npoint_z0/points_64/{}_{}_{}.csv'.format(dist,name,label,track), index = False)
#
#            
                if(flg_train == True):
                    df.to_csv('D:\Sudhu\Project\codes\my_pointnet\Data_npoint_z0\points_1024\Train\{}_{}_{}.csv'.format(name,label,track), index = False)
                    
                if(flg_val == True):
                    df.to_csv('D:\Sudhu\Project\codes\my_pointnet\Data_npoint_z0\points_1024\Validation\{}_{}_{}.csv'.format(name,label,track), index = False)
                
                if(flg_tst == True):
                    df.to_csv('D:\Sudhu\Project\codes\my_pointnet\Data_npoint_z0\points_1024\Test\{}_{}_{}.csv'.format(name,label,track), index = False)
                        
                count = count + 1
                
                if(count < 6):
                    flg_train = True
                    flg_val = False
                    flg_tst = False
                elif(count<8): 
                    flg_train = False
                    flg_val = True
                    flg_tst = False
                elif(count<10): 
                    flg_train = False
                    flg_val = False
                    flg_tst = True
                else:
                    count = 0
                    flg_train = True
                    flg_val = False
                    flg_tst = False
        


    
