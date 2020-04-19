#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:34:17 2020

@author: m4rz910
"""

from model import Fuzzification, InferenceEngine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'

#construct linguistic value databases
wms = Fuzzification.wms()
vbbagvg = Fuzzification.vbbagvg()


######################################## WEALTH- STATUS #############################################
#START
wealth = pd.read_excel('./databases/indicator_db.xlsx', sheet_name='wealth').round(Fuzzification.nd)

#NORMALIZATION
curves = []
indicators = ['roe','roa'] #CHANGE ME 
for basic_indicator in indicators: 
    assert wealth[wealth['basic']==basic_indicator].shape[1] == 9 , 'Warning, not expected shape' #check,sensitve to number of basic indicators, CHANGE ME
    basic_df = wealth[wealth['basic']==basic_indicator]
    iv = basic_df['intensive_value']
    
    discourse = np.arange(iv.min(),iv.max()+Fuzzification.dx,Fuzzification.dx).round(Fuzzification.nd)
    
    if basic_indicator == 'roe': #MIDDLE IS BETTER, CHANGE ME
        x = Fuzzification.trapezoid(z= discourse,
                                    c_l=iv.min(),
                                    tc=iv.mean(),
                                    Tc=iv.mean(),
                                    c_u=iv.max());
        
    if basic_indicator == 'roa': #HIGHER IS BETTER, CHANGE ME
        x = Fuzzification.trapezoid(z=discourse,
                                    c_l=iv.min(),
                                    tc=iv.max(),
                                    Tc=iv.max(),
                                    c_u=iv.max())
        
    ncurve = pd.Series(data=x.round(Fuzzification.nd),
                        index=discourse.round(Fuzzification.nd),
                        name=basic_indicator)
    curves.append(ncurve)

plt.figure(figsize=(10,5));plt.title('Normalization Cruves');
for curve in curves:
    curve.plot();
plt.grid();plt.xlabel('z');plt.ylabel('x'); plt.legend(indicators);
    
#FUZZIFICATION
frames = []
for curve in curves: #order of curves matters
    df_base = wealth[wealth['basic']==curve.name].drop(['raw_value','raw_value_units','intensive_units','source'],axis='columns').reset_index(drop=True) #reset index so that it concats properly
    z = df_base['intensive_value'].values
    x = curve.loc[z] # pass through normalization curve
    wms_v = wms.loc[x.values].reset_index(); #reset index to make concat work
    df_out = pd.concat([df_base,wms_v],axis='columns')
    assert df_base.shape[0] == df_out.shape[0], 'different amount of rows, error'
    frames.append(df_out)
wealth = pd.concat(frames).reset_index(drop=True)
wealth

#INFERENCE
frames2=[]
for company in wealth['company'].unique():
    frames = []
    for year in wealth['year'].unique():
        company_year = wealth[(wealth['company'] == company) &
                              (wealth['year'] == year)]

        basic_indicators = [company_year[company_year['basic']=='roe'].iloc[0], #CHANGE ME converted to series
                            company_year[company_year['basic']=='roa'].iloc[0]]

        #apply inference engine
        frames.append(InferenceEngine.b2__s(basic_indicators)) #CHANGE ME
    frames2.append(pd.concat(frames,axis='columns').T)
df_status = pd.concat(frames2).reset_index(drop=True)


####################################### WEALTH- RESPONSE #############################################
wealth = pd.read_excel('./databases/indicator_db.xlsx', sheet_name='wealth').round(Fuzzification.nd)

#NORMALIZATION
indicators = ['CF/CAPEX','dividend payout ratio'] #CHANGE ME 
curves = []
for basic_indicator in indicators: 
    assert wealth[wealth['basic']==basic_indicator].shape[1] == 9 , 'Warning, not expected shape' #check,sensitve to number of basic indicators, CHANGE ME
    basic_df = wealth[wealth['basic']==basic_indicator]
    iv = basic_df['intensive_value']
   
    discourse = np.arange(iv.min(),iv.max()+Fuzzification.dx,Fuzzification.dx).round(Fuzzification.nd)
    
    if basic_indicator == 'CF/CAPEX': #HIGHER IS BETTER, CHANGE ME
        x = Fuzzification.trapezoid(z=discourse,
                                    c_l=iv.min(),
                                    tc=iv.max(),
                                    Tc=iv.max(),
                                    c_u=iv.max())
        
    if basic_indicator == 'dividend payout ratio': #HIGHER IS BETTER, CHANGE ME
        x = Fuzzification.trapezoid(z=discourse,
                                    c_l=iv.min(),
                                    tc=iv.max(),
                                    Tc=iv.max(),
                                    c_u=iv.max())
        
    ncurve = pd.Series(data=x.round(Fuzzification.nd),
                       index=discourse.round(Fuzzification.nd),
                       name=basic_indicator)
    curves.append(ncurve)

#plotting normalization curve
plt.figure(figsize=(10,5));plt.title('Normalization Cruves');
for curve in curves:
    curve.plot();
plt.grid();plt.xlabel('z');plt.ylabel('x'); plt.legend(indicators);

#FUZZIFICATION
frames = []
for curve in curves: #order of curves matters
    df_base = wealth[wealth['basic']==curve.name].drop(['raw_value','raw_value_units','intensive_units','source'],axis='columns').reset_index(drop=True) #reset index so that it concats properly
    z = df_base['intensive_value'].values
    x = curve.loc[z] # pass through normalization curve
    wms_v = wms.loc[x.values].reset_index(); #reset index to make concat work
    df_out = pd.concat([df_base,wms_v],axis='columns')
    assert df_base.shape[0] == df_out.shape[0], 'different amount of rows, error'
    frames.append(df_out)
wealth = pd.concat(frames).reset_index(drop=True)

#INFERENCE
frames2=[]
for company in wealth['company'].unique():
    frames = []
    for year in wealth['year'].unique():
        company_year = wealth[(wealth['company'] == company) &
                              (wealth['year'] == year)]

        basic_indicators = [company_year[company_year['basic']=='CF/CAPEX'].iloc[0], #CHANGE ME converted to series
                            company_year[company_year['basic']=='dividend payout ratio'].iloc[0]]

        #apply inference engine
        frames.append(InferenceEngine.b2__s(basic_indicators)) #CHANGE ME
    frames2.append(pd.concat(frames,axis='columns').T)
df_response = pd.concat(frames2).reset_index(drop=True)

######################################## WEALTH - PRESSURE #############################################
#IMPORT DATA
wealth = pd.read_excel('./databases/indicator_db.xlsx', sheet_name='wealth').round(Fuzzification.nd)


#NORMALIZATION
indicators = ['debt to equity ratio','operating expenses','effective tax rate'] #CHANGE ME 
curves = []
for basic_indicator in indicators: 
    assert wealth[wealth['basic']==basic_indicator].shape[1] == 9 , 'Warning, not expected shape' #check,sensitve to number of companies, CHANGE ME
    basic_df = wealth[wealth['basic']==basic_indicator]
    iv = basic_df['intensive_value']
    discourse = np.arange(iv.min(),iv.max()+Fuzzification.dx,Fuzzification.dx)
    
    if basic_indicator == 'debt to equity ratio': #MIDDLE IS BETTER, CHANGE ME
        x = Fuzzification.trapezoid(z= discourse,
                                    c_l=iv.min(),
                                    tc=iv.mean(),
                                    Tc=iv.mean(),
                                    c_u=iv.max());
        
    if basic_indicator == 'operating expenses': # LOWER IS BETTER, CHANGE ME
        x = Fuzzification.trapezoid(z=discourse,
                                    c_l=iv.min(),
                                    tc=iv.min(),
                                    Tc=iv.min(),
                                    c_u=iv.max())
        
    if basic_indicator == 'effective tax rate': # LOWER IS BETTER, CHANGE ME
        x = Fuzzification.trapezoid(z=discourse,
                                    c_l=iv.min(),
                                    tc=iv.min(),
                                    Tc=iv.min(),
                                    c_u=iv.max())
    ncurve = pd.Series(data=x.round(Fuzzification.nd),
                       index=discourse.round(Fuzzification.nd),
                       name=basic_indicator)
    curves.append(ncurve)

#plotting normalization curve
for curve in curves:
    plt.figure(figsize=(10,5));plt.title('Normalization Cruve: {}'.format(curve.name));
    curve.plot(); plt.grid();plt.xlabel('z');plt.ylabel('x');
    
#FUZZIFICATION
frames = []
for curve in curves: #order of curves matters
    df_base = wealth[wealth['basic']==curve.name].drop(['raw_value','raw_value_units','intensive_units','source'],axis='columns').reset_index(drop=True) #reset index so that it concats properly
    z = df_base['intensive_value'].values
    x = curve.loc[z] # pass through normalization curve
    wms_v = wms.loc[x.values].reset_index(); #reset index to make concat work
    df_out = pd.concat([df_base,wms_v],axis='columns')
    assert df_base.shape[0] == df_out.shape[0], 'different amount of rows, error'
    frames.append(df_out)
wealth = pd.concat(frames).reset_index(drop=True)

#INFERENCE
frames2=[]
for company in wealth['company'].unique():
    frames = []
    for year in wealth['year'].unique():
        company_year = wealth[(wealth['company'] == company) &
                              (wealth['year'] == year)]

        basic_indicators = [company_year[company_year['basic']=='debt to equity ratio'].iloc[0], #CHANGE ME converted to series
                            company_year[company_year['basic']=='operating expenses'].iloc[0],
                            company_year[company_year['basic']=='effective tax rate'].iloc[0]]

        #apply inference engine
        frames.append(InferenceEngine.b3__s(basic_indicators)) #CHANGE ME
    frames2.append(pd.concat(frames,axis='columns').T)
df_pressure = pd.concat(frames2).reset_index(drop=True)

################################# AGGREGATE #######################################
df_wealth = pd.concat([df_status,df_response,df_pressure],axis='index')