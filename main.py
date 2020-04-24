#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:05:40 2020

@author: m4rz910
"""

from model import Fuzzification, InferenceEngine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'


struct = {'wealth':{
                    'status':{'roe':'middle_is_better',
                              'roa':'higher_is_better'},
                    'pressure':{'cf_capex':'higher_is_better',
                                'dividend payout ratio':'middle_is_better'}, 
                    'response':{'debt to equity ratio':'middle_is_better',
                                'operating expenses':'lower_is_better',
                                'effective tax rate':'lower_is_better'}
                   },
          'ecos':{
                    'air':{'clean generation':'lower_is_better', #bc we want to see a low percentage of coal
                           'co2 emissions':'lower_is_better'},
                    'land':{'spills':'lower_is_better',
                            'solid waste':'lower_is_better',
                            'hazardous waste':'lower_is_better'}, 
                    'water':{'total water withdrawal':'lower_is_better',
                             'percent water consumed':'lower_is_better'}
                   },
          'hums':{
                    'health':{'fatalities':'lower_is_better', #bc we want to see a low percentage of coal
                              'osha recordable rate (ticr)':'lower_is_better'},
                    'polic':{'lobbying spending':'lower_is_better',
                            'charitable giving':'higher_is_better'}
                   }
         }

def basic_to_secondary(year, delta, sensitivity_ind=None):
    print('Starting Basic -> Secondary')
    for primary_indicator in struct.keys():
        frames = []
        for secondary_indicator in struct[primary_indicator]:
            #print('{}-{}'.format(primary_indicator, secondary_indicator))
            frames.append(Fuzzification.main(primary_indicator=primary_indicator,
                                             secondary_indicator=secondary_indicator,
                                             basic_indicators=struct[primary_indicator][secondary_indicator],
                                             indicator_type='basic',
                                             year=year,
                                             sensitivity_ind=sensitivity_ind,
                                             delta=delta))
        secondary_agg = pd.concat(frames).reset_index(drop=True)
        out_file = os.path.join(annuals_dir,'{}_{}_{}_{}_secondary_agg.csv'.format(delta,sensitivity_ind, year,primary_indicator))
        secondary_agg.to_csv(out_file,index=False)
        
def secondary_to_primary(year, delta, sensitivity_ind=None):
    print('Starting Secondary -> Primary')
    frames2=[]
    for primary_ind_name in struct.keys():
        #print(primary_ind_name)
        in_file = os.path.join(annuals_dir,'{}_{}_{}_{}_secondary_agg.csv'.format(delta,sensitivity_ind, year,primary_ind_name))
        secondary_agg = pd.read_csv(in_file)
        indicator_type = 'secondary'
        #INFERENCE
        frames=[]
        for company in secondary_agg['company'].unique():
            company_year = secondary_agg[(secondary_agg['company'] == company) &
                                         (secondary_agg['year'] == year)]
            s_indicators = [company_year[company_year[indicator_type]==indicator].iloc[0] for indicator in secondary_agg[indicator_type].unique()]
            frames.append(InferenceEngine.s_p(s_indicators,primary_ind_name,indicator_type)) #apply inference engine
        primary = pd.DataFrame(frames)
        frames2.append(primary)
    primary_agg = pd.concat(frames2)
    
    out_file = os.path.join(annuals_dir,'{}_{}_{}_primary_agg.csv'.format(delta,sensitivity_ind,year))           
    primary_agg.to_csv(out_file,index=False)

def primary_to_osus(year, delta, sensitivity_ind=None): 
    print('Starting Primary  -> OSUS')
    in_file = os.path.join(annuals_dir,'{}_{}_{}_primary_agg.csv'.format(delta,sensitivity_ind, year))
    primary_agg = pd.read_csv(in_file)
    
    osus_ind_name = 'osus'
    indicator_type = 'primary'
    #INFERENCE
    frames=[]
    for company in primary_agg['company'].unique():
        company_year = primary_agg[(primary_agg['company'] == company) &
                                   (primary_agg['year'] == year)]
        s_indicators = [company_year[company_year[indicator_type]==indicator].iloc[0] for indicator in primary_agg[indicator_type].unique()]
        frames.append(InferenceEngine.s_p(s_indicators,osus_ind_name,indicator_type)) #apply inference engine
    osus = pd.DataFrame(frames)
    
    out_file = os.path.join(annuals_dir,'{}_{}_{}_osus_fuzz.csv'.format(delta,sensitivity_ind,year))      
    osus.to_csv(out_file, index=False)
    
def defuzzification(year, delta, sensitivity_ind=None):
    print('Defuzzifing')
    in_file = os.path.join(annuals_dir,'{}_{}_{}_osus_fuzz.csv'.format(delta,sensitivity_ind,year))   
    osus = pd.read_csv(in_file)
    elvllflifhhvheh = Fuzzification.elvllflifhhvheh()
    osus['crisp']= (elvllflifhhvheh['el'].idxmax()*osus['el'] +
                    elvllflifhhvheh['vl'].idxmax()*osus['vl'] + 
                    elvllflifhhvheh['l' ].idxmax()*osus['l' ] + 
                    elvllflifhhvheh['fl'].idxmax()*osus['fl'] + 
                    elvllflifhhvheh['i' ].idxmax()*osus['i' ] +
                    elvllflifhhvheh['fh'].idxmax()*osus['fh'] +
                    elvllflifhhvheh['h' ].idxmax()*osus['h' ] + 
                    elvllflifhhvheh['vh'].idxmax()*osus['vh'] + 
                    elvllflifhhvheh['eh'].idxmax()*osus['eh'])/osus[['el','vl','l','fl','i','fh','h','vh','eh']].sum(axis='columns')
    crisp = osus[['company','year','crisp']]
    
    out_file = os.path.join(annuals_dir,'{}_{}_{}_osus_crisp.csv'.format(delta,sensitivity_ind,year)) 
    crisp.to_csv(out_file, index=False)

def generate_time_series_plot(years = range(2014,2019), delta=0):
    print('GENERATING TIME SERIES CHART')
    frames = [pd.read_csv(os.path.join(annuals_dir,'{}_{}_{}_osus_crisp.csv'.format(delta,None,year))) for year in years]
    osus = pd.concat(frames).set_index('year',drop=True)
    osus.index= pd.to_datetime(osus.index.astype(str), format='%Y')
    fig, ax = plt.subplots(figsize=(10,5))
    for idx, gp in osus.groupby('company'):
        gp.plot(ax=ax,label=idx)
    plt.grid(True); plt.xlabel('Year'); plt.ylabel('OSUS');plt.legend(osus.groupby('company').indices)
    
    out_file = os.path.join(pkg_dir,'outputs/final_chart_output.png') 
    fig.savefig(out_file, dpi=300, bbox_inches='tight')

def run_model(years = range(2014,2019), sensitivity_ind=None, delta=0):    
    print('\n'); print('RUNNING MODEL')
    for year in years: 
        print(year)
        basic_to_secondary(year=year, sensitivity_ind=sensitivity_ind, delta=delta)
        secondary_to_primary(year=year, sensitivity_ind=sensitivity_ind, delta=delta)
        primary_to_osus(year=year, sensitivity_ind=sensitivity_ind, delta=delta)
        defuzzification(year=year, sensitivity_ind=sensitivity_ind, delta=delta)
    print('MODEL RUN COMPLETE')
    
def run_sensitivity_analysis(sensitivity_year = [2018]):
    """
    Sensitivity analysis is only run on one year
    """
    print('BEGINNING SENSITIVITY ANALYSIS')
    
    basic_indicator_list = [] #get a list of the basic indicators
    for key in struct.keys():
        for key2 in struct[key].keys():
             basic_indicator_list =  basic_indicator_list + list(struct[key][key2].keys())
    
    for delta in [-0.1, 0.1]: # normalized number purturbation
        for sensitivity_ind in basic_indicator_list:
            run_model(years = sensitivity_year, sensitivity_ind = sensitivity_ind, delta = delta)

def compile_sensitivity_results(sensitivity_year = [2018]):
    basic_indicator_list = [] #get a list of the basic indicators
    frames = []
    for key in struct.keys():
        for key2 in struct[key].keys():
             basic_indicator_list =  basic_indicator_list + list(struct[key][key2].keys())
             
             normalized = pd.read_csv(os.path.join(annuals_dir,'{}_{}_{}_intensive_normalized_basic_indicators.csv'.format(0,
                                                   key,
                                                   key2)))
             frames.append(normalized)
    df = pd.concat(frames)
    df = df[df['year']==sensitivity_year[0]]
    df.rename(columns={"basic": "sensitivity_ind"}, inplace=True)    
         
    frames =[]         
    for sensitivity_ind in basic_indicator_list:
        
        #crisp values    
        base = pd.read_csv(os.path.join(annuals_dir,'{}_{}_{}_osus_crisp.csv'.format(0,
                                                                                     'None',
                                                                                     sensitivity_year[0])))        
        for delta in [-0.1,0.1]:
            pert = pd.read_csv(os.path.join(annuals_dir,'{}_{}_{}_osus_crisp.csv'.format(delta,
                                                                                     sensitivity_ind,
                                                                                     sensitivity_year[0])))
            base[delta] = pert['crisp']
            
        base['sensitivity_ind'] = sensitivity_ind
        frames.append(base)
    df2 = pd.concat(frames)
    
    df3 = pd.merge(left=df2,right=df,on=['sensitivity_ind','company','year'])
    df3['D_h'] = abs( (1-df3['index']) * (df3[0.1]-df3['crisp']) )
    df3['D_l'] = abs( (1-df3['index']) * (df3[-0.1]-df3['crisp']))
    df3['D'] = df3[['D_h', 'D_l']].max(axis=1)
    
    #full rank information
    frames = []
    for idx, gp in df3.groupby('company'):
        gp['D_rank'] = gp['D'].rank(ascending=False)
        frames.append(gp)   
    df = pd.concat(frames)
    df.to_csv(os.path.join(annuals_dir,'sensitivity_analysis.csv'),index=False)

    # average rank data
    avg_rank=df.loc[:,['sensitivity_ind','D_rank']].groupby('sensitivity_ind').mean().sort_values(by='D_rank',ascending=True)
    avg_rank.to_csv(os.path.join(annuals_dir,'sensitivity_analysis_rank_stats.csv'),index=False)
    
    # average rank plot
    fig, ax = plt.subplots(figsize=(5,10))
    y_pos = np.arange(len(avg_rank.index))
    ax.barh(y_pos, avg_rank['D_rank'], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avg_rank.index)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Average Rank')
    ax.set_title('Sensitivity Analysis: Average Basic Indicator Rank')
    fig.savefig(os.path.join(pkg_dir,'outputs/sensitivity_average_rank_bar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
pkg_dir = 'C:\\Users\\tzave\\OneDrive - The Cooper Union for the Advancement of Science and Art\\102-cooper\\150-masters\\sustainability\\sustainability_project1\\IOU-Sustainability-Model'

annuals_dir = os.path.join(pkg_dir, 'outputs/annuals')
if not os.path.exists(annuals_dir): os.makedirs(annuals_dir)

norm_dir = os.path.join(pkg_dir, 'outputs/normalization_curves')
if not os.path.exists(norm_dir): os.makedirs(norm_dir)

if __name__ == '__main__':
    tic = time.perf_counter()
    
    #Fuzzification.create_normalization_curves(struct)
       
    run_model(); generate_time_series_plot() # BASELINE MODEL
    
    run_sensitivity_analysis(); compile_sensitivity_results() # you must rerun baseline to get accurate sensitivity analysis
     
    toc = time.perf_counter()
    print('COMPLETED! Duration: {} mins'.format((toc-tic)/60))
    
