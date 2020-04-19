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

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titleweight'] = 'bold'


struct = {'wealth':{
                    'status':{'roe':'middle_is_better',
                              'roa':'higher_is_better'},
                    'pressure':{'CF_CAPEX':'higher_is_better',
                                'dividend payout ratio':'middle_is_better'}, 
                    'response':{'debt to equity ratio':'middle_is_better',
                                'operating expenses':'lower_is_better',
                                'effective tax rate':'lower_is_better'}
                   },
          'ecos':{
                    'air':{'clean generation':'lower_is_better', #bc we want to see a low percentage of coal
                           'CO2 emissions':'lower_is_better'},
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

def basic_to_secondary(year):
    print('Starting Basic -> Secondary')
    for primary_indicator in struct.keys():
        frames = []
        for secondary_indicator in struct[primary_indicator]:
            print('{}-{}'.format(primary_indicator, secondary_indicator))
            frames.append(Fuzzification.main(primary_indicator=primary_indicator,
                                             basic_indicators=struct[primary_indicator][secondary_indicator],
                                             indicator_type='basic',
                                             year=year))
        secondary_agg = pd.concat(frames).reset_index(drop=True)
        secondary_agg.to_csv('./outputs/annuals/{}_{}_secondary_agg.csv'.format(year,primary_indicator),index=False)
        
def secondary_to_primary(year):
    print('Starting Secondary -> Primary')
    frames2=[]
    for primary_ind_name in struct.keys():
        print(primary_ind_name)
        secondary_agg = pd.read_csv('./outputs/annuals/{}_{}_secondary_agg.csv'.format(year,primary_ind_name))
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
    primary_agg.to_csv('./outputs/annuals/{}_primary_agg.csv'.format(year),index=False)

def primary_to_osus(year): 
    print('Starting Primary  -> OSUS')
    primary_agg = pd.read_csv('./outputs/annuals/{}_primary_agg.csv'.format(year))
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
    osus.to_csv('./outputs/annuals/{}_osus_fuzz.csv'.format(year),index=False)
    
def defuzzification(year):
    print('Defuzzification')
    osus = pd.read_csv('./outputs/annuals/{}_osus_fuzz.csv'.format(year))
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
    crisp.to_csv('./outputs/annuals/{}_osus_crisp.csv'.format(year),index=False)

def make_final_plot(years):
    frames = [pd.read_csv('./outputs/annuals/{}_osus_crisp.csv'.format(year)) for year in years]
    osus = pd.concat(frames).set_index('year',drop=True)
    osus.index= pd.to_datetime(osus.index.astype(str), format='%Y')
    fig, ax = plt.subplots(figsize=(8,4))
    for idx, gp in osus.groupby('company'):
        gp.plot(ax=ax,label=idx)
    plt.grid(); plt.xlabel('Year'); plt.ylabel('OSUS');plt.legend(osus.groupby('company').indices)
    fig.savefig('./outputs/final_chart_output.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # print('Creating Normalization Curves')
    # Fuzzification.create_normalization_curves(struct)
    
    years = range(2014,2019)
    # for year in years: 
    #     print('\n'+year)
    #     basic_to_secondary(year)
    #     secondary_to_primary(year)
    #     primary_to_osus(year)
    #     defuzzification(year)
    
    make_final_plot(years)
    print('Finished')
