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
                                'dividend payout ratio':'higher_is_better'}, 
                    'response':{'debt to equity ratio':'higher_is_better',
                                'operating expenses':'higher_is_better',
                                'effective tax rate':'higher_is_better'}
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

#=============================================================================
## secondary TO SECONDARY
frames2 = []
for primary_indicator in struct.keys():
#     if (primary_indicator == 'wealth') or (primary_indicator == 'ecos'): 
#         continue # skip for now
    frames = []
    for secondary_indicator in struct[primary_indicator]:
        print('{}-{}'.format(primary_indicator ,secondary_indicator))
        frames.append(Fuzzification.primary(primary_indicator=primary_indicator,
                                            basic_indicators=struct[primary_indicator][secondary_indicator],
                                            indicator_type='basic'))
    secondary_agg = pd.concat(frames,axis='index').reset_index(drop=True)
    secondary_agg.to_excel('./intermediate_outputs/{}_secondary_agg.xlsx'.format(primary_indicator))
    frames2.append(secondary_agg)

#=============================================================================
 
#SECONDARY TO PRIMARY
secondary_agg = pd.read_excel('./intermediate_outputs/wealth_secondary_agg.xlsx')

indicator_type = 'secondary'
#INFERENCE
frames2=[]
for company in secondary_agg['company'].unique():
    frames = []
    for year in secondary_agg['year'].unique():
        company_year = secondary_agg[(secondary_agg['company'] == company) &
                                     (secondary_agg['year'] == year)]
        #print(year,company)
        s_indicators = [company_year[company_year[indicator_type]==indicator].iloc[0] for indicator in secondary_agg[indicator_type].unique()]

        frames.append(InferenceEngine.s_p(s_indicators)) #apply inference engine
    frames2.append(pd.concat(frames,axis='columns').T)
secondary = pd.concat(frames2).reset_index(drop=True)