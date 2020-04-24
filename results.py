# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:00:11 2020

@author: m4rz910
"""
import pandas as pd


def primary_indicator_tables():
    primary = []
    for year in range(2014,2019):
        df = pd.read_csv('./outputs/annuals/0_None_{}_primary_agg.csv'.format(year));
        primary.append(df)
    primary = pd.concat(primary)[['company','primary','year','vb','b','a','g','vg']]
    for idx, gp in primary.groupby('company'):
        gp.sort_values('primary').round(2).to_csv('./outputs/report_tables/{}_fuzz_primary.csv'.format(idx), index=False)
        

def secondary_indicator_tables():
    secondary = []
    for year in range(2014,2019):
        for primary in ['hums','wealth','ecos']:
            df = pd.read_csv('./outputs/annuals/0_None_{}_{}_secondary_agg.csv'.format(year,primary));
            secondary.append(df)
    secondary = pd.concat(secondary)[['company','secondary','year','vb','b','a','g','vg']]
    for idx, gp in secondary.groupby('company'):
        gp.sort_values(['secondary','year']).round(2).to_csv('./outputs/report_tables/{}_fuzz_secondary.csv'.format(idx), index=False)
        

if __name__ == '__main__':
    #primary_indicator_tables()
    secondary_indicator_tables()