import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
plt.style.use('seaborn-whitegrid')

dx = 0.001 # dx for linguistic variable function
nd = 3

pkg_dir = 'C:\\Users\\tzave\\OneDrive - The Cooper Union for the Advancement of Science and Art\\102-cooper\\150-masters\\sustainability\\sustainability_project1\\IOU-Sustainability-Model'

  
class Fuzzification:

    @classmethod
    def main(cls,primary_indicator,secondary_indicator, basic_indicators,indicator_type,year, sensitivity_ind, delta):       
        
        wms = cls.wms()
        pi_db = pd.read_excel('./databases/indicator_db.xlsx', sheet_name=primary_indicator).round(nd)
        
        #FUZZIFICATION
        frames = []
        for basic_indicator in basic_indicators: 
            df = pd.read_csv('./outputs/normalization_curves/{}.csv'.format(basic_indicator)).round(nd)
            s = pd.Series(data=df.iloc[:,1].values,index=df.iloc[:,0].values)
            df_base = pi_db[pi_db[indicator_type]==basic_indicator].drop(['raw_value_units','intensive_units','source'],axis='columns').reset_index(drop=True) #reset index so that it concats properly
            z = df_base['intensive_value'].values
            
            x = s.loc[z] # pass through normalization curve
            
            if sensitivity_ind != None: #if its a sensitivity analysis, perturb the normalized value
                assert delta != 0, 'Something may be wrong, you indicated sensitivity analysis but enterd no perturbation'
                if basic_indicator == sensitivity_ind:
                    x = (x + delta).round(nd)
                    x[x>1]=1 #fix any values that were made greater than 1 or less than 0
                    x[x<0]=0
                    print('Perturbed {} by {}'.format(basic_indicator, delta))
            
            wms_v = wms.loc[x.values].reset_index(); #fuzzify normalzed values and reset index to make concat work
            df_out = pd.concat([df_base,wms_v],axis='columns')
            assert df_base.shape[0] == df_out.shape[0], 'different amount of rows, error'
            frames.append(df_out)
        fuzz = pd.concat(frames).reset_index(drop=True)
        if year == 2018: #only have to do this once
            fuzz.to_csv('./outputs/annuals/{}_{}_{}_intensive_normalized_basic_indicators.csv'.format(delta, primary_indicator, secondary_indicator))
        
        #before inference, fix missing values based on flag in raw data column
        fuzz = cls.missing_values(fuzz)
        
        #INFERENCE
        frames=[]
        for company in fuzz['company'].unique():    
            company_year = fuzz[(fuzz['company'] == company) &
                                (fuzz['year'] == year)]
            b_indicators = [company_year[company_year[indicator_type]==indicator].iloc[0] for indicator in basic_indicators]
            frames.append(InferenceEngine.b_s(b_indicators)) #apply inference engine
        secondary = pd.DataFrame(frames)
        return secondary
    
    @staticmethod
    def missing_values(in_fuzz):
        """
        todo: Implement missing values. if a secondary indicator is missing all basic indicator values. apply the missing values method at the second indicator level.
        """
        
        out_fuzz = in_fuzz
        
        return out_fuzz
    
    @classmethod
    def create_normalization_curves(cls,struct):
        """
        Creates normalization curves for ALL data statistically.
        """
        print('CREATING NORMALIZATION CURVES')
        for primary_indicator in struct.keys():
            pi_db = pd.read_excel('./databases/indicator_db.xlsx', sheet_name=primary_indicator).round(nd)
            for secondary_indicator in struct[primary_indicator].keys():
                # CONSTRUCT NORMALIZATION CURVES
                for basic_indicator in struct[primary_indicator][secondary_indicator].keys(): 
                    #print(secondary_indicator,basic_indicator)
                    assert pi_db[pi_db['basic']==basic_indicator].shape[1] == 9 , 'Warning, not expected shape' #sensitve to number of companies
                    basic_df = pi_db[pi_db['basic']==basic_indicator]
                    iv = basic_df['intensive_value']
                    discourse = np.arange(iv.min(),
                                          iv.max()+dx,
                                          dx).round(nd)
                    x = cls.norm_type(iv=iv,
                                      typeofnorm=struct[primary_indicator][secondary_indicator][basic_indicator],
                                      discourse=discourse)
                    ncurve = pd.Series(data=x.round(nd),
                                        index=discourse.round(nd),
                                        name=basic_indicator)
                    ncurve.to_csv('./outputs/normalization_curves/{}.csv'.format(basic_indicator))
                    
                    # PLOT/SAVE NORMALIZATION CURVES
                    fig = plt.figure(figsize=(10,5));plt.title('Normalization Curve: {}'.format(basic_indicator));
                    ncurve.plot();
                    plt.grid();plt.xlabel('z [{}]'.format(pi_db[pi_db['basic']==basic_indicator].iloc[0]['intensive_units']));plt.ylabel('x');
                    fig.savefig('./outputs/normalization_curves/{}.png'.format(basic_indicator), dpi=300, bbox_inches='tight')
                    plt.close(fig)
    
    @classmethod
    def norm_type(cls,iv,typeofnorm,discourse):
        
        if typeofnorm == 'lower_is_better':
            x = cls.trapezoid(z= discourse,
                              c_l=iv.min(),
                              tc=iv.min(),
                              Tc=iv.min(),
                              c_u=iv.max());
    
        elif typeofnorm == 'middle_is_better':
            x = cls.trapezoid(z= discourse,
                              c_l=iv.min(),
                              tc=iv.mean(),
                              Tc=iv.mean(),
                              c_u=iv.max());
            
        elif typeofnorm == 'higher_is_better':
            x = cls.trapezoid(z=discourse,
                              c_l=iv.min(),
                              tc=iv.max(),
                              Tc=iv.max(),
                              c_u=iv.max())
        else:
            raise NameError('Not a recognized norm_type')
        return x
    
    @classmethod
    def create_linguistic_variables(cls):
        ling_dir = os.path.join(pkg_dir, 'outputs/linguistic_variables')
        if not os.path.exists(ling_dir): os.makedirs(ling_dir)
        
        cls.wms().to_csv(os.path.join(ling_dir,'wms.csv'))
        cls.vbbagvg().to_csv(os.path.join(ling_dir,'vbbagvg.csv'))
        cls.elvllflifhhvheh().to_csv(os.path.join(ling_dir,'elvllflifhhvheh.csv'))    
        
    @classmethod
    def trapezoid(cls,z,c_l,tc,Tc,c_u):
        """
        Trapezoid function that can be used to create linguistic values or normalization curves.
        
        Rule #1 of Fuzzy System: Completeness of Inputs - Make sure z covers the full universe of discourse.
        """
        
        if (c_l==tc) and (c_l==Tc): # DECREASING LINE
            out = np.piecewise(z,
                               [z<c_l, z>c_u, (z>=Tc)&(z<=c_u)],
                               [0    , 0    , lambda z: -z/(c_u-c_l) + c_u/(c_u-c_l)])
            
        elif (c_u==tc) and (c_u==Tc): # INCREASING LINE
            out = np.piecewise(z,
                               [z<c_l, z>c_u, (z>=c_l)&(z<=tc)],
                               [0    , 0    , lambda z: z/(c_u-c_l) - c_l/(c_u-c_l)])

        else: # TRAPEZOID AND TRIANGLE
            out = np.piecewise(z,
                               [z<=c_l, z>=c_u, (z>c_l)&(z<tc)            , (z>=tc)&(z<=Tc), (z>Tc)&(z<c_u)],
                               [0     , 0     , lambda z: (z-c_l)/(tc-c_l), 1              , lambda z: (c_u-z)/(c_u-Tc)])
        out = out.round(nd) # critical - get rid of annoying decimals
        return out 
    
    @classmethod
    def wms(cls):
        """
        Linguistic Variable WMS
        """
        x= np.arange(0,1+dx,dx).round(nd)

        medium_val = 0.7 
        df = pd.DataFrame(data = {'w':Fuzzification.trapezoid(x,0,0,0,medium_val),
                                  'm':Fuzzification.trapezoid(x,0,medium_val,medium_val,1),
                                  's':Fuzzification.trapezoid(x,medium_val,1,1,1)},
                          index = x)
        Fuzzification.check_ruspini_partition(df)
        Fuzzification.check_consistency(df)
        return df.round(nd)
    
    @classmethod
    def vbbagvg(cls):
        """
        Linguistic Variable VBBAGVG
        """
        x=np.arange(0,1+dx,dx).round(nd)
        d = 0.25
        data = {}
        for i,l in enumerate(['vb','b','a','g','vg']):
            data[l] =Fuzzification.trapezoid(x,d*(i-1),d*(i),d*(i),d*(i+1))
        df = pd.DataFrame(data = data,index = x)
        cls.check_ruspini_partition(df)
        cls.check_consistency(df)
        return df.round(nd)
    
    @classmethod
    def elvllflifhhvheh(cls):
        """
        Linguistic Variable elvllflifhhvheh
        """
        x=np.arange(0,1+dx,dx).round(nd)
        d = 0.125
        data = {}
        for i,l in enumerate(['el','vl','l','fl','i','fh','h','vh','eh']):
            data[l] = Fuzzification.trapezoid(x,d*(i-1),d*(i),d*(i),d*(i+1))
        df = pd.DataFrame(data = data,index = x)
        cls.check_ruspini_partition(df)
        cls.check_consistency(df)
        return df.round(nd)
    
    @staticmethod
    def check_ruspini_partition(df):
        """
        Special case of Rule #2 of Fuzzy System: Consistency of Unions for WMS and VBBAGVG
        """
        assert any(df.sum(axis='columns').apply(lambda x: round(x,2)==1)) == True , 'Ruspini Partition Not Satisfied'
    
    @staticmethod
    def check_consistency(df):
        """
        Rule #2 of Fuzzy System: Consistency of Unions - for any input, the membership functions of all the fuzzy sets it belongs too should be less than or equal to 1.
        """
        assert any(df.sum(axis='columns').apply(lambda x: round(x,2)<=1)) == True , 'Not Consistent'


class InferenceEngine:

    @classmethod
    def s_p(cls,secondary_indicators,primary_ind_name,indictor_type):
        """
        SECONDARY TO PRIMARY INFERENCE ENGINE & PRIMARY TO OSUS
        *** ONLY WORKS FOR 2 AND 3 SECONDARY INDICATOR INPUTS
        """
        #checks
        assert all(i['company']==secondary_indicators[0]['company'] for i in secondary_indicators), 'secondary indicators included dont have the same company'
        secondary_ind_company = secondary_indicators[0]['company']     
    
    
        assert all(i['year']==secondary_indicators[0]['year'] for i in secondary_indicators), 'secondary indicators included dont have the same year'
        secondary_ind_year = secondary_indicators[0]['year']
    
        #search the rule base for the secondary indicator
        rb = pd.read_excel('./databases/rulebase.xlsx', sheet_name=primary_ind_name,skiprows=13)
        lval = []
        indices = []
    
        if len(secondary_indicators)==2: #if there are two secondary indicators
            for in1_lv in ['vb','b','a','g','vg']:
                for in2_lv in ['vb','b','a','g','vg']:
                    lval.append(secondary_indicators[0][in1_lv] * secondary_indicators[1][in2_lv]) #LARSEN IMPLICATION-sensitive to the number of indicators
                    rule = rb[(rb[secondary_indicators[0][indictor_type]] == in1_lv) &
                              (rb[secondary_indicators[1][indictor_type]] == in2_lv)] #selecting the right row from the rulebase, sensitive to the number of indicators
                    indices.append(rule[primary_ind_name].iloc[0]) #
    
        elif len(secondary_indicators)==3: #if there are three secondary indicators
            for in1_lv in ['vb','b','a','g','vg']:
                for in2_lv in ['vb','b','a','g','vg']:
                    for in3_lv in ['vb','b','a','g','vg']:
                        lval.append(secondary_indicators[0][in1_lv] * secondary_indicators[1][in2_lv] * secondary_indicators[2][in3_lv]) #LARSEN IMPLICATION-sensitive to the number of indicators
                        rule = rb[(rb[secondary_indicators[0][indictor_type]] == in1_lv) &
                                  (rb[secondary_indicators[1][indictor_type]] == in2_lv) &
                                  (rb[secondary_indicators[2][indictor_type]] == in3_lv)] #selecting the right row from the rulebase, sensitive to the number of indicators
                        indices.append(rule[primary_ind_name].iloc[0]) #    
        else:
            print('WARNING, THER ARE SOME OTHER NUMBER OF secondary INDICATORS')
    
        s = pd.Series(data=lval,
                      index=indices).round(nd)
        s = s.groupby(s.index).sum() #add up all vals with same linguistic var
    
        s['primary'] = primary_ind_name
        s['year'] = secondary_ind_year
        s['company'] = secondary_ind_company
        return s
       
    @classmethod
    def b_s(cls,basic_indicators):
        """
        BASIC TO SECONDARY INFERENCE ENGINE

        *** ONLY FOR 2 BASIC INDICATOR INPUTS
        """
        #checks
        assert all(i['company']==basic_indicators[0]['company'] for i in basic_indicators), 'basic indicators included dont have the same company'
        secondary_ind_company = basic_indicators[0]['company']     

        assert all(i['secondary']==basic_indicators[0]['secondary'] for i in basic_indicators), 'basic indicators included dont have the same secondary indicator'
        secondary_ind_name = basic_indicators[0]['secondary'] #status

        assert all(i['year']==basic_indicators[0]['year'] for i in basic_indicators), 'basic indicators included dont have the same year'
        secondary_ind_year = basic_indicators[0]['year']

        #search the rule base for the secondary indicator
        rb = pd.read_excel('./databases/rulebase.xlsx', sheet_name=secondary_ind_name,skiprows=13)
        lval = []
        indices = []
        
        if len(basic_indicators)==2: #if there are two basic indicators
            for in1_lv in ['w','m','s']:
                for in2_lv in ['w','m','s']:
                    lval.append(basic_indicators[0][in1_lv] * basic_indicators[1][in2_lv]) #LARSEN IMPLICATIONsensitive to the number of indicators
                    rule = rb[(rb[basic_indicators[0]['basic']] == in1_lv) &
                              (rb[basic_indicators[1]['basic']] == in2_lv)] #selecting the right row from the rulebase, sensitive to the number of indicators
                    indices.append(rule[secondary_ind_name].iloc[0]) #
                    
        elif len(basic_indicators)==3: #if there are three basic indicators
            for in1_lv in ['w','m','s']:
                for in2_lv in ['w','m','s']:
                    for in3_lv in ['w','m','s']:
                        lval.append(basic_indicators[0][in1_lv] * basic_indicators[1][in2_lv] * basic_indicators[2][in3_lv]) #LARSEN IMPLICATIONsensitive to the number of indicators
                        rule = rb[(rb[basic_indicators[0]['basic']] == in1_lv) &
                                  (rb[basic_indicators[1]['basic']] == in2_lv) &
                                  (rb[basic_indicators[2]['basic']] == in3_lv)] #selecting the right row from the rulebase, sensitive to the number of indicators
                        indices.append(rule[secondary_ind_name].iloc[0]) #    
        else:
            print('WARNING, THERE ARE SOME OTHER NUMBER OF BASIC INDICATORS')
                
        s = pd.Series(data=lval,index=indices).round(nd)
        s = s.groupby(s.index).sum() #add up all similar vals
        s['secondary'] = secondary_ind_name
        s['year'] = secondary_ind_year
        s['company'] = secondary_ind_company
        return s[['secondary','company','year','vb','b','a','g','vg']]
   
