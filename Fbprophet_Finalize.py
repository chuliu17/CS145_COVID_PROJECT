import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from sklearn.model_selection import ParameterGrid
import itertools

# prepare and clean data
df = pd.read_csv('data/train_round2.csv', index_col=None)
df = df[['ID','Province_State','Date','Confirmed','Deaths']] #choose the columns needed
df['Date'] = pd.to_datetime(df['Date'])

# df1 for predicting confirmed cases
df1 = df.rename(columns = {'Date':'ds', 'Confirmed': 'y'})

# df2 for predicting death cases
df2 = df.rename(columns = {'Date':'ds', 'Deaths': 'y'})
#df1,df2

States = df['Province_State'].drop_duplicates()
#len(States)

def generate_ForecastID(df_pred, target_str, start_date, state, States):
    '''
    df_pred: predicted data of one state from fbprophet 
    target_str: for renaming 'yhat'
    start_date: desired cutoff date
    States: list of states
    state: state of df_pred
    '''
    df = df_pred[['ds', 'yhat']]
    # select data after start_date
    df = df[df['ds'] >= start_date ].reset_index(drop = True) #'2020-12-07'

    #push index to first column and use 'index' column to generate 'ForecastID'
    df = df.reset_index() 
    df['index'] = df['index']*50 + States.index(state)

    df = df.rename(columns = {'index':'ForecastID', 'yhat': target_str }) 
    
    return df

def get_bestParameters(df, all_params):
    '''
    df: data of one state, has 'ds' and 'y'
    
    
    ## simple hyperparameter tuning
    '''
    mapes = []  # Store the RMSEs for each params here
    min_MAPE = 1000
    
    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = Prophet(**params).fit(df)  # Fit model with given params
        df_cv = cross_validation(m, initial='30 days', period='30 days', horizon = '30 days')
        df_p = performance_metrics(df_cv, rolling_window=1)
        mape = df_p['mape'].values[0]
        mapes.append(mape)

        #find min mape and best parameters
        if min_MAPE > mape:
            min_MAPE = mape
            best_params = params

    # keep track of tuning results
    tuning_results = pd.DataFrame(all_params)
    tuning_results['mape'] = mapes
    
    return best_params
    


# ### Hyperparameter Tuning
# Generate more model
# Generate all combinations of parameters
from sklearn.model_selection import ParameterGrid
params_grid = {'seasonality_mode':['multiplicative','additive'], 
               'changepoint_prior_scale': [0.005, 0.01, 0.05, 0.5],  # default 0.05, reasonable range [0.001, 0.5]
               'seasonality_prior_scale': [0.01, 0.05, 0.1, 1, 10.0],  # default 10, reasonable range [0.01, 10]
               'changepoint_range' : [0.8, 0.85, 0.9, 0.95] 
              }

all_params = ParameterGrid(params_grid)
cnt = 0
for params in all_params:
    #print(params)
    cnt = cnt+1
    
print('Total Possible Models',cnt)


# **NOTE**
# 
# Since we have possible models upto 160, we need to run 160 * 7(cross validation) * 2 (cases/death cases) * 50 (states). Approximately 66.6 hours in total. So we want to break up the following for loop into several one and run on differnt kernels to save time.

'''
Uncomment the part that you would like to run
'''

### df1 => predict confirmed cases
# Youjun

'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[:5]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed0.csv', index = False) 
'''


'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[5:10]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed1.csv', index = False) 
'''


'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[10:15]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed2.csv', index = False) 
'''


'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[15:20]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed3.csv', index = False) 
'''


'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[20:25]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed4.csv', index = False) 
'''


# Yu-Hsuan
'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[25:30]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed5.csv', index = False) 
'''


'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[30:35]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed6.csv', index = False) 
'''


'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[35:40]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed7.csv', index = False) 
'''


'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[40:45]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed8.csv', index = False) 
'''



'''
df1_all_states = [] #store predicted data for each state
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[45:50]: #test for just one state

    # training data: 2020-04-12 to 2020-11-22
    # predict confirmed cases
    df1_one_state = df1[df1['Province_State'] == state]
    
    
    best_params1 = get_bestParameters(df1_one_state, all_params)
    m1 = Prophet(**best_params1).fit(df1_one_state) # fit data to model
    
    future = m1.make_future_dataframe(periods=21, freq = 'D') #predict the data from 11/23 - 12/13
    forecast1 = m1.predict (future) # predict
    
    df1_one_state_pred = generate_ForecastID(forecast1, 'Confirmed', start_date, state,States)  #generate ForecastID
    df1_all_states.append(df1_one_state_pred)

df1_pred = pd.concat(df1_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df1_pred.to_csv('output/confirmed9.csv', index = False) 
'''


## df2 => predict death cases

# Raghav
'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[:5]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death0.csv', index = False) 
'''


'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[5:10]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death1.csv', index = False) 
'''


'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[10:15]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death2.csv', index = False) 
'''


'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[15:20]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death3.csv', index = False) 
'''


'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[20:25]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death4.csv', index = False) 
'''


# Saurav
'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[25:30]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death5.csv', index = False) 
'''


'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[30:35]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death6.csv', index = False) 
'''


'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[35:40]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death7.csv', index = False) 
'''


'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[40:45]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death8.csv', index = False) 
'''


'''
df2_all_states = []
States = df['Province_State'].drop_duplicates().tolist()
start_date = '2020-12-07'
for state in States[45:50]: #test for just one state
    # training data: 2020-04-12 to 2020-11-22
    # predict death cases
    df2_one_state = df2[df2['Province_State'] == state]
    
    best_params2 = get_bestParameters(df2_one_state, all_params)
    m2 = Prophet(**best_params2).fit(df2_one_state)
    
    future = m2.make_future_dataframe(periods=21, freq = 'D')#predict the data from 11/23 - 12/13
    forecast2 = m2.predict (future)
    
    df2_one_state_pred = generate_ForecastID(forecast2, 'Deaths', start_date, state,States) 
    df2_all_states.append(df2_one_state_pred)

df2_pred = pd.concat(df2_all_states, ignore_index=True).sort_values(by=['ForecastID'])
df2_pred.to_csv('output/death9.csv', index = False) 
'''

