#Global variables:
states = [
"Alabama",
"Alaska",
"Arizona",
"Arkansas",
"California",
"Colorado",
"Connecticut",
"Delaware",
"Florida",
"Georgia",
"Hawaii",
"Idaho",
"Illinois",
"Indiana",
"Iowa",
"Kansas",
"Kentucky",
"Louisiana",
"Maine",
"Maryland",
"Massachusetts",
"Michigan",
"Minnesota",
"Mississippi",
"Missouri",
"Montana",
"Nebraska",
"Nevada",
"New Hampshire",
"New Jersey",
"New Mexico",
"New York",
"North Carolina",
"North Dakota",
"Ohio",
"Oklahoma",
"Oregon",
"Pennsylvania",
"Rhode Island",
"South Carolina",
"South Dakota",
"Tennessee",
"Texas",
"Utah",
"Vermont",
"Virginia",
"Washington",
"West Virginia",
"Wisconsin",
"Wyoming",
]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics
from arima import ArimaImpl
import csv


ar = ArimaImpl()
model_dict = defaultdict(str)
total_mape = 0
stateID = -1
prediction_confirmed = []
prediction_deaths = []
forecast_final = []


best_params_confirmed = []
best_params_deaths = []

with open('bestparamsconfirmed.csv', 'r') as r:
    csv_reader = csv.reader(r)
    best_params_confirmed = list(csv_reader)
    best_params_confirmed = [[int(element) for element in row] for row in best_params_confirmed]

with open('bestparamsdeaths.csv', 'r') as r:
    csv_reader = csv.reader(r)
    best_params_deaths = list(csv_reader)
    best_params_deaths = [[int(element) for element in row] for row in best_params_deaths]



for state in states:
    stateID += 1
    for target in ["Confirmed", "Deaths"]:
        ar.load_project_data('../data/train.csv', '../data/test.csv', state, target)
        
        params = []
        if target == "Confirmed":
            params = best_params_confirmed[stateID]
        else:
            params = best_params_deaths[stateID]

        model_fit = ar.train(target, params[0], params[1], params[2])        
        predictions = ar.test(target, model_fit)
        
        forecastID = []
        for i in range(len(predictions)):
            forecastID.append((stateID + (50 * i)))
        
        if target == "Confirmed":
            prediction_confirmed.extend(predictions)
            forecast_final.extend(forecastID)
        else:
            prediction_deaths.extend(predictions)
        

submission_df = pd.DataFrame()
submission_df['ForecastID'] = forecast_final
submission_df['Confirmed'] = prediction_confirmed
submission_df['Deaths'] = prediction_deaths

submission_df = submission_df.sort_values(by = ['ForecastID'])

submission_df = submission_df.set_index('ForecastID')

submission_df.to_csv('submission.csv')


