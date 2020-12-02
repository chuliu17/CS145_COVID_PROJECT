import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def getDataFrames(trainFilePath, testFilePath, state, target_y):
    df = pd.read_csv(trainFilePath, parse_dates=['Date'], index_col=['Date'])
    state_df = df[df['Province_State']==state]
    temp_df = state_df.drop(['Province_State', 'Recovered', 'Active', 'Incident_Rate', 'People_Tested', 'People_Hospitalized', 'Mortality_Rate', 'Testing_Rate', 'Hospitalization_Rate'], axis=1)
    final_train_df = pd.DataFrame()
    if target_y == 'Confirmed':
        final_train_df = temp_df.drop(['Deaths'], axis=1)
    else:
        final_train_df = temp_df.drop(['Confirmed'], axis=1)

    final_train_df.index = pd.DatetimeIndex(final_train_df.index.values, freq=final_train_df.index.inferred_freq)

    
    df = pd.read_csv(testFilePath, parse_dates=['Date'], index_col=['Date'])
    state_df = df[df['Province_State']==state]
    temp_df = state_df.drop(['ForecastID', 'Province_State'], axis=1)
    final_test_df = pd.DataFrame()
    if target_y == 'Confirmed':
        final_test_df = temp_df.drop(['Deaths'], axis=1)
    else:
        final_test_df = temp_df.drop(['Confirmed'], axis=1)

    final_test_df.index = pd.DatetimeIndex(final_test_df.index.values, freq=final_test_df.index.inferred_freq)
        
    return final_train_df, final_test_df
    


class ArimaImpl(object):
    def __init__(self):
        self.training = pd.DataFrame()
        self.testing = pd.DataFrame()

    def load_project_data(self, train_file, test_file, state, target_y):
        self.training, self.testing = getDataFrames(train_file, test_file, state, target_y)

    def train(self, target_y, p, d, q):
        model = ARIMA(self.training[target_y], order=(p, d, q))
        model_fit = model.fit()
        return model_fit

    def test(self, target_y, model_fit):
        start = len(self.training)
        end = start + len(self.testing) - 1

        pred = model_fit.predict(start, end, typ = 'levels')
        return pred

    def MAPE(self, target_y, predicted):
        return np.mean(np.abs((np.array(self.testing[target_y].tolist()) - np.array(predicted.values.tolist())) / np.array(self.testing[target_y].tolist()))) * 100

