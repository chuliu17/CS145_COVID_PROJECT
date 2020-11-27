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




#RUN THIS FIRST : TRAIN THE MODEL AND STORE IN DIC

import numpy as np
import pandas as pd
import sys
import random as rd
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2


import statistics
from hw1code.linear_regression import LinearRegression
from collections import defaultdict

lm=LinearRegression()
model_dict = defaultdict(str)
for state in states:
    for target in ["Confirmed", "Deaths"]:
        lin_reg_name = state + target
        print(lin_reg_name)
        lm.load_project_data('./data/train.csv', state, target)
        #print('Training data shape: ', lm.train_x.shape)
        #print('Training labels shape:', lm.train_y.shape)

        training_error= 0

        closed_form_beta = lm.train('0')
        predicted_train_y = lm.predict(lm.train_x,closed_form_beta)

        training_error = lm.compute_mse(predicted_train_y,lm.train_y)
        print('Training error is: ', training_error)



        plt.plot(predicted_train_y)
        plt.plot(lm.train_y.to_numpy())
        plt.ylabel('{} cases in {}'.format(target, state))
        plt.xlabel('Days since 2020/01/01')
        plt.show()



        #store the model
        model_dict[lin_reg_name] = lm
        print("\n")

print(model_dict)




#RUN THIS SECOND : MAKE PREDICTIONS AND WRITE TO CSV


submission = "ForecastID,Confirmed,Deaths\n"

basedate = pd.Timestamp('2020-01-01')
startdate = pd.Timestamp('2020-09-01')
enddate = pd.Timestamp('2020-09-26')

start_day = (startdate - basedate).days
end_day = (enddate - basedate).days
test_days = range(start_day, end_day+1)


train_start = pd.Timestamp('2020-04-12')
train_end = pd.Timestamp('2020-08-31')
train_start_day = (train_start - basedate).days
train_end_day = (train_end - basedate).days
train_days = range(train_start_day, train_end_day+1 )
print(train_days)

std = statistics.stdev(train_days)
mean = statistics.mean(train_days)
print(std)
print(mean)

days_normalized = [(day-mean)/std for day in test_days]

#print(days_normalized)
ForecastID = 0
for day in days_normalized:
    for state in states:
        submission = submission + str(ForecastID) + ","
        ForecastID +=1
        for target in ["Confirmed", "Deaths"]:
            lin_reg_name = state + target
            lm = model_dict[lin_reg_name]
            day_arr = [day]
            numpy_day = np.array(day_arr).reshape((1,1))
            #print(day)
            #print(numpy_day)
            predicted = lm.predict(numpy_day,lm.beta)
            predicted_num = int(predicted[0])
            print(lin_reg_name)
            print(predicted_num)
            submission = submission + str(predicted_num)
            if target == "Confirmed":
                submission += ","
            else:
                submission += "\n"


f = open('normalized_submission.csv','w')
f.write(submission) #Give your csv text here.
## Python will convert \n to os.linesep
f.close()
