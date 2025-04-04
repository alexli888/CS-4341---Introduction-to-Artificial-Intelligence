import pandas as pd
import numpy as np
'''Write your code here '''
df = pd.read_csv("/Users/alexli/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/CS 4341 Introduction to Artificial Intelligence/CS-4341---Introduction-to-Artificial-Intelligence/Project 2/Life Expectancy Data.csv")

'''end of student code'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
total_df = df[['Year', 'GDP', 'Adult Mortality', 'Alcohol', ' BMI ', 'Schooling', 'Life expectancy ', 'Status', 'Country']]
total_df = total_df.dropna()

for status in ["Developing", "Developed"]:
    x_dim = 5
    #Step 1:You should define the train_x, each row of it represents a year of the 5 features of a country with 5 columns. The Year column should be used to select the samples.
    #Step 2:Define train_y, each row of it represents a year of Life expectancy of a country. The Year column should be used to select the samples.
    #Step 3:Define a LinearRegression model, and fit it using train_X and train_y.
    #Step 4:Calculate rmse and r2_score using the fitted model.
    #Step 5:Print the coefficients of the linear regression model
    '''Write your code here '''


    '''end of student code'''
    print(f'Status = {status}, Training data, RMSE={rmse:.3f}, R2={r2_score:.3f}')
    for feature_i, feature in enumerate(['GDP', 'Adult Mortality', 'Alcohol', ' BMI ', 'Schooling']):
        print(f'coef for {feature} = {model.coef_[feature_i]:.7f}')
    #Step 1: Define test_x and test_y by selecting the remaining years of the data
    #Step 2: Use model.predict to generate the prediction
    #Step 3: Calculate rmse and r2_score on test_x and test_y.
    '''Write your code here '''


    '''end of student code'''
    print(f'Status = {status}, Testing data, RMSE={rmse:.3f}, R2={r2_score:.3f}')