import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

'''Write your code here '''
df = pd.read_csv("/Users/alexli/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/CS 4341 Introduction to Artificial Intelligence/CS-4341---Introduction-to-Artificial-Intelligence/Project 2/Life Expectancy Data.csv")
'''end of student code'''

total_df = df[df.Country == 'Afghanistan'][['Year', 'GDP', 'Life expectancy ']].dropna()

for degree in [1, 2, 3, 4]: # 1 linear, 2 quadratic, 3 cubic, 4 quartic
    
    # Step 1: You should define the train_x, each row of it represents a year of GDP of Afghanistan,
    # and each column of it represents a power of the GDP. The Year column should be used to select the samples.
    train_df = total_df[total_df['Year'] < total_df['Year'].quantile(0.8)]
    train_X = train_df[['GDP']].values  # GDP as independent variable
    train_y = train_df['Life expectancy '].values  # Life expectancy as dependent variable

    # Step 2: Define train_y, each row of it represents a year of Life expectancy of Afghanistan.
    # The Year column should be used to select the samples.
    test_df = total_df[total_df['Year'] >= total_df['Year'].quantile(0.8)]
    test_X = test_df[['GDP']].values
    test_y = test_df['Life expectancy '].values

    # Step 3: Define a LinearRegression model, and fit it using train_X and train_y.
    poly = PolynomialFeatures(degree=degree)
    train_X_poly = poly.fit_transform(train_X)  # Transform GDP to polynomial features
    test_X_poly = poly.transform(test_X)  

    model = LinearRegression()
    model.fit(train_X_poly, train_y)

    # Step 4: Calculate rmse and r2_score using fitted model.
    train_predictions = model.predict(train_X_poly)
    train_rmse = np.sqrt(mean_squared_error(train_y, train_predictions))
    train_r2 = r2_score(train_y, train_predictions)

    print(f'Train set, degree={degree}, RMSE={train_rmse:.3f}, R2={train_r2:.3f}')
    # feel free to change the variable name if needed. 
    # DO NOT change the output format.

    # Step 1: Define test_x and test_y by selecting the remaining years of the data
    # Step 2: Use model.predict to generate the prediction
    # Step 3: Calculate rmse and r2_score on test_x and test_y.
    test_predictions = model.predict(test_X_poly)
    test_rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
    test_r2 = r2_score(test_y, test_predictions)

    print(f'Test set,  degree={degree}, RMSE={test_rmse:.3f}, R2={test_r2:.3f}')
