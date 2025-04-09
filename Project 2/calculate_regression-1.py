import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

'''Write your code here '''
df = pd.read_csv("/Users/alexli/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/CS 4341 Introduction to Artificial Intelligence/CS-4341---Introduction-to-Artificial-Intelligence/Project 2/Life Expectancy Data.csv")
'''end of student code'''

# Filter and clean the dataset
total_df = df[['Year', 'GDP', 'Adult Mortality', 'Alcohol', ' BMI ', 'Schooling', 'Life expectancy ', 'Status', 'Country']]
total_df = total_df.dropna()

# Features to use
features = ['GDP', 'Adult Mortality', 'Alcohol', ' BMI ', 'Schooling']

for status in ["Developing", "Developed"]:
    x_dim = 5
    # Step 1: Define the train_x, each row of it represents a year of the 5 features of a country with 5 columns. The Year column should be used to select the samples.
    # Step 2: Define train_y, each row of it represents a year of Life expectancy of a country. The Year column should be used to select the samples.
    # Step 3: Define a LinearRegression model, and fit it using train_X and train_y.
    # Step 4: Calculate rmse and r2_score using the fitted model.
    # Step 5: Print the coefficients of the linear regression model

    '''Write your code here '''
    status_df = total_df[total_df['Status'] == status]
    train_df = status_df[(status_df['Year'] >= 2000) & (status_df['Year'] <= 2013)]

    train_X = train_df[features]
    train_y = train_df['Life expectancy ']

    # Standardize features to avoid issues like overflow or division by zero
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)

    model = LinearRegression()
    model.fit(train_X_scaled, train_y)

    train_preds = model.predict(train_X_scaled)
    rmse = np.sqrt(mean_squared_error(train_y, train_preds))
    r2_score_train = r2_score(train_y, train_preds)

    '''end of student code'''
    print(f'Status = {status}, Training data, RMSE={rmse:.3f}, R2={r2_score_train:.3f}')
    for feature_i, feature in enumerate(features):
        print(f'coef for {feature} = {model.coef_[feature_i]:.7f}')

    # Step 1: Define test_x and test_y by selecting the remaining years of the data
    # Step 2: Use model.predict to generate the prediction
    # Step 3: Calculate rmse and r2_score on test_x and test_y.

    '''Write your code here '''
    test_df = status_df[(status_df['Year'] >= 2014) & (status_df['Year'] <= 2015)]
    test_X = test_df[features]
    test_y = test_df['Life expectancy ']

    test_X_scaled = scaler.transform(test_X)
    test_preds = model.predict(test_X_scaled)
    rmse = np.sqrt(mean_squared_error(test_y, test_preds))
    r2_score_test = r2_score(test_y, test_preds)

    '''end of student code'''
    print(f'Status = {status}, Testing data, RMSE={rmse:.3f}, R2={r2_score_test:.3f}')

    # THIS REG MODEL IS MEANT TO predict Libya life expectancy in 2010.
    libya_df = total_df[(total_df['Country'] == 'Libya') & (total_df['Year'] == 2010)]
    if not libya_df.empty:
        libya_X = libya_df[features]
        libya_X_scaled = scaler.transform(libya_X)
        libya_pred = model.predict(libya_X_scaled)[0]
        libya_actual = libya_df['Life expectancy '].values[0]
        print(f"[{status}] Predicted life expectancy for Libya in 2010: {libya_pred:.2f}")
        print(f"[{status}] Actual life expectancy for Libya in 2010: {libya_actual:.2f}")
    else:
        print(f"[{status}] Libya 2010 data not found.")
