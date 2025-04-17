import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Loading dataset........
df = pd.read_csv("/Users/alexli/Library/CloudStorage/OneDrive-WorcesterPolytechnicInstitute(wpi.edu)/CS 4341 Introduction to Artificial Intelligence/CS-4341---Introduction-to-Artificial-Intelligence/Project 2/Life Expectancy Data.csv")

df = df[['Year', 'GDP', 'Life expectancy ', 'Status', 'Country']].dropna()

results = {
    "Developing": {1: [], 2: [], 3: [], 4: []},
    "Developed": {1: [], 2: [], 3: [], 4: []}
}

# Iterate through each country in the dataset(AS its all country)
for country in df['Country'].unique():
    country_df = df[df['Country'] == country]
    status = country_df['Status'].iloc[0]

    for degree in [1, 2, 3, 4]:
        # SPLIT into train/test sets
        train_df = country_df[country_df['Year'] <= 2013]  # Training data: 2000-2013
        test_df = country_df[country_df['Year'] >= 2014]   # Testing data: 2014-2015

        # Skip if not enough data points
        if len(train_df) < degree + 1 or len(test_df) == 0:
            continue

        train_X = train_df[['GDP']].values
        train_y = train_df['Life expectancy '].values
        test_X = test_df[['GDP']].values
        test_y = test_df['Life expectancy '].values

        poly = PolynomialFeatures(degree=degree)
        train_X_poly = poly.fit_transform(train_X)
        test_X_poly = poly.transform(test_X)

        model = LinearRegression()
        model.fit(train_X_poly, train_y)

        train_preds = model.predict(train_X_poly)
        test_preds = model.predict(test_X_poly)

        # Metrics calculation w/ nan check
        train_rmse = np.sqrt(mean_squared_error(train_y, train_preds))
        train_r2 = r2_score(train_y, train_preds) if len(train_y) > 1 else np.nan  
        test_rmse = np.sqrt(mean_squared_error(test_y, test_preds))
        test_r2 = r2_score(test_y, test_preds) if len(test_y) > 1 else np.nan 

        # Append results only if R2 is valid 
        if not np.isnan(train_r2) and not np.isnan(test_r2):
            results[status][degree].append((train_rmse, train_r2, test_rmse, test_r2))



print("\n")
print(f"{'Status':<12} {'Degree':<7} {'Train RMSE':<12} {'Train R2':<10} {'Test RMSE':<12} {'Test R2'}")
print("-" * 65)
for status in ["Developing", "Developed"]:
    for degree in [1, 2, 3, 4]:
        metrics = results[status][degree]
        if metrics:
            avg_train_rmse = np.mean([m[0] for m in metrics])
            avg_train_r2 = np.mean([m[1] for m in metrics])
            avg_test_rmse = np.mean([m[2] for m in metrics])
            avg_test_r2 = np.mean([m[3] for m in metrics])
            print(f"{status:<12} {degree:<7} {avg_train_rmse:<12.3f} {avg_train_r2:<10.3f} {avg_test_rmse:<12.3f} {avg_test_r2:.3f}")
        else:
            print(f"{status:<12} {degree:<7} {'N/A':<12} {'N/A':<10} {'N/A':<12} {'N/A'}")
