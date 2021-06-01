"""Import required libraries"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

"""Import data"""
df = pd.read_csv("final_embedded.csv", index_col=0)
df = df.join(pd.get_dummies(df['target'])).drop('target', axis=1)  # one hot encoding subreddits

"""Division into training and test samples"""
X, y = df.drop(columns=['score']), df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

"""
Regression
LinearRegression
"""
pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('LinearRegression', LinearRegression())
    ]
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("MSE", mean_squared_error(y_pred, y_test))
print("R^2", r2_score(y_pred, y_test))

"""RandomForestRegressor"""
pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('RandomForestRegressor', RandomForestRegressor())
    ]
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("MSE", mean_squared_error(y_pred, y_test))
print("R^2", r2_score(y_pred, y_test))

"""GradientBoostingRegressor"""
pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('GradientBoostingRegressor', GradientBoostingRegressor())
    ]
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("MSE", mean_squared_error(y_pred, y_test))
print("R^2", r2_score(y_pred, y_test))