from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
import joblib
import math
import mlflow

# getting the data

from influxdb import InfluxDBClient
import pandas as pd

client = InfluxDBClient(host = "influxus.itu.dk", port =8086, username = "lsda", password = "icanonlyread")
client.switch_database ("orkney")

def get_df(results):
    values = results.raw ["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame( values , columns = columns ) . set_index ("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime - index
    return df

# Get the last 90 days of power generation data
generation = client.query ("SELECT * FROM Generation where time > now()-90d") # Query written in InfluxQL

# Get the last 90 days of weather forecasts with the shortest lead time
wind = client.query("SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'") # Query written in InfluxQL

gen_df = get_df (generation)
wind_df = get_df (wind)

## Methods for custom transformer

# Convert direction for radian
def toRadian(x):
    match x:
        case 'E':
            return math.radians(0)
        case 'ENE':
            return math.radians(22.5)
        case 'NE':
            return math.radians(45)
        case 'NNE':
            return math.radians(67.5)
        case 'N':
            return math.radians(90)
        case 'NNW':
            return math.radians(112.5)
        case 'NW"':
            return math.radians(135)
        case 'WNW':
            return math.radians(157.5)
        case 'W':
            return math.radians(180)
        case 'WSW':
            return math.radians(202.5)
        case 'SW':
            return math.radians(225)
        case 'SSW':
            return math.radians(247.5)
        case 'S':
            return math.radians(270)
        case 'SSE':
            return math.radians(292.5)
        case 'SE':
            return math.radians(315)
        case 'ESE':
            return math.radians(337.5)


# Convert all directions in dataframe 'x' to radian
def dfToRadian(x):
    return x.applymap(toRadian)


# Perform linear interpolation on datafram 'x'
def linearInterpolation(x):
    return x.interpolate(method='linear')

plt.close("all")

# For aligning two dataframes on time
def mergeOnTime(df1, df2, delta):
    combined_df = pd.merge_asof(df1.iloc[:,2:], df2.iloc[:,[0,3]], on="time", tolerance=pd.Timedelta(delta))
    combined_df.set_index('time', inplace=True)
    return combined_df

combined_df = mergeOnTime(gen_df, wind_df, "3m")


def testSplit(df):
    X = df.drop("Total", axis = 1)
    y = df["Total"]
    tss = TimeSeriesSplit(n_splits = 3)
    for train_index, test_index in tss.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = testSplit(combined_df)

# Pipeline converting directions to radians, and performing linear interpolation to fill missing data
categorical_pipeline = Pipeline(steps=[
    ('one-hot', FunctionTransformer(dfToRadian)),
    ('interpolate', FunctionTransformer(linearInterpolation))
])

# Perform independent transformations on columns "Direction" and "Speed"
column_processor = ColumnTransformer(transformers=[
    ('categorical', categorical_pipeline, ["Direction"]),
    ('interpolate', FunctionTransformer(linearInterpolation), ["Speed"])
])

# Impute the final missing values and scale
full_processor = Pipeline(steps=[
    ('fill', column_processor),
    ('impute', SimpleImputer(strategy='most_frequent')),
    ("scale", MinMaxScaler())
])

#mlflow.set_tracking_uri('http://training.itu.dk:5000/')
#experiment = mlflow.set_experiment("magnj - tracking")

import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse


# Simular pipeline, but transforming PolynomialFeatures
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


degree = 4
with mlflow.start_run():
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    linReg = LinearRegression()

    polynominal_pipeline = Pipeline(steps=[
        ("preprocessor", full_processor),
        ('polynominal', poly),
        ('model', linReg)
    ])

    # Parameters poly:'degree': 4, 'include_bias': False, 'interaction_only': False, 'order': 'C'
    # Parameters linReg:'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False

    # Estimate error for polynominal pipeline
    poly_pipeline_model = polynominal_pipeline.fit(X_train, y_train)
    poly_preds = poly_pipeline_model.predict(X_test)
    (rmse, mae, r2) = eval_metrics(y_test, poly_preds)

    print("Polynominal linear regression with degree: " + str(degree))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("degree", 4)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
