#!/usr/bin/python

#import modules
import pandas as pd
import tensorflow as tf
import autokeras as ak
import mlflow.keras

#mlflow logger
mlflow.keras.autolog()

#data preparation
dataset = "input.csv"
dataset = pd.read_csv(dataset, sep=",")
dataset_columnname = dataset.head()

val_split = int((len(dataset)) * 0.7)
data_train = dataset[:val_split]
validation_data = dataset[val_split:]

data_x = data_train[
    [
        "R",
        "G",
        "B",
        "BLU",
        "GRN",
        "NIR",
        "RGE",
        "RED",
    ]
].astype("float64")

data_x_val = validation_data[
    [
        "R",
        "G",
        "B",
        "BLU",
        "GRN",
        "NIR",
        "RGE",
        "RED",
    ]
].astype("float64")

# Data with train data and the unseen data from subsequent time steps.
data_x_test = dataset[
    [
        "R",
        "G",
        "B",
        "BLU",
        "GRN",
        "NIR",
        "RGE",
        "RED",
    ]
].astype("float64")

data_y = data_train["HT"].astype("float64")

data_y_val = validation_data["HT"].astype("float64")

print(data_x.shape)  # (6549, 12)
print(data_y.shape)  # (6549,)

#prediction step
predict_from = 1
predict_until = 10
lookback = 3
clf = ak.TimeseriesForecaster(
    lookback=lookback,
    predict_from=predict_from,
    predict_until=predict_until,
    max_trials=1,
    objective="val_loss",
)
# Train the TimeSeriesForecaster with train data
clf.fit(
    x=data_x,
    y=data_y,
    validation_data=(data_x_val, data_y_val),
    batch_size=32,
    epochs=10,
)
# Predict with the best model(includes original training data).
predictions = clf.predict(data_x_test)
print(predictions.shape)
# Evaluate the best model with testing data.
print(clf.evaluate(data_x_val, data_y_val))