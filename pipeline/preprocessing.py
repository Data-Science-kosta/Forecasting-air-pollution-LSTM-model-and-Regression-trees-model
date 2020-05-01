import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.common import read_csv_series, make_directory_tree


def preprocess_pollution_report(path, outpath):
    """
    Preprocess a pollution report

    Create a time series of only the PM10 column, no matter what other columns are present.
    Remove values measured over 1000.
    Interpolate missing data for single hours, remove larger periods of NaN values.

    Parameters
    -----
    path : path to input dataset

    outpath : path at which to save the processed dataset

    Returns
    -----
    ret : processed pollution series
    """

    print("Processing {}".format(path))

    df = read_csv_series(path)

    # convert series values to numeric type (np.float64 by default)
    # place NaN values in rows with invalid format
    df = pd.to_numeric(df["PM10"], errors="coerce")

    # create a new datetime index with hourly frequency
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    idx_name = df.index.name # ova komanda je izgleda suvisna
    # reindex our series
    df = df.reindex(idx) # places NaN in locations having no value in previous index
    df.index.name = idx_name

    # if data is missing for more than 1 hour or > 1000, remove those rows
    # also set negative measurements to np.nan
    # otherwise, interpolate
    df.loc[(df > 1000) | (df < 0)] = np.nan
    df = df.interpolate(method="linear", limit=1, limit_area="inside")
    df = df.dropna()

    pd.DataFrame(df).to_csv(outpath)

    return df


def preprocess_weather_report(path, outpath):
    """
    Preprocess a weather report

    Remove icon column.
    Set numeric values for precipAccumulation, cloudCover; add "no precip" category for precipType.
    Interpolate values for pressure column.
    One-hot encode categorical columns.

    Parameters
    -----
    path : path to input dataset

    outpath : path at which to save the processed dataset

    Returns
    -----
    ret : processed weather series
    """

    df = read_csv_series(path)

    df = df[1:].drop(["icon", "windBearing"], axis=1)
 
    df = df.fillna({"precipType": "no precip",
                    "precipAccumulation": 0, "cloudCover": 0, "ozone": 0, "UV": 0})
    df["pressure"].interpolate(inplace=True, limit=3) 
    df["windGust"].interpolate(inplace=True, limit=5) 
   # df = df.dropna(subset=["windGust","pressure"], axis = 0)
    
    # check for columns with discrete values, one-hot encode them
    categorical_column_names = list(
        df.select_dtypes(include=["object"]).columns)
    # cannot one-hot encode NaN values
    for column_name in categorical_column_names:
        one_hot = pd.get_dummies(df[column_name])
        df = df.join(one_hot)

    df = df.drop(categorical_column_names, axis=1)

    pd.DataFrame(df).to_csv(outpath)

    return df


def combine_reports(df_pollution, df_weather, outpath):
    """
    Produce a combined report from pollution and weather reports

    Parameters
    -----
    df_pollution : pollution series

    df_weather : weather series

    outpath : path at which to save the combined dataset
    """

    df = pd.concat([df_pollution, df_weather], axis=1)
    # drop columns with NaN for PM10
    df = df.dropna(subset=["PM10"])

    # save combined dataset
    pd.DataFrame(df).to_csv(outpath)


def run(input_dir, output_dir):
    """
    Iterate over the input datasets and process them, generating new datasets in output directory.

    Parameters
    -----
    input_dir : path to directory containing raw pollution and weather data

    output_dir : output directory for storing processed data
    """
    # make output directories for processed data
    make_directory_tree(["pollution", "weather", "combined"], output_dir)
    # sort the pollution_reports and weather_reports so we can iterate through them
    pollution_reports = sorted([f for f in os.listdir(
        os.path.join(input_dir, "pollution")) if os.path.isfile(os.path.join(input_dir, "pollution", f)) and f.endswith(".csv")])
    weather_reports = sorted([f for f in os.listdir(
        os.path.join(input_dir, "weather")) if os.path.isfile(os.path.join(input_dir, "weather", f)) and f.endswith(".csv")])
    # iterate through the reports for every station
    for pollution_report, weather_report in zip(pollution_reports, weather_reports):
        loc = pollution_report.split('.')[0].split('_')[-1]
        if loc != weather_report.split('.')[0].split('_')[-1]: # if the reports do not match
            raise ValueError("Pollution and weather report for different place! {} and {}".format(
                pollution_report, weather_report))
        # processing reports
        print("Processing reports for {}".format(loc))
        # processing pollution reports
        df_pollution = preprocess_pollution_report(os.path.join(input_dir, "pollution", pollution_report), os.path.join(
            output_dir, "pollution", pollution_report))
        # processing weather reports
        df_weather = preprocess_weather_report(os.path.join(input_dir, "weather", weather_report), os.path.join(
            output_dir, "weather", weather_report))
        # combine wather and pollution reports and save them
        combine_reports(df_pollution, df_weather, os.path.join(
            output_dir, "combined", loc + ".csv"))
