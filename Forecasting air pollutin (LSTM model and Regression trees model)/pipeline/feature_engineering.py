import os

import pandas as pd

from pipeline.common import read_csv_series, get_datasets


def select_features(path, outpath):
    """
    Perform feature selection on the dataset by adding and selecting predefined features

    Parameters
    -----
    df : DataFrame

    Returns
    ----
    ret : DataFrame
        a new complete dataframe describing the final dataset that will be used for training
    """

    df = read_csv_series(path).drop(["apparentTemperature", "dewPoint", "windGust",
                                     "uvIndex", "ozone"], axis=1)

    # lag features
    df["PM10_before_24h"] = df["PM10"].shift(24)
    df["temperature_before_24h"] = df["temperature"].shift(24)

    # calendar/seasonal features
    df["month"] = df.index.month
    df["season"] = df.index.month  
    df["season"].loc[(df["season"] > 2)  & (df["season"] < 6)] = 1 # spring = 1
    df["season"].loc[(df["season"] > 5) & (df["season"] < 9)] = 2 # summer = 2
    df["season"].loc[(df["season"] > 8)  & (df["season"] < 11)] = 3 # autumn = 3
    df["season"].loc[(df["season"] < 3)  | (df["season"] == 12)] = 4 # winter = 4
    # stat features
    df["PM10_rolling_12h_mean"] = df["PM10"].rolling("12h").mean()
    # EWM for the past 7 days for each hour separately (temperature) 
    # capturing the trend how the temperature has changed for the past 7 days (for each hour)
    df["temperature_7days_EWM_by_hour"] = df.groupby(by = df.index.hour)["temperature"].transform(lambda x: x.ewm(span = 7).mean())
    # EWM for the past 7 days for each hour separately (PM10)
    # capturing the trend how the concetration of PM10 has changed for the past 7 days (for each hour)
    df["PM10_7days_WA"] = df.groupby(by = df.index.hour)["PM10"].transform(lambda x: x.ewm(span = 7).mean())  
    pd.DataFrame(df).to_csv(outpath)
    return df


def run(input_dir, output_dir):
    """
    Iterate over the combined datasets (containing pollution and weather data),
    select features and generate the final, usable datasets in output_dir.

    Parameters
    -----
    input_dir : path to directory containing combined and processed pollution and weather data

    output_dir : output directory for storing feature-selected data
    """
    # get .csv filenames from input directory
    datasets = get_datasets(input_dir)
    # iterate through filenames(different stations) and select features
    for dataset in datasets:
        print("Selecting features for {}".format(dataset))
        select_features(os.path.join(input_dir, dataset),
                        os.path.join(output_dir, dataset))
