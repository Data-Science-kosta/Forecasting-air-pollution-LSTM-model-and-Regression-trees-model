# -*- coding: utf-8 -*-
"""
Feature selection for LSTM model
"""
import os
import pandas as pd
from pipeline.common import read_csv_series, get_datasets

def selectFeatures(path):
    """
    Drop features that does not affect air pollution, and add some new features.
    
    Parameters
    -----
    path: full path to the .csv file
    
    Returns
    -----
    df: DataFrame with selected features
    """
    df = read_csv_series(path).drop(["apparentTemperature", "dewPoint", "windGust","uvIndex", "ozone",
                                    "precipAccumulation","precipIntensity","precipProbability"], axis=1)
    # calendar/seasonal features
    df["month"] = df.index.month
    df["season"] = df.index.month  
    df["season"].loc[(df["season"] > 2)  & (df["season"] < 6)] = 1 # spring = 1
    df["season"].loc[(df["season"] > 5) & (df["season"] < 9)] = 2 # summer = 2
    df["season"].loc[(df["season"] > 8)  & (df["season"] < 11)] = 3 # autumn = 3
    df["season"].loc[(df["season"] < 3)  | (df["season"] == 12)] = 4 # winter = 4
    return df
def dropUnmatchedColumns(sets,data):
    """
    Drops the columns that are not in common for all DataFrames, but first it handles simillar columns.
    
    Parameters
    -----
    sets: list of sets of columns for all Dataframes
    data: list of all DataFrames
    
    Returns
    -----
    data: list of all DataFrames without dropped columns
    """
    intersection = sets[0] #  set that contains in common columns
    for columns_set in sets:
        intersection = intersection & columns_set
    for i,df in enumerate(data):
        if i != 7:
            df['rain'].loc[df['sleet'] == 1] = 1
        if i != 3: 
            df['Light Rain'].loc[df['Drizzle'] == 1] = 1
        for column in df.columns[9:]:
            if column not in intersection:
                df.drop([column],axis=1,inplace=True)
    return data

def run(input_dir,output_dir):
    """
    Select features for all datasets and concatenate all datasets into 1 dataset.
    
    Parameters
    -----
    input_dir: directory which contains preprocessed .csv files 
    output_dir: directory in which you want to save the concatenated dataset
    
    Returns
    -----
    None
    """
    dataset_files = get_datasets(input_dir)
    data=[] # list that contains data frames for every dataset
    sets=[] # list that contains sets of columns from categorical variables
    for file in dataset_files:
        df = selectFeatures(os.path.join(input_dir,file))
        df_set = set(df.columns[9:]) # from 11 to end are columns that correspond to categorical variables
        sets.append(df_set)
        data.append(df)
    # drop columns that are not in common for all datasets, but preprocess them first
    data = dropUnmatchedColumns(sets,data)
    # concatenate data frames 
    data=pd.concat(data)
    # save to file
    pd.DataFrame(data).to_csv(os.path.join(output_dir,'concatenated.csv'))
    print("SAVED to {}".format(os.path.join(output_dir,'concatenated.csv')))
    return 

