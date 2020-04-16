import os
import errno

import pandas as pd


def read_csv_series(path, ts_column="time",keep_duplicates=False):
    """
    Read a time series from a CSV file.

    The CSV file must contain a column with either a UNIX timestamp or a datetime
    string with any format supported by Pandas. 

    Parameters
    -----
    path : path to CSV file

    ts_column : name of the column containing time data, "time" by default

    Returns
    -----
    ret : Pandas Series object with datetime as index
    """

    # read CSV
    df = pd.read_csv(path, parse_dates=[ts_column])
    # convert timestamps to datetime objects using panda's to_datetime
    df[ts_column] = pd.to_datetime(df[ts_column], unit="s")
    # set datetime as index (make time series)
    df.index = df[ts_column]
    # delete original time column
    del df[ts_column]
    
    # remove rows with duplicated time if there are any, keep first duplicate row
    if not keep_duplicates:
        df = df.loc[~df.index.duplicated(keep="first")]

    df.index.name = ts_column

    return df


def describe_series(df):
    """
    Show basic information about a Pandas Series or DataFrame

    Parameters
    -----
    df : a Pandas Series or DataFrame object
    """

    print("Head:")
    print(df.head())
    print("Stats:")
    print(df.describe())
    print("Count:")
    print(df.count())
    print("Columns: {}".format(df.columns))

    print("Start of time: {}".format(str(df.index[0])))
    print("End of time: {}".format(str(df.index[-1])))


def make_directory_tree(tree, output_dir):
    """
    Create the output directory tree structure specified by `tree` in `output_dir`

    Parameters
    -----
    tree : list of paths to create under `output_dir`

    output_dir : path to root of output directory tree
    """

    for d in tree:
        try:
            path = os.path.join(output_dir, d)
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                print("Path already exists: {}".format(d))
                print("Files may be overwritten")
                continue
            else:
                raise


def get_datasets(data_dir):
    """
    Get all .csv filenames from the specified directory

    Parameters
    -----
    data_dir : path to directory containing .csv files

    Returns
    -----
    ret : list containing dataset filenames
    """

    return [f for f in os.listdir(data_dir) if os.path.isfile(
        os.path.join(data_dir, f)) and f.endswith(".csv")]
