import os
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

from pipeline.common import read_csv_series, make_directory_tree, get_datasets

import xgboost as xgb

def plot_predictions(y, yhat, title="Predictions vs Actual", output_dir=None):
    """
    Plot the predictions against the actual values

    Parameters
    -----
    y : actual (real) values

    yhat : predicted values

    title : plot title
    """

    fig = plt.figure(figsize=(15, 6))
    plt.xlabel('Time')
    plt.ylabel('PM10')
    plt.plot(y, label="actual", figure=fig)
    plt.plot(yhat, label="predicted", figure=fig)
    plt.title(title)
    fig.legend()

    if output_dir != None:
        plt.savefig(os.path.join(output_dir, "{}.png".format(title)))

    plt.close(fig)


def make_pipeline(model):
    """
    Make a pipieline with the regressor

    The pipeline contains 2 steps:
    1. Imputation - scikit-learn does not allow NaN values in the data, so they are imputed
    2. Normalization - performed using MinMaxScaler
    3. Regression - performed using the selected model

    Parameters
    -----
    model : regressor implementing the scikit-learn model API

    Returns
    -----
    ret : a scikit-learn pipeline with normalization and regression
    """

    steps = [
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("norm", MinMaxScaler()),
        ("reg", model)
    ]
    pipeline = Pipeline(steps=steps)

    return pipeline


def train_and_evaluate(name, model, train, test, evaluation, final_eval, output_dir):
    """
    Train and evaluate a model through the scikit-learn pipeline

    A pipeline is build using the supplied model as the final step in the pipeline (see make_pipeline)
    Then, the input data is transformed accordingly to be able to fit into the pipeline.

    The model is evaluated and its performance is visualized in multiple steps:
    1. Train on ~65% of the data, evaluate on the following ~25%
    2. Train on ~90% of the data, evaluate on the last ~10%
    3. Train on the complete data without the last X hours, score the model on the last X hours

    After each step, the predictions are plotted against the actual values and the plots
    are saved in `output_dir`/plots and can be analyzed later.

    The model is saved after the last step under `output_dir`/models with the filename `name`.joblib,
    if you wish to load them later and reevaluate/retrain them.

    Parameters
    -----
    name : the name of the model

    model : a scikit-learn model object

    train : train portion of the dataset

    test : test portion of the dataset

    evaluation : evaluation portion of the dataset

    final_eval : last X hours from the dataset

    output_dir : base directory for saving output files

    Returns
    -----
    ret : mean absolute error of the trained model on last X hours
    """

    print("---" * 5)
    print("Running pipeline for {}".format(name))

    plot_dir = os.path.join(output_dir, "plots")

    pipeline = make_pipeline(model)

    X_train, y_train = train.drop(
        ["PM10"], axis=1).values, train["PM10"].values
    X_test, y_test = test.drop(["PM10"], axis=1).values, test["PM10"].values
    X_eval, y_eval = evaluation.drop(
        ["PM10"], axis=1).values, evaluation["PM10"].values
    X_final, y_final = final_eval.drop(
        ["PM10"], axis=1), final_eval["PM10"].values

    # first round - fit on train, predict on test
    print("Fitting pipeline on train data")
    pipeline.fit(X_train, y_train)
    yhat = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, yhat)
    print("MAE: {}".format(mae))
    plot_predictions(
        y_test, yhat, title="{} - Predicted vs. Actual on Test".format(name), output_dir=plot_dir)

    # second round - fit on train + test, predict on evaluation
    X_train = np.concatenate([X_train, X_test])
    y_train = np.concatenate([y_train, y_test])
    print("Fitting pipeline on train + test data")
    pipeline.fit(X_train,y_train)
    yhat = pipeline.predict(X_eval)
    mae = mean_absolute_error(y_eval,yhat)
    print("MAE: {}".format(mae))
    plot_predictions(y_eval,yhat,title="{} - Predicted vs. Actual on Evaluation".format(name),output_dir=plot_dir)

    # final round - fit on last X hours, by which the actual score will be measured
    X_train = np.concatenate([X_train, X_eval])
    y_train = np.concatenate([y_train, y_eval])
    print("Fitting pipeline on all \"all available data\"")
    pipeline.fit(X_train, y_train)
    yhat = pipeline.predict(X_final)
    mae = mean_absolute_error(y_final, yhat)
    print("MAE: {}".format(mae))
    plot_predictions(
        y_final, yhat, title="{} - Predicted vs. Actual".format(name), output_dir=plot_dir)

    # save the model
    joblib.dump(model, os.path.join(
        output_dir, "models", "{}.joblib".format(name)))

    return yhat, mae


def run(input_dir, output_dir, team_name="OrganizersTeam", predict_window=12):
    """
    Train and evaluate models for each dataset under `input_dir`

    This script trains and evaluates 2 models:
    1. LinearRegression
    2. ExtraTreesRegressor

    This is a naive method for selecting models, it only chooses between
    two predefined models and does not do any hyperparameter tuning/optimization.

    Check out :obj:`sklearn.model_selection.GridSearchCV` and 
    :obj:`sklearn.model_selection.RandomizedSearchCV` for parameter optimization

    Parameters
    -----
    input_dir : directory containing datasets

    output_dir : directory for saving useful output files (models, plots, etc)

    predict_window : number of hours needed to predict, default=12

    Returns
    -----
    ret : a Pandas DataFrame containing the scores for each trained model
    """

    models_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots")
    sub_dir = os.path.join(output_dir, "submissions")
    submission_file_name_fmt = "{}_{}.csv"

    make_directory_tree(["models", "plots", "submissions"], output_dir)

    datasets = get_datasets(input_dir)

    print("Will train a total of {} models".format(len(datasets) * 3))

    # create a scores table to keep MAE for each location:model pair
    scores = pd.DataFrame(columns=["Location", "Model", "MAE"])

    for dataset in datasets:
        # load the dataset
        df = read_csv_series(os.path.join(input_dir, dataset))
        loc = dataset.split(".")[0]

        # shift PM10 for `predict_window` hours ahead
        df["PM10"] = df["PM10"].shift(-predict_window)

        # split dataset into train, test and evaluation by dates
        # additionally, leave the last 48 hours for final evaluation
        train_len = int(len(df) * 0.65) - (2 * predict_window)
        test_len = int(len(df) * 0.25) - (2 * predict_window)
        eval_len = len(df) - train_len - test_len - (2 * predict_window)
        train, test, evaluation = df[:train_len], df[train_len:train_len +
                                                     test_len], df[train_len+test_len:train_len+test_len+eval_len]
        final_eval = df[-(2 * predict_window):-predict_window].copy()

        # initialize models
        models = [
            ("Linear Regression", LinearRegression()),
            ("Extra Trees Regressor", ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25,
                            min_samples_leaf=35, random_state=0)),
            ("XGBoost Trees Regression", xgb.XGBRegressor(objective="reg:linear", random_state=0))
        ]

        mae_min = 1e10
        yhat_sub = []

        for model in models:
            # get predictions and MAE
            yhat, mae = train_and_evaluate("{} - {}".format(loc,model[0]),model[1],train,test,evaluation,final_eval, output_dir)

            # save the score (MAE) for the model
            scores = scores.append(
                {"Location": loc, "Model": model[0], "MAE": mae}, ignore_index=True)

            # save the better predictions to `yhat_sub`
            if mae < mae_min:
                mae_min = mae
                yhat_sub = yhat

        sub_df = pd.DataFrame(yhat_sub, columns=["PM10"])
        sub_df.to_csv(os.path.join(sub_dir, submission_file_name_fmt.format(team_name, loc)))

    scores.to_csv(os.path.join(output_dir, "scores.csv"))

    print("Done")
    print("Saved models can be found at {}".format(models_dir))
    print("Plots can be found at {}".format(plots_dir))
    print("Submissions can be found at {}".format(sub_dir))

    return scores
