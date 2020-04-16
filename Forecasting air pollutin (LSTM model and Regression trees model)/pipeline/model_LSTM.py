import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, LSTM, Multiply,Dropout,Dense
from keras.optimizers import Adam,SGD
from keras.models import load_model, Model
from keras.models import load_model
from keras.regularizers import l2

from pipeline.common import read_csv_series, make_directory_tree, get_datasets, describe_series

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_absolute_error

def MyModel(Tx,n,optimizer,dropout_rate,num_of_neurons_LSTM,num_of_neurons_Dense1,num_of_neurons_Dense2,num_of_neurons_Dense3,l2_constant):
    """
    LSTM Model built in keras API. It uses Adam optimizer, mean_squared_error loss and mean_absolut_error metric.
    
    Parameters:
    -----
    Tx: number of time steps for LSTM layer
    n: number of features in training data
    optimizer: optimizer used to find the minimum(example:SGD,Adam)
    dropout_rate: rate used in Dropout layer
    num_of_neurons_LSTM: number of neurons in LSTM layer
    num_of_neurons_Dense1: number of neurons in first Dense layer
    num_of_neurons_Dense2: number of neurons in second Dense layer
    num_of_neurons_Dense3: number of neurons in third Dense layer  
    l2_constant: constant for l2 regulatization
    
    Returns:
    -----
    keras.models.Model()
    """
    inputs = Input(shape=(Tx,n))
    X = LSTM(units=num_of_neurons_LSTM,return_state=False,kernel_regularizer=l2(l2_constant))(inputs)
    X = Dropout(rate=dropout_rate)(X)
    if num_of_neurons_Dense1 is not None:
        X = Dense(units=num_of_neurons_Dense1,activation='relu',kernel_regularizer=l2(l2_constant))(X)
        X = Dropout(rate=dropout_rate)(X)
    if num_of_neurons_Dense2 is not None:
        X = Dense(units=num_of_neurons_Dense2,activation='relu',kernel_regularizer=l2(l2_constant))(X)
        X = Dropout(rate=dropout_rate)(X)
    if num_of_neurons_Dense3 is not None:
        X = Dense(units=num_of_neurons_Dense3,activation='relu',kernel_regularizer=l2(l2_constant))(X)
        X = Dropout(rate=dropout_rate)(X)
    outputs = Dense(units=1,activation='linear')(X)
    
    model = Model(inputs=inputs,outputs=outputs)
    opt = optimizer
    model.compile(optimizer=opt,loss='mean_absolute_error',metrics=['mae'])
    return model
def dropNaN(X,Y,print_flag=""):
    """
    Drops rows from axis=0 if there are any NaN values in axes 1 and 2
    
    Parameters:
    -----
    X: numpy array that contains feature, with shape (m,Tx,n)
    Y: numpy array that contains labels, with shape (m,1)
    
    Returns:
    -----
    X and Y with dropped rows.
    """
    m = X.shape[0]
    mask = np.any(np.isnan(X), axis=(1,2)) 
    X = X[~mask]
    Y = Y[~mask]
    print("Dropped {} NaN values from {} set".format(m-X.shape[0], print_flag))
    return X,Y
def dropRows(df,Y,Tx,dropY):
    """
    This function is called by function shiftAndReshape().
    It drops last Tx*24 rows for each station in the concatenated DataFrames df and Y
    
    Parameters
    -----
    df: concatenated pandas DataFrame with features
    Y: concatenated pandas DataFrame with labels
    Tx: number of time steps for LSTM model
    dropY: flags whether to drop rows for Y or not (you should drop Y only once)
    """
    # drop last Tx*24 rows 
    df = df.iloc[:-Tx*24] 
    if dropY:
        Y = Y.iloc[:-Tx*24]
    i = 0
    m = len(df.index)-1
    while i < m:
        if (df.index[i].year - df.index[i+1].year > 2):
            df = pd.concat([df[:i+1-24*Tx], df[i+1:]])
            if dropY:
                Y = pd.concat([Y[:i+1-24*Tx], Y[i+1:]])
            i = i - 24*Tx
            m = m - 24*Tx
        i = i + 1
    return df,Y      
def shiftAndReshape(data,Tx):
    """
    it shifts the data Tx times (with stride -i*24, where i goes from 0 to Tx-1) in order to create timesteps for LSTM model.
    
    Parameters
    -----
    data: Pandas DataFrame taht contains both features and labels("PM10" column)
    Tx: Number of time steps for LSTM model
    
    Returns
    -----
    numpy array of features with shape (examples,Tx,features)
    numpy array of labels with shape (examples,1)
    """
    X, Y = data.drop(["PM10"], axis=1), data["PM10"]
    d = []
    dropY = True # make sure to only drop Y once
    for i in range(Tx):
        df = X.shift(-i*24)
        df,Y = dropRows(df,Y,Tx,dropY)
        dropY = False
        d.append(df.values)
    return np.stack(d,axis=1),Y.values
def scale(data,scaler,isTrainingData):
    """
    Scale the data using scaler (sklearn scaler: MinMax or Stanrad scaler)
    You should fit the scaler only on the train data, and provide isTrainingData=True for train data
    
    Parameters
    -----
    data: data to be scaled
    scaler: sklearn scaler object
    isTrainingData: boolean, that flags whether you fit and transform data or just transform
    
    Returns
    -----
    scaled data
    """
    m, Tx, n = data.shape
    data = np.reshape(data, newshape=(-1, n))
    if isTrainingData:
        data = scaler.fit_transform(data)
    else:
        data = scaler.transform(data)
    data = np.reshape(data, newshape=(m, Tx, n))
    return data

def plotPredictions(y, yhat, title, output_dir):
    """
    Plot the predictions against the actual values

    Parameters
    -----
    y: actual (real) values

    yhat: predicted values

    title: plot title
    """

    fig = plt.figure(figsize=(15, 6))
    plt.xlabel('Time')
    plt.ylabel('PM10')
    plt.plot(y, label="actual", figure=fig)
    plt.plot(yhat, label="predicted", figure=fig)
    plt.title(title)
    fig.legend()
    plt.savefig(os.path.join(output_dir, "{}.png".format(title)))
    plt.close(fig)
    return
def plotLoss(history):
    """
    Plots the train and test loss from the given keras history
    """
    plt.plot(history.history['loss'], 'b', label='training history')
    plt.plot(history.history['val_loss'],  'r',label='testing history')
    plt.title("Train and Test Loss for the LSTM")
    plt.legend()
    plt.show()
    return 
def predictAndEval(model,X,Y,print_flag=""):
    """
    Predict and evaluate the predictions with mean absolute error
    
    Parameters
    -----
    model: trained keras model
    X: feature vector
    Y: actual labels
    print_flag: string to be printed 
    
    Returns
    -----
    Yhat: predictions
    """
    Yhat = model.predict(X)
    mae = mean_absolute_error(Yhat,Y)
    print("\n{} MAE = {}".format(print_flag,mae))
    return Yhat
def saveModel(model,path):
    """
    Saves the keras model to the given path
    """
    model.save(os.path.join(path,'model.h5'))
    return 
def loadModel(path):
    """
    Loads the keras model from the given path
    
    Returns
    -----
    model: keras model
    """
    model = load_model(os.path.join(path,'model.h5'))
    return model
def run(input_dir,output_dir,optimizer,dropout_rate,num_of_neurons_LSTM,num_of_neurons_Dense1=None,num_of_neurons_Dense2=None,num_of_neurons_Dense3 = None,l2_constant=0,epochs=100,batch_size=128,predict_window=12,Tx=7):
    """
    LOAD the concatenated data set, DROP NaN values, SPLIT the data, RESHAPE the data, SCALE the data,
    CREATE the model, TRAIN the model, EVALUATE the model, PLOT predictions
    
    Parameters:
    -----
    input_dir: path to the concatenated .csv file (dataset)
    output_dir: output directory where the model and the plots will be saved
    optimizer: optimizer used to find the minimum(example:SGD,Adam)
    dropout_rate: rate used in Dropout layer
    num_of_neurons_LSTM: number of neurons in LSTM layer
    num_of_neurons_Dense1: number of neurons in first Dense layer (None if you do not want layer)
    num_of_neurons_Dense2: number of neurons in second Dense layer (None if you do not want layer)
    num_of_neurons_Dense3: number of neurons in third Dense layer (None if you do not want layer)
    l2_constant: constant for l2 regulatization
    epochs: number of epochs
    batch_size: size of a single batch
    predict_window: number of hours you forecast ahead (default is 12)
    Tx: number of previous days you take into consideration for prediction (default is 7)
    """
    # create output dirs for models plots and submission
    model_dir = os.path.join(output_dir, "model")
    plots_dir = os.path.join(output_dir, "plots")
    make_directory_tree(["model", "plots"], output_dir)
    
    print("-------------------------------------------")
    print("Loading dataset")
    # load dataset
    dataset = get_datasets(input_dir)
    dataset = dataset[0]
    df = read_csv_series(os.path.join(input_dir, dataset),keep_duplicates=True)
    print("Dataset loaded!")
    # shift PM10 for `predict_window` hours ahead
    df["PM10"] = df["PM10"].shift(-predict_window)
    
    print("-------------------------------------------")
    print("Reshaping dataset")
    # reshape the data so you can input it into LSTM model (examples,time_steps,features) 
    X,Y = shiftAndReshape(df,Tx)
    print("New shape is: {}".format(X.shape))
    print("Dataset reshaped!")
    
    print("-------------------------------------------")
    print("Dropping NaN values")
    # KERAS DOES NOT SUPPORT NaN VALUES, we need to impute them
    X,Y = dropNaN(X,Y,'complete dataset')
    
    print("-------------------------------------------")
    print("Schuffling dataset")
    indicies = np.random.permutation(X.shape[0])
    X,Y = X[indicies],Y[indicies]
    print("Dataset schuffled!")

    print("-------------------------------------------")
    print("Splitting dataset")                                                         
    # split dataset into train, test and evaluation by dates
    # additionally, leave the last 48 hours for final evaluation
    train_len = int(X.shape[0]* 0.65)
    test_len = int(X.shape[0] * 0.25)
    eval_len = X.shape[0] - train_len - test_len - 1
    X_train,X_test,X_eval=X[:train_len],X[train_len:train_len+test_len],X[train_len+test_len:train_len+test_len+eval_len]
    X_final_eval = X[-1:]
    Y_train,Y_test,Y_eval=Y[:train_len],Y[train_len:train_len+test_len],Y[train_len+test_len:train_len+test_len+eval_len]
    Y_final_eval = Y[-1:]
    print("Train set: {} examples".format(X_train.shape[0]))
    print("Test set: {} examples".format(X_test.shape[0]))
    print("Evaluation set: {} examples".format(X_eval.shape[0]))
    print("Final evaluation set: {} examples".format(X_final_eval.shape[0]))
    print("Dataset splitted!")

    print("-------------------------------------------")
    print("Scaling datasets")
    # scale the data
    scaler = MinMaxScaler()
    X_train = scale(X_train,scaler,True)
    X_test = scale(X_test,scaler,False)
    X_eval = scale(X_eval,scaler,False)
    X_final_eval = scale(X_final_eval,scaler,False)
    print("Datasets scaled!")
    
    # create the model
    model = MyModel(Tx,X_train.shape[2],optimizer,dropout_rate,num_of_neurons_LSTM,num_of_neurons_Dense1,num_of_neurons_Dense2,num_of_neurons_Dense3,l2_constant)
    print("\nMODEL SUMMARY:\n")
    print(model.summary())
    
    # fit the model
    print("-------------------------------------------")
    print("\nTRAINING HAS STARTED:\n")
    history = model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_test,Y_test),verbose=2,shuffle=True)
    # plot the loss 
    plotLoss(history)
    # evaluate
    Yhat_eval = predictAndEval(model,X_eval,Y_eval,'Evaluation')
    Yhat_final_eval = predictAndEval(model,X_final_eval,Y_final_eval,'Final evaluation')
    # plot predicted vs actual
    Yhat_train = model.predict(X_train)
    Yhat_test = model.predict(X_test)
    plotPredictions(Y_train,Yhat_train,"Train",plots_dir)
    plotPredictions(Y_test,Yhat_test,"Test",plots_dir)
    plotPredictions(Y_eval,Yhat_eval,"Eval",plots_dir)
    # save the model
    saveModel(model,model_dir)
    print("-------------------------------------------")
    print("Model is saved to {}".format(model_dir))
    
    
    
    
    
    
    
    

