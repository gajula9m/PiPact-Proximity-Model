import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib

# makes a dataframe from the data file
def dfMaker(filePath):
    return pd.read_csv(filePath)







def model_Maker(folderPath, filterAd = None):
    
    
    
    # uploading train and test files into pandas dataframes
    train_data = dfMaker(folderPath)
    
    # filtering scanned devices in case filters were not set when scanning data
    if filterAd != None:
        train_data = train_data[train_data.ADDRESS == filterAd]
    # test_data = test_data[test_data.ADDRESS == filterAd]
    
    # assigning data to x and y, reshaping x data to fit logistic regression model
    x = train_data.RSSI.values.reshape(-1,1)
    y = train_data.SAFE_DISTANCE
    
    # creaing and fitting logistic regression model
    lg = LogisticRegression()
    lg_model = lg.fit(x, y)

    return lg_model




def parse_args(args):
    """Input argument parser.
    Args:
        args (list): Input arguments as taken from sys.argv.
        
    Returns:
        Dictionary containing parsed input arguments. Keys are argument names.
    """
    # Parse command line arguments

    parser = argparse.ArgumentParser(
        description=("BLE beacon advertiser or scanner. Command line "
                     "arguments will override their corresponding value in "
                     "a configuration file if specified."))

    parser.add_argument('train_data', type = str, help="Train Data File")
    parser.add_argument('save_model_file', type = str,  help ="File in which you want your model saved")
    parser.add_argument('--filters',  type = str, help = "Any Bluetooth Filters for Data, if data not filtered")

    return vars(parser.parse_args(args))



def main(args):
    parsed_args = parse_args(args)

    model = model_Maker(parsed_args['train_data'], parsed_args['filters'])
    print('Model Created')

    model_filename = parsed_args['save_model_file']

    joblib.dump(model, model_filename)

    print('File created and models stored')


if __name__ == "__main__":
    """Script Execution."""
    main(sys.argv[1:])