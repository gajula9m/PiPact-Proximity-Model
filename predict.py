import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib


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

    parser.add_argument('model_file', help = "Load Model File")
    parser.add_argument('test_file', help="Test Data File")

    return vars(parser.parse_args(args))


def main(args):

    parsed_args = parse_args(args)

    models_filename = parsed_args['model_file']
    model = joblib.load(models_filename)

    test_data = pd.read_csv(parsed_args['test_file'])
    x_test = test_data.RSSI.values.reshape(-1,1)
    y_test = test_data.SAFE_DISTANCE

    result = model.score(x_test, y_test)

    print(f"Model Accuracy is: {result}")

    lg_probs = model.predict_proba(x_test)[:,1]
        
    # get an array for No Skill Model
    ns_probs = [0 for _ in range(len(y_test))]

    # find false positive rate, true positive rate, and thresholds for each model
    ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, lg_probs)

    # display curve
    plt.figure(figsize = (8,8))
    plt.plot(ns_fpr, ns_tpr, linestyle = '--', color = 'r', label = 'No Skill')
    plt.plot(lr_fpr, lr_tpr, marker = '.', color = 'orange', label = 'Logistic')
    plt.legend(loc = "lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


if __name__ == "__main__":
    """Script execution."""
    main(sys.argv[1:])