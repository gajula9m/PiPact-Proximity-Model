import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib

# function that takes in csv files for specific cases and places data into dataframes
def dfMaker(folderPath):
    li = []

    for file in os.listdir(folderPath):
        filename = os.fsdecode(file)
        print(filename)
        filepath = folderPath + filename
        df = pd.read_csv(filepath, index_col = None, header =0)            
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)






def model_Maker(folderPath, title, filterAd):
    
    
    
    # uploading train and test files into pandas dataframes
    train_data = dfMaker(folderPath)
    
    # filtering scanned devices in case filters were not set when scanning data
    train_data = train_data[train_data.ADDRESS == filterAd]
    # test_data = test_data[test_data.ADDRESS == filterAd]
    
    # assigning data to x and y, reshaping x data to fit logistic regression model
    x = train_data.RSSI.values.reshape(-1,1)
    y = train_data.SAFE_DISTANCE
    # x_test = test_data.RSSI.values.reshape(-1,1)
    # y_test = test_data.SAFE
    
    # creaing and fitting logistic regression model
    lg = LogisticRegression()
    lg_model = lg.fit(x, y)
    
    # get an array for the True Positive Rate for each data point
    # lg_probs = lg_model.predict_proba(x_test)[:,1]
    
    # get an array for No Skill Model
    # ns_probs = [0 for _ in range(len(y_test))]
    
    # find false positive rate, true positive rate, and thresholds for each model
    # ns_fpr, ns_tpr, thresholds = metrics.roc_curve(y_test, ns_probs)
    # lr_fpr, lr_tpr, thresholds2 = metrics.roc_curve(y_test, lg_probs)
    
    # display curve
    # plt.figure(figsize = (8,8))
    # plt.plot(ns_fpr, ns_tpr, linestyle = '--', color = 'r', label = 'No Skill')
    # plt.plot(lr_fpr, lr_tpr, marker = '.', color = 'orange', label = 'Logistic')
    # plt.legend(loc = "lower right")
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(title)
    # plt.show()

    return lg_model



model = model_Maker('/Users/Mehul/mit/trainData/', 'ROC Curve for all Data', "DC:A6:32:2C:42:BC")
print('Model Created')

model_filename = './models.sav'

joblib.dump(model, model_filename)

print('File created and models stored')