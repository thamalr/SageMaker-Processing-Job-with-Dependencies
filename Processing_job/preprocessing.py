################################ import libraries #########################################
import boto3

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
import warnings
warnings.simplefilter(action='ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


############################# import custom libraries #####################################

import my_package.file1 as mp
        # OR
#from my_package.file1 import change_format
#from my_package.file1 import missing_value
#from my_package.file1 import data_manipulation


##############################3 install predefined modules ##################################
import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/input/code/my_package/requirements.txt",
])
import numpy as np
import pandas as pd
############################################################################################


def cat_encoder(df, variable_list):
    dummy = pd.get_dummies(df[variable_list], drop_first = True)
    df = pd.concat([df, dummy], axis=1)
    df.drop(df[cat_var], axis = 1, inplace = True)
    
    print("Encoded successfully")
    return df

def scaling(X):  
    min_max=MinMaxScaler()
    X=pd.DataFrame(min_max.fit_transform(X),columns=X.columns)
    
    return X



if __name__ == "__main__":

    input_data_path = os.path.join("/opt/ml/processing/input", "telco_cutomer_churn.csv")


    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    
    columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
           'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

    df.columns = columns

    cat_var = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'Churn']
    
################################## calling functions in file1 ####################################### 

    df = mp.data_manipulation(mp.missing_value(mp.change_format(df)))
    
#######################################################################################################
    df = cat_encoder(df, cat_var)

    X = df.iloc[:, 0:30]
    y = df.iloc[:, -1]
    X = scaling(X)
    
  
    
    print("Saving the outputs")
    X_output_path = os.path.join("/opt/ml/processing/output1", "X.csv")   
        
    print("Saving output to {}".format(X_output_path))
    pd.DataFrame(X).to_csv(X_output_path, header=False, index=False)
    
    y_output_path = os.path.join("/opt/ml/processing/output2", "y.csv")   
        
    print("Saving output to {}".format(y_output_path))
    pd.DataFrame(y).to_csv(y_output_path, header=False, index=False)

