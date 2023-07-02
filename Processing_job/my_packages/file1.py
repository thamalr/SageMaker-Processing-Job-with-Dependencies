
import pandas as pd

def change_format(df):
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
    
    return df

def missing_value(df):
    
    print("c_ount of missing values: (before treatment)", df.isnull().sum())
    
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    print("count of missing values: (before treatment)", df.isnull().sum())
    print("missing values successfully replaced")
    return df

def data_manipulation(df):
    df = df.drop(['customerID'], axis = 1)
    
    return df
