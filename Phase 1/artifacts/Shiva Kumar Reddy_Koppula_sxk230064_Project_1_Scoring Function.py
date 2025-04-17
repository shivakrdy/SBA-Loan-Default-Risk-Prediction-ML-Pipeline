#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib
import category_encoders as ce

def project_1_scoring(input_df: pd.DataFrame):
    # Load the saved model and encoders
    logReg = pickle.load(open('./artifacts3/LogisticRegressionModel.pkl', 'rb'))
    one_hot_encoder = pickle.load(open('./artifacts3/one_hot_encoder.pkl', 'rb'))  
    woe_encoder =  pickle.load(open('./artifacts3/woe_encoder.pkl', 'rb'))
    glm_model = pickle.load(open('./artifacts3/glm.pkl', 'rb'))
    
    # Consider rows without missing values in target column mis_status
    input_df= input_df.query('MIS_Status != "Missing"')

    # Remove index column
    input_df.drop('index', axis=1, inplace=True)

    # Replace encode Na/Null values
    input_df.fillna(0, inplace=True)
    input_df.isnull().values.any()
        # Replace null values for string data type columns
    string_cols_names = input_df.select_dtypes(include='object').columns.tolist()

    for col in string_cols_names:
        input_df[col].fillna('Missing',inplace= True)

    # Replace null values for float data type columns
    float_cols_names= input_df.select_dtypes(include='float').columns.tolist()

    for col in float_cols_names:
        input_df[col]=input_df[col].fillna(input_df[col].mode()[0])

    input_df.isnull().values.any()
 
    
    # One-hot encoding
    input_df = one_hot_encoder.transform(input_df)
    
    # WOE encoding
    input_df = woe_encoder.transform(input_df)
    
    # Rename columns
    # Define a dictionary to map old column names to new ones
    column_names = {
    'City': 'City_woe',
    'State': 'State_woe',
    'Bank': 'Bank_woe',
    'BankState': 'Bankstate_woe',
    'RevLineCr': 'RevLinecr_woe',
    'Zip':'Zip_woe',
    'NAICS':'NAICS_woe',
    'NoEmp':'NoEmp_woe',
    'NewExist':'NewExist_woe',
    'CreateJob':'CreateJob_woe',
    'RetainedJob':'RetainedJob_woe',
    'FranchiseCode':'FranchiseCode_woe',
    'UrbanRural':'UrbanRural_woe',
    'DisbursementGross':'DisbursementGross_woe',
    'BalanceGross':'BalanceGross_woe',
    'GrAppv':'GrAppv_woe',
    'SBA_Appv':'SBA_Appv_woe'}
    input_df.rename(columns=column_names, inplace=True)
    
    # GLM transformation
    input_df_glm = glm_model.predict(input_df[['SBA_Appv_woe', 'UrbanRural_woe', 'NoEmp_woe', 'GrAppv_woe']])
    

    # Add GLM columns
    def add_glm_columns(input_df, glm_data):
        input_df["GLM1"] = glm_data
        features = ['SBA_Appv_woe', 'UrbanRural_woe', 'NoEmp_woe', 'GrAppv_woe']
        for i, feature in enumerate(features):
            input_df[f"GLM{i+2}"] = glm_data * input_df[feature]
    add_glm_columns(input_df, input_df_glm)
    
    # Make predictions using the loaded model
    y_pred_prob = logReg.predict_proba(input_df.drop('MIS_Status',axis=1))[:, 1]
    y_pred_class = (y_pred_prob >= 0.4).astype(int)

    # Create the output DataFrame
    output_df = pd.DataFrame({
        "record_index": input_df.index,
        "predicted_class": y_pred_class,
        "probability_0": 1 - y_pred_prob,
        "probability_1": y_pred_prob
    })

    return output_df


# In[2]:


# Load new data and remove the target column (if it's present)
import pandas as pd
import pickle
new_data = pd.read_csv('C:/Users/koppu/Downloads/Project 1/SBA_loans_project_1.csv')

# Get predictions using the scoring function
results_df = project_1_scoring(new_data)
print(results_df.head())


# In[ ]:




