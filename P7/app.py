# 1. Library imports
import uvicorn
from fastapi import FastAPI, Response
from Model import Parameters, Predproba, FeatureImportance
from sklearn.neighbors import NearestNeighbors
import joblib
import json
import shap
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 2. Create app and model objects
app = FastAPI()
model_fname = 'model/final_model.pkl'
best_model = joblib.load(model_fname)
application_test = pd.read_pickle('data/application_test.pkl')
data_test = pd.read_pickle("data/data_test_ltd.pkl")
df_customer_info = pd.read_pickle("data/df_customer_info.pkl")
df_customer_loan = pd.read_pickle("data/df_customer_loan.pkl")
file_shap_values = 'data/shap_values_ltd.pkl'
with open(file_shap_values, 'rb') as shap_values:
            shap_values = pickle.load(shap_values) 

@app.get("/")
def read_home():
    return {"msg":"System is healthy"}

@app.get('/sk_ids')
def sk_ids():
    # Extract list of all the 'SK_ID_CURR' ids in the data_test dataframe
    test_set= data_test.reset_index(drop=True)
    test_set = test_set.set_index('SK_ID_CURR')
    sk_ids = pd.Series(list(test_set.index.sort_values())) 
    # Convert pd.Series to JSON
    sk_ids_json = json.loads(sk_ids.to_json())
    # Returning the processed data
    
    return {'data': sk_ids_json}


@app.get('/customer_data/')
def customer_data(SK_ID_CURR: int):
    # Get the personal data for the customer (pd.Series)
    test_set= data_test.reset_index(drop=True)
    test_set = test_set.set_index('SK_ID_CURR')
    X_customer = test_set.loc[SK_ID_CURR, :]
    # Convert customer's data to JSON
    X_customer_json = json.loads(X_customer.to_json())
    
    return {'data': X_customer_json}

@app.get('/origin_data/')
def origin_data(SK_ID_CURR: int):
    # Get the personal data for the customer
    df_origin_data = application_test.set_index('SK_ID_CURR')
    X_data_origin = df_origin_data.loc[[SK_ID_CURR]]
    # Convert the dataframe of customer's data to JSON
    result = X_data_origin.to_json(orient="records", lines=True)
    X_origin_cust = json.loads(result)
    # Return the cleaned data
    
    return X_origin_cust

@app.get('/customer_info/')
def customer_info(SK_ID_CURR: int):
    customer_info = df_customer_info.set_index('SK_ID_CURR')
    X_customer_info = customer_info.loc[[SK_ID_CURR]]
    # Convert customer's data to JSON
    result = X_customer_info.to_json(orient="records", lines=True)
    X_customer_info_json = json.loads(result)
    # Return the cleaned data
    
    return X_customer_info_json

@app.get('/customer_loan/')
def customer_loan(SK_ID_CURR: int):
    customer_loan = df_customer_loan.set_index('SK_ID_CURR')
    X_customer_loan = customer_loan.loc[[SK_ID_CURR]]
    # Convert the pd.Series (df row) of customer's data to JSON
    result = X_customer_loan.to_json(orient="records", lines=True)
    X_customer_loan_json = json.loads(result)
    # Return the cleaned data
    
    return X_customer_loan_json

@app.post('/scoring', response_model=Predproba)
def scoring_cust(features: Parameters):
        
    data = features.dict()
    data_in = np.array(list(data.values())).reshape(1, -1)     
        # Compute the score of the customer 
    prediction = best_model.predict(data_in)
    probability = best_model.predict_proba(data_in).max()

    return {'prediction': prediction[0],
            'probability': probability
           }


@app.get('/shap_data/', response_model=FeatureImportance)
def shap_data(SK_ID_CURR: int):
    
    explainer = shap.TreeExplainer(best_model)
    test = data_test.reset_index(drop=True)
    X_set = test.set_index('SK_ID_CURR')
    X_test_selected = X_set.loc[[SK_ID_CURR]]
    X_test_selected_array = X_test_selected.values.reshape(1, -1)
    shap_values_selected = explainer.shap_values(X_test_selected_array)
    
    output = {'0': shap_values_selected[0].tolist(),
              '1': shap_values_selected[1].tolist()}
    
    return output


#    JSON data and return the prediction with the confidence
@app.post('/predict', response_model=Predproba)
def predict(features: Parameters):
    
    data = features.dict()
    data_in = np.array(list(data.values())).reshape(1, -1)            
    
    prediction = best_model.predict(data_in)
    probability = best_model.predict_proba(data_in).max()
#    return probability

    return {'prediction': prediction[0],
            'probability': probability
           }

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)