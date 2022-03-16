
# ! usr/bin/env python 3
# coding: utf-8

import datetime
import sys
import time
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import os
import pickle
from collections import Counter

# Visualisation
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Data pré-processing
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
    PowerTransformer, RobustScaler
# Oversampling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
# UnderSampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Modelisation
# Pre-processing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer

# Models
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, \
                             GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

# Optimisation optuna
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration import LightGBMPruningCallback

# Metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, \
                            recall_score, precision_score, matthews_corrcoef, \
                            cohen_kappa_score, fbeta_score, make_scorer, \
                            average_precision_score, log_loss, confusion_matrix, \
                            classification_report
# Interprétation
import shap

# Warnings
import warnings
from warnings import simplefilter
simplefilter("ignore", category=Warning)
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML

# --------------------------------------------------------------------
# -- Function 1 
# --------------------------------------------------------------------

def train_models(model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    output = {
      'AUC': roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1]),
      'Accuracy': accuracy_score(y_valid, model.predict(X_valid)),
      'Precision': precision_score(y_valid, model.predict(X_valid)),
      'Recall': recall_score(y_valid, model.predict(X_valid)),
      'F1': f1_score(y_valid, model.predict(X_valid))
      }
          
    return output

# --------------------------------------------------------------------
# -- Function 2 
# --------------------------------------------------------------------

def metrics_display(model, X_train, X_valid, y_train, y_valid, title,
                           df_res):
   
    # Start of running time
    time_start = datetime.now()

    # Data training
    model.fit(X_train, y_train, verbose=0)

    # End of training time
    time_end = datetime.now() - time_start
    
    # Predictions on validation dataset
    y_pred = model.predict(X_valid)

    # Probabilities
    y_proba = model.predict_proba(X_valid)[:, 1]
    
    # Adapted metrics
    AUC = roc_auc_score(y_valid, y_proba)
    Recall = recall_score(y_valid, y_pred)
    Accuracy = accuracy_score(y_valid, y_pred)
    Precision = precision_score(y_valid, y_pred)
    # Fbeta
    f1_score = fbeta_score(y_valid, y_pred, beta=1)
        
     
    # Saving metrics in dataframe
    df_res = df_res.append(pd.DataFrame({
        'Model': [title],
        'AUC': [AUC],
        'Recall': [Recall],
        'Accuracy': [Accuracy],
        'Precision': [Precision],
        'F1': [f1_score],
        'Run time': [time_end],
    }), ignore_index=True)

    display_confusion_matrix(y_valid, y_pred, title)
    
    return df_res

# --------------------------------------------------------------------
# -- Function 2 
# --------------------------------------------------------------------

def metrics_display_threshold(model, threshold, X_train, X_valid, y_train, y_valid, title,
                           df_res_threshold):
   
    # Start of running time
    time_start = datetime.now()

    # Data training
    model.fit(X_train, y_train, verbose=0)

    # End of training time
    time_end = datetime.now() - time_start
    
    # Predictions on validation dataset
    y_pred = model.predict(X_valid)

    # Probabilities
    y_proba = model.predict_proba(X_valid)[:, 1]
    
    # Score > solvency threshold: 1 or 0
    y_pred = (y_proba > threshold)
    y_pred = np.multiply(y_pred, 1)
    
    # Adapted metrics
    AUC = roc_auc_score(y_valid, y_proba)
    Recall = recall_score(y_valid, y_pred)
    Accuracy = accuracy_score(y_valid, y_pred)
    Precision = precision_score(y_valid, y_pred)
    f1_score = fbeta_score(y_valid, y_pred, beta=1)
        
     
    # Saving metrics in dataframe
    df_res_threshold = df_res_threshold.append(pd.DataFrame({
        'Model': [title],
        'AUC': [AUC],
        'Recall': [Recall],
        'Accuracy': [Accuracy],
        'Precision': [Precision],
        'F1': [f1_score],
        'Run time': [time_end],
    }), ignore_index=True)

    display_confusion_matrix(y_valid, y_pred, title)
    
    return df_res_threshold

# --------------------------------------------------------------------
# -- Function 3 
# --------------------------------------------------------------------

def display_confusion_matrix(y_true, y_pred, title):

    '''This function returns a confusion matrix.'''
    fig = plt.figure(figsize=(12,6))
  
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ['True Negative\n(Actual & predicted\nnon-defaulter)',
                   'False Positive\n(Actual non-defaulter\nPredicted defaulter)',
                   'False Negative\n(Actual defaulter\nPredicted non-defaulter)',
                   'True Positive\n(Actual & Predicted\ndefaulter)']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, annot_kws={"size":20}, fmt="", cmap='Blues')
    plt.xlabel("Predicted class", weight='bold', size=14)
    plt.ylabel("Actual class", weight='bold', size=14)
    plt.title(f'Confusion Matrix: {title}', weight='bold', size=16)
    plt.show()
    
# --------------------------------------------------------------------
# -- Function 4 
# --------------------------------------------------------------------

def false_PN(model, y_true, X_true):
    '''Cost function analyzes prediction errors False Pos. and False Neg.'''
    cm = confusion_matrix(y_true, model.predict(X_true))
    FP = cm[0][1]/np.sum(cm)
    FN = cm[1][0]/np.sum(cm)
  
    print("False Pos: {0:.2%}".format(FP))
    print("False Neg: {0:.2%}".format(FN))
#    return FP, FN

# --------------------------------------------------------------------
# -- Function 5 
# --------------------------------------------------------------------

def FP(model, y_true, X_true):
    '''Cost function analyzes prediction errors False Positive.'''
    cm = confusion_matrix(y_true, model.predict(X_true))
    FP = cm[0][1]/np.sum(cm)

    return round(FP*100, 2)

# --------------------------------------------------------------------
# -- Function 6 
# --------------------------------------------------------------------

def FN(model, y_true, X_true):
    '''Cost function analyzes prediction errors False Positive.'''
    cm = confusion_matrix(y_true, model.predict(X_true))
    FN = cm[1][0]/np.sum(cm)

    return round(FN*100, 2)

# --------------------------------------------------------------------
# -- Function 7 
# --------------------------------------------------------------------

def TN(model, y_true, X_true):
    '''Cost function analyzes prediction errors True Negative.'''
    cm = confusion_matrix(y_true, model.predict(X_true))
    TN = cm[0][0]/np.sum(cm)

    return round(TN*100, 2)

# --------------------------------------------------------------------
# -- Function 8 
# --------------------------------------------------------------------

def TP(model, y_true, X_true):
    '''Cost function analyzes prediction errors True Positive.'''
    cm = confusion_matrix(y_true, model.predict(X_true))
    TP = cm[1][1]/np.sum(cm)

    return round(TP*100, 2)

# --------------------------------------------------------------------
# -- Function 8 
# --------------------------------------------------------------------

def NPV(model, y_true, X_true):
    '''Cost function analyzes prediction errors True Positive.'''
    cm = confusion_matrix(y_true, model.predict(X_true))
    NPV = cm[1][0]/(cm[1][0] + cm[0][0])

    return round(NPV*100, 2)

# --------------------------------------------------------------------
# -- Function 9 
# --------------------------------------------------------------------

def custom_score(y_true, y_pred, tn_value=1, fp_value=0, fn_value=-10, tp_value=0):
    '''
    Custom score penalizing False negatives.
    Parameters
    ----------
    y_true : True class (0 or 1).
    y_pred : Predicted class (0 or 1).
    tn_value : True Negative, loan is paid back, good for the bank (default value=1)
              
    fp_value : False positive, loan is refused while the customer is a non-defaulter
               Bank will lose the interests (Type I error)(default value=0),
               To penalize
    fn_value : False negative, loan is granted to a defaulter
               Bank will lose lots of money (Type II error)(default value=-10)
               To penalize,
    tp_value : True positive, loan is refused as the customer is a defaulter optionnel (default value=1),
               Neutral for the bank, money not lost but not gained either
    Returns
    -------
    custom score : Normalize gain (between 0 & 1), greater is better
    '''
    # Confusion matrix
    (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred).ravel()
    # Total gain
    total_gain = tn * tn_value + fp * fp_value + fn * fn_value + tp * tp_value
    # Max gain : All predictions are correct
    max_gain = (fp + tn) * tn_value + (fn + tp) * tp_value
    # Min Gain: Any loan is refused, bank does not make any money
    min_gain = (fp + tn) * fp_value + (fn + tp) * fn_value
    
    custom_score = (total_gain - min_gain) / (max_gain - min_gain)
    
    return custom_score

# --------------------------------------------------------------------
# -- Function 10
# --------------------------------------------------------------------

def custom_score1(y_true, y_pred, tn_value=1, fp_value=-5, fn_value=-20, tp_value=0):
    '''
    Custom score penalizing False negatives.
    Parameters
    ----------
    y_true : True class (0 or 1).
    y_pred : Predicted class (0 or 1).
    tn_value : True Negative, loan is paid back, good for the bank (default value=1)
              
    fp_value : False positive, loan is refused while the customer is a non-defaulter
               Bank will lose the interests (Type I error)(default value=0),
               To penalize
    fn_value : False negative, loan is granted to a defaulter
               Bank will lose lots of monet (Type II error)(default value=*10)
               To penalize,
    tp_value : True positive, loan is refused as the customer is a defaulter optionnel (default value=1),
               Neutral for the bank, money not lost but not gained either
    Returns
    -------
    custom score : Normalize gain (between 0 & 1), greater is better
    '''
    # Confusion matrix
    (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred).ravel()
    # Total gain
    total_gain = tn * tn_value + fp * fp_value + fn * fn_value + tp * tp_value
    # Max gain : All predictions are correct
    max_gain = (fp + tn) * tn_value + (fn + tp) * tp_value
    # Min Gain: Any loan is refused, bank does not make any money
    min_gain = (fp + tn) * fp_value + (fn + tp) * fn_value
    
    custom_score1 = (total_gain - min_gain) / (max_gain - min_gain)
    
    return custom_score1