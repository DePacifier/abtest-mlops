# Importing Libraries
from scripts.ML_modelling_utils import *
import sys
import os
import numpy as np
import pandas as pd
import mlflow
import os

# Jupyter Notebook Settings
import matplotlib.pyplot as plt
# plt.style.use('ggplot')


# importing local scripts
# Adding scripts path
sys.path.append(os.path.abspath(os.path.join('..')))
# importing data_manipulator script

# Importing and Fixing Data

# Importing the collected Data
path = 'data/AdSmartABdata.csv'
repo = 'https://github.com/DePacifier/abtest-mlops'
all_dfs = import_all_data_using_tagslist(path=path, repo=repo, tags=[
                                         'chrome-mobile', 'chrome-mobile-view', 'facebook', 'samsung-internet', 'platform-6'])


# Spliting the date column to year, month and day columns and removing it
all_dfs_date_fixed = split_date_to_numbers(all_dfs, 'date')

# drop columns from each data, the grouping column b/c its a univalue column
# We have 5 dataframes of which 4 are grouped based on browser and 1 based on platform
for i in all_dfs_date_fixed:
    if(i != "platform-6"):
        all_dfs_date_fixed[i].drop('browser', axis=1, inplace=True)
    else:
        all_dfs_date_fixed[i].drop('platform_os', axis=1, inplace=True)


data_type_fixed_dfs = change_columns_to_numbers(
    all_dfs_date_fixed, ['experiment', 'device_make', 'browser'])
data_type_fixed_dfs['platform-6'].sample(5)

# #Get all train, validate and test sets
chrome_mobile_dict = get_train_validate_test_sets(
    data_type_fixed_dfs['chrome-mobile'], predicted_column='response', remove_columns=['auction_id'])
chrome_mobile_view_dict = get_train_validate_test_sets(
    data_type_fixed_dfs['chrome-mobile-view'], predicted_column='response', remove_columns=['auction_id'])
facebook_dict = get_train_validate_test_sets(
    data_type_fixed_dfs['facebook'], predicted_column='response', remove_columns=['auction_id'])
samsung_internet_dict = get_train_validate_test_sets(
    data_type_fixed_dfs['samsung-internet'], predicted_column='response', remove_columns=['auction_id'])
platform_6_dict = get_train_validate_test_sets(
    data_type_fixed_dfs['platform-6'], predicted_column='response', remove_columns=['auction_id'])

# # Training
# > Training only done for 4 or the 5 data(samsung-internet is omitted due to low data count)


# Starting Logging
mlflow.sklearn.autolog(log_input_examples=True,
                       silent=True, max_tuning_runs=10)

# import warnings
# warnings.filterwarnings('ignore')

# Train All Models For chrome mobile
with mlflow.start_run(experiment_id=1, run_name="Chrome Mobile"):
    # Logistic Regression Model
    chrome_mobile_lr_model = train_logistic_model(
        chrome_mobile_dict['train_x'], chrome_mobile_dict['train_y'], chrome_mobile_dict['val_x'], chrome_mobile_dict['val_y'])

    # Decision Trees
    chrome_mobile_tree_model = train_decision_tree(
        chrome_mobile_dict['train_x'], chrome_mobile_dict['train_y'], chrome_mobile_dict['val_x'], chrome_mobile_dict['val_y'])

    # XGB Boost
    chrome_mobile_xgbc_model = train_xgb_classifier(
        chrome_mobile_dict['train_x'], chrome_mobile_dict['train_y'], chrome_mobile_dict['val_x'], chrome_mobile_dict['val_y'])

# Train All Models For chrome mobile view
with mlflow.start_run(experiment_id=2, run_name="Chrome Mobile View"):
    # Logistic Regression Model
    chrome_mobile_view_lr_model = train_logistic_model(
        chrome_mobile_view_dict['train_x'], chrome_mobile_view_dict['train_y'], chrome_mobile_view_dict['val_x'], chrome_mobile_view_dict['val_y'])

    # Decision Trees
    chrome_mobile_view_tree_model = train_decision_tree(
        chrome_mobile_view_dict['train_x'], chrome_mobile_view_dict['train_y'], chrome_mobile_view_dict['val_x'], chrome_mobile_view_dict['val_y'])

    # XGB Boost
    chrome_mobile_view_xgbc_model = train_xgb_classifier(
        chrome_mobile_view_dict['train_x'], chrome_mobile_view_dict['train_y'], chrome_mobile_view_dict['val_x'], chrome_mobile_view_dict['val_y'])


# Train All Models For Facebook
with mlflow.start_run(experiment_id=3, run_name="Facebook"):
    # Logistic Regression Model
    facebook_lr_model = train_logistic_model(
        facebook_dict['train_x'], facebook_dict['train_y'], facebook_dict['val_x'], facebook_dict['val_y'])

    # Decision Trees
    facebook_tree_model = train_decision_tree(
        facebook_dict['train_x'], facebook_dict['train_y'], facebook_dict['val_x'], facebook_dict['val_y'])

    # XGB Boost
    facebook_xgbc_model = train_xgb_classifier(
        facebook_dict['train_x'], facebook_dict['train_y'], facebook_dict['val_x'], facebook_dict['val_y'])


# Train All Models For Platform 6
with mlflow.start_run(experiment_id=4, run_name="Platform 6"):
    # Logistic Regression Model
    platform_6_lr_model = train_logistic_model(
        platform_6_dict['train_x'], platform_6_dict['train_y'], platform_6_dict['val_x'], platform_6_dict['val_y'])

    # Decision Trees
    platform_6_tree_model = train_decision_tree(
        platform_6_dict['train_x'], platform_6_dict['train_y'], platform_6_dict['val_x'], platform_6_dict['val_y'])

    # XGB Boost
    platform_6_xgbc_model = train_xgb_classifier(
        platform_6_dict['train_x'], platform_6_dict['train_y'], platform_6_dict['val_x'], platform_6_dict['val_y'])


### Results ############

# CHROME MOBILE
# Best Models


# CHROME MOBILE VIEW

# FACEBOOK

# PLATFORM 6

# chrome_mobile_tree_model.best_estimator_.feature_importances_

# chrome_mobile_dict['train_x']

# chrome_mobile_xgbc_model.best_estimator_

# ## Too Much Compution Time Required to train xgb classifier with these parameters


# platform_6_xgbc_model.best_estimator_.feature_importances_

# platform_6_dict['train_x']
