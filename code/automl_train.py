import argparse
import pickle
import json
import logging
import os
import random

import pandas as pd
from sklearn import datasets

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.run import Run

from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

from azureml.telemetry import set_diagnostics_collection
import azureml.core
import numpy as np
from data_prep import *

import pickle

parser = argparse.ArgumentParser("automl_train")

parser.add_argument("--input_directory", type=str, help="input directory")
args = parser.parse_args()



print("input directory: %s" % args.input_directory)


# ws = Workspace.from_config()

# experiment_name =  'pred-maint-automl' # choose a name for experiment
project_folder = '.' # project folder


df_telemetry, df_errors, df_subset, df_fails, df_maint, df_machines = download_data()

with open(os.path.join(args.input_directory, "anoms.pkl"), "rb") as fp:
    obj = pickle.load(fp)
df_errors = obj['df_anoms']
rep_dir = {"volt":"error1", "rotate":"error2", "pressure":"error3", "vibration":"error4"}
df_errors = df_errors.replace({"errorID": rep_dir})
df_errors['errorID'] = df_errors['errorID'].apply(lambda x: int(x[-1]))


df_join = pd.merge(left=df_maint, right=df_fails.rename(columns={'failure':'comp'}), how = 'outer', indicator=True,
         on=['datetime', 'machineID', 'comp'], validate='one_to_one')
df_join.head()

df_left = df_telemetry.loc[:, ['datetime', 'machineID']] # we set this aside to this table to join all our results with

# this will make it easier to automatically create features with the right column names
#df_errors['errorID'] = df_errors['errorID'].apply(lambda x: int(x[-1]))
#df_maint['comp'] = df_maint['comp'].apply(lambda x: int(x[-1]))
#df_fails['failure'] = df_fails['failure'].apply(lambda x: int(x[-1]))

cols_to_average = df_telemetry.columns[-4:]

df_telemetry_rolling_3h = get_rolling_aggregates(df_telemetry, cols_to_average, 
                                                 suffixes = ['_ma_3', '_sd_3'], 
                                                 window = 3, on = 3, 
                                                 groupby = 'machineID', lagon = 'datetime')

df_telemetry_rolling_12h = get_rolling_aggregates(df_telemetry, cols_to_average, 
                                                  suffixes = ['_ma_12', '_sd_12'], 
                                                  window = 12, on = 3, 
                                                  groupby = 'machineID', lagon = 'datetime')

df_telemetry_rolling = pd.concat([df_telemetry_rolling_3h, df_telemetry_rolling_12h.drop(['machineID', 'datetime'], axis=1)], axis=1)

df_telemetry_feat_roll = df_left.merge(df_telemetry_rolling, how="inner", on=['machineID', 'datetime'], validate = "one_to_one")
df_telemetry_feat_roll.fillna(method='bfill', inplace=True)
df_telemetry_feat_roll.head()

del df_telemetry_rolling, df_telemetry_rolling_3h, df_telemetry_rolling_12h
df_errors_feat_roll = get_datetime_diffs(df_left, df_errors, catvar='errorID', prefix='e', window = 6, lagon = 'datetime', on = 3)
df_errors_feat_roll.tail()

df_errors_feat_roll.loc[df_errors_feat_roll['machineID'] == 2, :].head()

df_maint_feat_roll = get_datetime_diffs(df_left, df_maint, catvar='comp', prefix='m', 
                                        window = 6, lagon = 'datetime', on = 3, show_example=False)
df_maint_feat_roll.tail()

df_maint_feat_roll.loc[df_maint_feat_roll['machineID'] == 2, :].head()

df_fails_feat_roll = get_datetime_diffs(df_left, df_fails, catvar='failure', prefix='f', 
                                        window = 6, lagon = 'datetime', on = 3, show_example=False)
df_fails_feat_roll.tail()

assert(df_errors_feat_roll.shape[0] == df_fails_feat_roll.shape[0] == df_maint_feat_roll.shape[0] == df_telemetry_feat_roll.shape[0])
df_all = pd.concat([df_telemetry_feat_roll,
                    df_errors_feat_roll.drop(columns=['machineID', 'datetime']), 
                    df_maint_feat_roll.drop(columns=['machineID', 'datetime']), 
                    df_fails_feat_roll.drop(columns=['machineID', 'datetime'])], axis = 1, verify_integrity=True)

# df_all = pd.merge(left=df_telemetry_feat_roll, right=df_all, on = ['machineID', 'datetime'], validate='one_to_one')
df_all = pd.merge(left=df_all, right=df_machines, how="left", on='machineID', validate = 'many_to_one')
del df_join, df_left
del df_telemetry_feat_roll, df_errors_feat_roll, df_fails_feat_roll, df_maint_feat_roll

for i in range(1, 5): # iterate over the four components
    # find all the times a component failed for a given machine
    df_temp = df_all.loc[df_all['f_' + str(i)] == 1, ['machineID', 'datetime']]
    label = 'y_' + str(i) # name of target column (one per component)
    df_all[label] = 0
    for n in range(df_temp.shape[0]): # iterate over all the failure times
        machineID, datetime = df_temp.iloc[n, :]
        dt_end = datetime - pd.Timedelta('3 hours') # 3 hours prior to failure
        dt_start = datetime - pd.Timedelta('2 days') # n days prior to failure
        if n % 500 == 0: 
            print("a failure occured on machine {0} at {1}, so {2} is set to 1 between {4} and {3}".format(machineID, datetime, label, dt_end, dt_start))
        df_all.loc[(df_all['machineID'] == machineID) & 
                   (df_all['datetime'].between(dt_start, dt_end)), label] = 1

df_all.columns

X_drop = ['datetime', 'machineID', 'f_1', 'f_2', 'f_3', 'f_4', 'y_1', 'y_2', 'y_3', 'y_4', 'model']
Y_keep = ['y_1', 'y_2', 'y_3', 'y_4']

X_train = df_all.loc[df_all['datetime'] < '2015-10-01', ].drop(X_drop, axis=1)
y_train = df_all.loc[df_all['datetime'] < '2015-10-01', Y_keep]

X_test = df_all.loc[df_all['datetime'] > '2015-10-15', ].drop(X_drop, axis=1)
y_test = df_all.loc[df_all['datetime'] > '2015-10-15', Y_keep]

run = Run.get_context()
experiment = run.experiment
experiment_name = experiment.name

azureml.train.automl.constants.Metric.CLASSIFICATION_PRIMARY_SET

automl_config = AutoMLConfig(task='classification', 
                             preprocess=False,
                             name=experiment_name,
                             debug_log='automl_errors.log',
                             primary_metric='AUC_weighted',
                             max_time_sec=1200,
                             iterations=10,
                             n_cross_validations=2,
                             verbosity=logging.INFO,
                             X = X_train.values, # we convert from pandas to numpy arrays using .vaules
                             y = y_train.values[:, 0], # we convert from pandas to numpy arrays using .vaules
                             path=project_folder, )



run = experiment.submit(automl_config, show_output=True)
best_run, fitted_model = run.get_output()

best_accuracy = best_run.get_metrics()['accuracy']
print("Best run accuracy:", best_accuracy)
run.log('accuracy', best_accuracy)

# fitted_model = Pipeline([('scaler', StandardScaler()), ('classifier', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 10), random_state=1))])
# fitted_model.fit(X_train, y_train)

model_name = 'model.pkl'
with open(model_name, "wb") as file:
    joblib.dump(value = fitted_model, filename = model_name)

# Upload the model file explicitly into artifacts
run.upload_file(name = './outputs/'+ model_name, path_or_stream = model_name)
print('Uploaded the model {} to experiment {}'.format(model_name, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)

# Writing the run id to /aml_config/run_id.json
run_id = {}
run_id['run_id'] = run.id
run_id['experiment_name'] = run.experiment.name
with open('aml_config/run_id.json', 'w') as outfile:
  json.dump(run_id,outfile)
