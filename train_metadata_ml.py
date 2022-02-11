from curses import meta
import torch
import os
import numpy as np
import pandas as pd
import random

import xgboost as xgb
from sklearn.svm import SVR
from catboost import CatBoostRegressor

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

import seaborn as sns

import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster
    print(f'Setting all seeds to be {seed} to reproduce...')
seed_everything(1024)

path = '../TrainSet/'
image_path = path + 'TrainSet/'
metadata_path = path + 'trainClinData.xls'
metadata_df = pd.read_excel(metadata_path)

def label_sub_labels(row):
    if row['Prognosis'] == 'MILD' and row['Death'] == 0:
        return 0
    if row['Prognosis'] == 'MILD' and row['Death'] == 1:
        return 1
    if row['Prognosis'] == 'SEVERE' and row['Death'] == 0:
        return 2
    if row['Prognosis'] == 'SEVERE' and row['Death'] == 1:
        return 3

metadata_df['Sub_Label'] = metadata_df.apply(lambda row: label_sub_labels(row), axis=1)
    
mapping = {'SEVERE': 0, 'MILD': 1}
metadata_df['Prognosis'] = metadata_df['Prognosis'].apply(lambda class_id: mapping[class_id]) 

print(metadata_df['Sub_Label'].value_counts())
# for col_name in metadata_df.columns: 
#     print(col_name, metadata_df[col_name].count())
exit()

# related features as mentioned in the challenge
related_features = ['Age', 'Sex', 'Temp_C', 'Cough', 'DifficultyInBreathing', 'WBC', 'CRP', 'Fibrinogen', \
    'LDH', 'D_dimer', 'Ox_percentage', 'PaO2', 'SaO2', 'pH', 'CardiovascularDisease', 'RespiratoryFailure', 'Prognosis']
redundant_features = []

#dop unecessary columns
for col_name in metadata_df.columns: 
    if col_name not in related_features:
        redundant_features.append(col_name)
metadata_df = metadata_df.drop(redundant_features, axis=1)

#check nan
for col_name in metadata_df.columns: 
    print(col_name + ": ", metadata_df[col_name].isnull().values.any())
#fill nan
metadata_df['Age'] = metadata_df['Age'].fillna(0)
metadata_df['Temp_C'] = metadata_df['Temp_C'].fillna(0)
metadata_df['Cough'] = metadata_df['Cough'].fillna(0)
metadata_df['DifficultyInBreathing'] = metadata_df['DifficultyInBreathing'].fillna(0)
metadata_df['WBC'] = metadata_df['WBC'].fillna(0)
metadata_df['CRP'] = metadata_df['CRP'].fillna(0)
metadata_df['Fibrinogen'] = metadata_df['Fibrinogen'].fillna(0)
metadata_df['LDH'] = metadata_df['LDH'].fillna(0)
metadata_df['D_dimer'] = metadata_df['D_dimer'].fillna(0)
metadata_df['Ox_percentage'] = metadata_df['Ox_percentage'].fillna(0)
metadata_df['PaO2'] = metadata_df['PaO2'].fillna(0)
metadata_df['SaO2'] = metadata_df['SaO2'].fillna(0)
metadata_df['pH'] = metadata_df['pH'].fillna(0)
metadata_df['CardiovascularDisease'] = metadata_df['CardiovascularDisease'].fillna(0)
metadata_df['RespiratoryFailure'] = metadata_df['RespiratoryFailure'].fillna(0)

# 'Temp_C', 'Ox_percentage'
# metadata_df =  metadata_df[['LDH', 'PaO2', 'CRP', 'Age', 'WBC', 'pH', 'D_dimer', 'SaO2', 'Fibrinogen', \
#     'Sex', 'RespiratoryFailure', 'DifficultyInBreathing', 'CardiovascularDisease', 'Cough', 'Prognosis']]
#drop targets
X = metadata_df.drop('Prognosis', axis=1)
y = metadata_df.Prognosis


# taking holdout set for validating with stratified y
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

# 5 fold stratify for cv
cv = StratifiedKFold(5, shuffle=True, random_state=42)

xg = xgb.XGBClassifier(
    n_estimators=750,
    min_child_weight=0.81,
    learning_rate=0.025,
    max_depth=2,
    subsample=0.80,
    colsample_bytree=0.42,
    gamma=0.10,
    random_state=42,
    n_jobs=-1,
)
estimators = [xg]

# cross validation scheme

def model_check(X_train, y_train, estimators, cv):
    model_table = pd.DataFrame()

    row_index = 0
    for est in estimators:

        MLA_name = est.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name

        cv_results = cross_validate(est,
                                    X_train,
                                    y_train,
                                    cv=cv,
                                    scoring='balanced_accuracy',
                                    return_train_score=True,
                                    n_jobs=-1)

        model_table.loc[row_index,
                        'Train Balanced_Acc Mean'] = cv_results['train_score'].mean()
        model_table.loc[row_index,
                        'Test Balanced_Acc Mean'] = cv_results['test_score'].mean()
        model_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1

    model_table.sort_values(by=['Test Balanced_Acc Mean'],
                            ascending=False,
                            inplace=True)

    return model_table
# display cv results
# raw_models = model_check(X_train, y_train, estimators, cv)
# print((raw_models))

# fitting train data
xg.fit(X_train, y_train)

# predicting on holdout set
validation = xg.predict_proba(X_test)[:, 1]
# checking results on validation set
print("Roc AUC Score: ", roc_auc_score(y_test, validation))
validation = np.where(validation < 0.5, 1, 0)
print("Balanced Score: ", balanced_accuracy_score(y_test, validation))

exit()
# finding feature importances and creating new dataframe basen on them

feature_importance = xg.get_booster().get_score(importance_type='weight')

keys = list(feature_importance.keys())
values = list(feature_importance.values())

importance = pd.DataFrame(data=values, index=keys,
                          columns=['score']).sort_values(by='score',
                                                         ascending=False)
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(x=importance.score.iloc[:20],
            y=importance.index[:20],
            orient='h',
            palette='Reds_r')
ax.set_title('Feature Importances')
plt.show()

# Adversarial Validation
adv_train = X_train.copy()
adv_test = X_test.copy()

adv_train['dataset_label'] = 0
adv_test['dataset_label'] = 1

adv_master = pd.concat([adv_train, adv_test], axis=0)

adv_X = adv_master.drop('dataset_label', axis=1)
adv_y = adv_master['dataset_label']

adv_X_train, adv_X_test, adv_y_train, adv_y_test = train_test_split(adv_X,
                                                    adv_y,
                                                    test_size=0.4,
                                                    stratify=adv_y,
                                                    random_state=42)

xg_adv = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
)

# Fitting train data

xg_adv.fit(adv_X_train, adv_y_train)

# Predicting on holdout set
validation = xg_adv.predict_proba(adv_X_test)[:,1]

def plot_roc_feat(y_trues, y_preds, labels, est, x_max=1.0):
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    for i, y_pred in enumerate(y_preds):
        y_true = y_trues[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax[0].plot(fpr, tpr, label='%s; AUC=%.3f' % (labels[i], auc), marker='o', markersize=1)

    ax[0].legend()
    ax[0].grid()
    ax[0].plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')
    ax[0].set_title('ROC curve')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_xlim([-0.01, x_max])
    _ = ax[0].set_ylabel('True Positive Rate')
    
    
    feature_importance = est.get_booster().get_score(importance_type='weight')

    keys = list(feature_importance.keys())
    values = list(feature_importance.values())

    importance = pd.DataFrame(data=values, index=keys,
                          columns=['score']).sort_values(by='score',
                                                         ascending=False)
    
    sns.barplot(x=importance.score.iloc[:20],
            y=importance.index[:20],
            orient='h',
            palette='Reds_r', ax=ax[1])
    ax[1].set_title('Feature Importances')
    plt.show()

plot_roc_feat(
    [adv_y_test],
    [validation],
    ['Baseline'],
    xg_adv
)