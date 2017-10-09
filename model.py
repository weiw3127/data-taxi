import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_csv('train.csv', engine='c')
train_1 = train.copy()
test = pd.read_csv('test.csv', engine='c')
test_1 = test.copy()

do_not_use_for_training = ['id', 'log_trip_duration', 'trip_duration',
                           'dropoff_datetime', 'pickup_date',
                           'pickup_datetime', 'date']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]

print(feature_names)
print('We have %i features.' % len(feature_names))

#########################################
## the collection for first prediction ##
#########################################

T = test[feature_names].values
X = train[feature_names].values
y = np.log(train['trip_duration'].values + 1)
kf = KFold(n_splits=3)
kf_list = list(kf.split(X))

S_train = np.zeros((X.shape[0], 3))
S_test = np.zeros((T.shape[0], 3))

#########
## SVR ##
#########

print('SVR time......')
scaler = StandardScaler()
scaler.fit(X)
xtrain = scaler.transform(X)
xtest = scaler.transform(T)

S_test_i = np.zeros((T.shape[0], 3))
for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = xtrain[train_idx]
    y_train = y[train_idx]
    X_holdout = xtrain[test_idx]
    y_holdout = y[test_idx]

    svr = SVR(C=.001, gamma=1)
    svr.fit(X_train, y_train)

    y_pred = svr.predict(X_holdout)[:]
    S_train[test_idx, 0] = y_pred
    S_test_i[:, j] = svr.predict(xtest)[:]

S_test[:, 0] = S_test_i.mean(1)
print(pd.DataFrame(S_test).head())

###############################
## GradientBoostingRegressor ##
###############################

print('GradientBoostingRegressor ......')

S_test_i = np.zeros((T.shape[0], 3))
for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]

    rf = GradientBoostingRegressor()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_holdout)[:]
    S_train[test_idx, 1] = y_pred
    S_test_i[:, j] = rf.predict(T)[:]

S_test[:, 1] = S_test_i.mean(1)

print(pd.DataFrame(S_test).head())

############# 
## XGBoost ##
#############

print('and XGB......')

xgb_params = {'min_child_weight': 20,
              'eta': .05,
              'colsample_bytree': .3,
              'max_depth': 12,
              'subsample': .8,
              'lambda': 3,
              'booster': 'gbtree',
              'silent': 1,
              'eval_metric': 'rmse',
              'objective': 'reg:linear'}

S_test_i = np.zeros((T.shape[0], 3))
for j, (train_idx, test_idx) in enumerate(kf_list):
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_holdout = X[test_idx]
    y_holdout = y[test_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_holdout, label=y_holdout)
    dtest = xgb.DMatrix(T)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    model = xgb.train(xgb_params,
                      dtrain,
                      2000,
                      watchlist,
                      early_stopping_rounds=10,
                      maximize=False,
                      verbose_eval=10)

    y_pred = model.predict(dvalid)[:]
    S_train[test_idx, 2] = y_pred
    S_test_i[:, j] = model.predict(dtest)[:]
S_test[:, 2] = S_test_i.mean(1)

print(pd.DataFrame(S_test).head())

######################
## Model ensembling ##
######################

print('Ensembling with AdaBoost......')

kf = KFold(n_splits=5)

fx = train_set.values
fy = y

KFold(n_splits=5, random_state=300, shuffle=True)
for train_index, test_index in kf.split(fx):
    fx_train, fx_test = fx[train_index], fx[test_index]
    fy_train, fy_test = fy[train_index], fy[test_index]

model = AdaBoostRegressor(n_estimators=50, learning_rate=.01)
model.fit(fx_train, fy_train)
predictions = model.predict(fx_test)
score = mean_absolute_error(fy_test, predictions)
print("\nScore {0}".format(score))

print('Final prediction...... ')
ptes = model.predict(test_set)
test['trip_duration'] = np.exp(ptes) - 1
test[['id', 'trip_duration']].to_csv('sub0831.csv', index=False)
print('Result in sub0831.csv ')
