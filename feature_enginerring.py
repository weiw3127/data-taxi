#sys
import warnings
warnings.filterwarnings('ignore')

# data processing 
import numpy as np 
import pandas as pd 
from datetime import timedelta
import datetime as dt

# extra feature creating 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

train = pd.read_csv('train0904.csv', engine='c')
test = pd.read_csv('test0904.csv', engine='c')

####################################
######### time data ################
####################################

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

train['pickup_date'] = train['pickup_datetime'].dt.date
test['pickup_date'] = test['pickup_datetime'].dt.date

train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
test['pickup_weekday'] = test['pickup_datetime'].dt.weekday

train['pickup_weekofyear'] = train['pickup_datetime'].dt.weekofyear
test['pickup_weekofyear'] = test['pickup_datetime'].dt.weekofyear

train['pickup_hour'] = train['pickup_datetime'].dt.hour
test['pickup_hour'] = test['pickup_datetime'].dt.hour

train['pickup_minute'] = train['pickup_datetime'].dt.minute
test['pickup_minute'] = test['pickup_datetime'].dt.minute

train['pickup_dt'] = (train['pickup_datetime'] - 
                      train['pickup_datetime'].min()).dt.total_seconds()
test['pickup_dt'] = (test['pickup_datetime'] - 
                     train['pickup_datetime'].min()).dt.total_seconds()

train['pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']
test['pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']

train['pickup_dayofyear'] = train['pickup_datetime'].dt.dayofyear
test['pickup_dayofyear'] = test['pickup_datetime'].dt.dayofyear

##################################
###### geogrpahic data ###########
##################################

# reference: http://www.movable-type.co.uk/scripts/latlong.html

#############
## bearing ##
#############

# Formula: θ = atan2(sin Δλ * cos φ2 , 
#                    cos φ1 * sin φ2 − sin φ1 * cos φ2 * cos Δλ)
# where φ1,λ1 is the start point, 
#       φ2,λ2 the end point (Δλ is the difference in longitude)

def bearing(lat1, lng1, lat2, lng2):
    del_lam = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(del_lam) * np.cos(lat2)
    x = (np.cos(lat1) * np.sin(lat2) -
         np.sin(lat1) * np.cos(lat2) * np.cos(del_lam))
    return np.degrees(np.arctan2(y, x))

train.loc[:, 'bearing'] = bearing(train['pickup_latitude'].values,
                                  train['pickup_longitude'].values,
                                  train['dropoff_latitude'].values,
                                  train['dropoff_longitude'].values)

test.loc[:, 'bearing'] = bearing(test['pickup_latitude'].values,
                                 test['pickup_longitude'].values,
                                 test['dropoff_latitude'].values,
                                 test['dropoff_longitude'].values)

#########################
## centure of the trip ##
#########################

train['center_lat'] = (train['pickup_latitude'].values +
                       train['dropoff_latitude'].values) / 2
train['center_long'] = (train['pickup_longitude'].values +
                        train['dropoff_longitude'].values) / 2
test['center_lat'] = (test['pickup_latitude'].values +
                      test['dropoff_latitude'].values) / 2
test['center_long'] = (test['pickup_longitude'].values +
                       test['dropoff_longitude'].values) / 2

#########
## PCA ##
#########

# to imporve the boosting 
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)

train['pick_pca_lat'] = pca.transform(train[['pickup_latitude',
                                             'pickup_longitude']])[:, 0]
train['pick_pca_long'] = pca.transform(train[['pickup_latitude',
                                              'pickup_longitude']])[:, 1]
train['drop_pca_lat'] = pca.transform(train[['dropoff_latitude',
                                             'dropoff_longitude']])[:, 0]
train['drop_pca_long'] = pca.transform(train[['dropoff_latitude',
                                              'dropoff_longitude']])[:, 1]
test['pick_pca_lat'] = pca.transform(test[['pickup_latitude',
                                           'pickup_longitude']])[:, 0]
test['pick_pca_long'] = pca.transform(test[['pickup_latitude',
                                            'pickup_longitude']])[:, 1]
test['drop_pca_lat'] = pca.transform(test[['dropoff_latitude',
                                           'dropoff_longitude']])[:, 0]
test['drop_pca_long'] = pca.transform(test[['dropoff_latitude', 
                                            'dropoff_longitude']])[:, 1]


###############
## Haversine ##
###############

# Formula: 
# a = sin²(Δφ/2) + cos φ1 * cos φ2 * sin²(Δλ/2)
# d = 2 * 6371 * atan2( √a, √(1−a) )

def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    lat = lat2 - lat1
    lng = lng2 - lng1
    a = (np.sin(lat * 0.5) ** 2 + 
         np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2)
    d = 2 * 6371 * np.arcsin(np.sqrt(a))
    return d

train['haversine'] = haversine(train['pickup_latitude'].values,
                               train['pickup_longitude'].values,
                               train['dropoff_latitude'].values,
                               train['dropoff_longitude'].values)

test['haversine'] = haversine(test['pickup_latitude'].values,
                              test['pickup_longitude'].values,
                              test['dropoff_latitude'].values,
                              test['dropoff_longitude'].values)


####################
## Taxicab Metric ##
####################

# Formula: |x1 - x2| + |y1 - y2|

def taxicab_dist(lat1, lng1, lat2, lng2):
    a = haversine(lat1, lng1, lat1, lng2)
    b = haversine(lat1, lng1, lat2, lng1)
    return a + b

train['taxicab_dist'] = taxicab_dist(train['pickup_latitude'].values,
                                     train['pickup_longitude'].values,
                                     train['dropoff_latitude'].values,
                                     train['dropoff_longitude'].values)

test['taxicab_dist'] = taxicab_dist(test['pickup_latitude'].values,
                                    test['pickup_longitude'].values,
                                    test['dropoff_latitude'].values,
                                    test['dropoff_longitude'].values)

################
## clustering ##
################

# clustering the data by their pickup/drop location
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100,
                         batch_size=10000).fit(coords[sample_ind])

train['pickup_cluster'] = kmeans.predict(train[['pickup_latitude',
                                                'pickup_longitude']])
train['dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude',
                                                 'dropoff_longitude']])
test['pickup_cluster'] = kmeans.predict(test[['pickup_latitude',
                                              'pickup_longitude']])
test['dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude',
                                               'dropoff_longitude']])

#######################################################
############## categorical to numerical ###############
#######################################################

########################
## store_and_fwd_flag ##
########################

train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

#################
### outlier #####
#################

ulimit = np.percentile(train.trip_duration.values, 99.5)
train_1 = train_1[train_1.trip_duration.values <= ulimit]
