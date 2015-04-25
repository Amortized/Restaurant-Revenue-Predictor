import sys;
from datetime import datetime
from sets import Set
from sklearn.svm import SVR
import numpy as np;
from sklearn import preprocessing, cross_validation;
import math;
import copy;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.grid_search import ParameterGrid;
from sklearn.preprocessing import OneHotEncoder;


def build_FeatureVal(myfile, mydict):
  with open(myfile, "r") as f:

    header = f.readline(); 
    for feature in range(0, len(header.strip().split(','))):
        mydict[feature] = Set();

    for line in f:
        data = line.strip().split(',');
        for i in range(0, len(data)):
            mydict[i].add(data[i]);

    return mydict;

def buildFeatures(myfile, train_feature_val, test_feature_val, data_X, data_Y, data_restaurant_ids, train_or_test="train"):
  current_date_obj = datetime.strptime('01/01/2015', "%m/%d/%Y");  

  #Prepare integer categorical representation for string variables as Onehot encoder can only handle that
  common_feature_integer_ranges = dict();
  for i in [2,3,4]:
    counter = 1;
    for val in train_feature_val[i]:
        if val in test_feature_val[i]:
          common_feature_integer_ranges[(i, val)] = counter;
          counter += 1;
    common_feature_integer_ranges[(i, "NULL")] = counter;

 

  with open(myfile, "r") as f:
    next(f);
    for line in f:
        data = line.strip().split(',');

        if train_or_test == "train":
          feature_range = len(data) -1;
          data_Y.append(float(data[feature_range]));
        else:
          feature_range = len(data);

        features = [];
        for i in range(0, feature_range):
            if i == 0:
                data_restaurant_ids.append(data[i]);
            elif i == 1:
                days_since_open = (current_date_obj - datetime.strptime(data[i], "%m/%d/%Y")).days;
                features.append(days_since_open)
            else:
                if data[i] in train_feature_val[i] and data[i] in test_feature_val[i]:
                    if i in [2,3,4]:
                       features.append(common_feature_integer_ranges[(i, data[i])]);
                    else:
                       features.append(float(data[i]));
                else:
                    if i in [2,3,4]:
                       features.append(common_feature_integer_ranges[(i, "NULL")]);
                    else:
                       features.append(float(data[i])); 

        data_X.append(features);


def calculate_RMSE(estimator, X, y):
    y_hat = estimator.predict(X);
    error = 0;
    for i in range(0, len(y_hat)):
        error += math.pow(y_hat[i] - y[i], 2);
    return math.sqrt(error/float(len(y_hat)));


def train_model(features, label):
    params          = {'max_features' : 'sqrt', 'n_estimators' : 50, 'n_jobs' : -1 }
    #params         = {'kernel' : 'linear' }

    #Preprocessing
    #scaled_features = preprocessing.scale(features);
    scaled_features  = features;

    # Set the parameters by cross-validation
    paramaters_grid   = {'max_depth': [3,4,5,6], 'min_samples_split' : [2,3,4,5,6,7],  'min_samples_leaf' : [2,3,4,5,6,7]};
    # Set the parameters by cross-validation
    #paramaters_grid    = {'C': [0.0000001, 0.001, 0.005, 0.008, 0.01, 0.02, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 10, 100, 0.004]};

    paramaters_search = list(ParameterGrid(paramaters_grid));


    best_rmse         = sys.float_info.max;
    best_params       = None;

    for ps in paramaters_search:
        for param in ps.keys():
            params[str(param)] = ps[param];


        total_rmse  = 0.0;
        count       = 0;

        lpo         = cross_validation.LeaveOneOut(len(scaled_features));
        for train_index, validation_index in lpo:

            X_train, X_validation = scaled_features[train_index], scaled_features[validation_index];
            Y_train, Y_validation = label[train_index], label[validation_index];

            #estimator               = SVR(**params)
            estimator                = RandomForestRegressor(**params)

            estimator.fit(X_train, Y_train);

            current_rmse          = calculate_RMSE(estimator, X_validation, Y_validation);

            total_rmse     += current_rmse;
            count          += 1;

        #Average across all samples
        avg_current_rmse   = total_rmse / float(count);
        #print("Avg Current RMSE " + str(avg_current_rmse));

        if avg_current_rmse < best_rmse:
            best_rmse   = avg_current_rmse;
            best_params = copy.deepcopy(params);

        print(params);
        print("RMSE : " + str(avg_current_rmse));



    print("Best RMSE : " + str(best_rmse));
    print("Best Params : " + str(best_params));


    #Train the model on the entire set
    estimator                = RandomForestRegressor(**params)
    #estimator               = SVR(**best_params)
    estimator.fit(scaled_features, label);
    print(estimator.feature_importances_)

    return  estimator;

def predict_and_save(model, test_features, test_restaurant_ids):
    predictions = model.predict(test_features);
    f  = open("./data/submission.csv", "w");
    f.write("Id,Prediction\n");
    for i in range(0, len(test_features)):
        f.write(str(test_restaurant_ids[i]) + ","  + str(predictions[i]) + "\n");
    f.close();

  


def compute(train, test):

  #Train data
  train_X              = [];
  train_restaurant_ids = [];
  test_X               = [];
  test_restaurant_ids  = [];
  train_Y              = [];

  #Common feature values in train/test
  train_feature_val    = {};
  test_feature_val     = {};

  build_FeatureVal(train, train_feature_val);
  build_FeatureVal(test, test_feature_val);
 
  buildFeatures(train, train_feature_val, test_feature_val, train_X, train_Y, train_restaurant_ids, "train");
  buildFeatures(test, train_feature_val, test_feature_val, test_X, None, test_restaurant_ids, "test");

  enc = OneHotEncoder(categorical_features=np.array([1,2,3,40]), sparse=False);

  enc.fit(test_X);

  train_X = enc.transform(train_X);
  test_X  = enc.transform(test_X);

  #Build a model
  estimator = train_model(train_X, np.array(train_Y));

  print("Writing Output");
  predict_and_save(estimator, test_X, test_restaurant_ids);

if __name__ == '__main__':
    compute("./data/train.csv","./data/test.csv");

    