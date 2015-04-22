import sys;
from datetime import datetime
from sets import Set
from sklearn.svm import SVR
import numpy as np;
from sklearn import preprocessing, cross_validation;
import math;
import copy;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.feature_selection import SelectKBest;
from sklearn.feature_selection import f_regression;

date_format = "%m/%d/%Y";
imputations = [];

def createListRepresentation(total, index):
    tt = [];
    for i in range(0, total):
        if i == index:
            tt.append(1);
        else:
            tt.append(0);
    return tt;




def computeFeatures(myfile, cities, ignore, train_or_test="train"):
    '''
      Generates a dict of labels
    '''
    features    = [];
    labels      = [];

    restaurant_ids = [];

    current_date_obj = datetime.strptime('01/01/2015', date_format);



    with open(myfile, "r") as f:
        next(f);
        for line in f:
            data = line.strip().split(',');

            temp = [];

            feature_range = None;
            if train_or_test == "train":
                feature_range = len(data) - 2;
            else:
                feature_range = len(data) - 1;

            for i in range(0, feature_range):
                if i == 0:
                    restaurant_ids.append(data[i]);
                elif i == 1:
                    days_since_open = (current_date_obj - datetime.strptime(data[i], date_format)).days;
                    temp.append(days_since_open);
                elif i == 2:
                    if ignore == 0:
                      #Don't add to features yet
                      cities.add(data[i]);
                    elif ignore == 1:
                      #Compute features
                      if data[i] in cities:
                        #City in both train and test
                        #Get the index
                        #pass;
                        temp.extend(createListRepresentation(len(cities), cities.index(data[i])));
                      else:
                        #pass;
                        temp.extend(createListRepresentation(len(cities), 1+ len(cities)));
                elif i == 3:
                    if data[i] == "Big Cities":
                        temp.extend([1,0]);
                    elif data[i] == "Other":
                        temp.extend([0,1]);
                elif i == 4:
                    if data[i] == "FC":
                        temp.extend([1,0,0,0]);
                    elif data[i] == "IL":
                        temp.extend([0,1,0,0]);
                    elif data[i] == "DT":
                        temp.extend([0,0,1,0]);
                    elif data[i] == "MB":
                        temp.extend([0,0,0,1]);
                else:
                    temp.append(float(data[i]));

            features.append(temp);

            if train_or_test == "train":
               labels.append(float(data[len(data)-1]));


    
    features = np.array(features);

    '''
    if train_or_test == "train":
      #Impute Missing values for P1 to P37.
      for i in range(35, len(features[0])):
        imputations.append(np.mean(features[:, i]));


    for i in range(0, len(features)):
      for k in range(35, len(features[i])):
         if features[i][k] == 0:
            features[i][k] = imputations[k-35];
    '''

    if train_or_test == "train":
      return features, np.array(labels), restaurant_ids;
    else:
      return features, restaurant_ids;


def calculate_RMSE(estimator, X, y):
    f = open("data.txt", "a")
    y_hat = estimator.predict(X);
    error = 0;
    for i in range(0, len(y_hat)):
        f.write(str(y_hat[i]) + "," + str(y[i]) + "\n");
        error += math.pow(y_hat[i] - y[i], 2);
    return math.sqrt(error/float(len(y_hat)));
    f.close();


def train_model(features, label):
    params          = {'max_features' : 'sqrt', 'n_estimators' : 30, 'n_jobs' : -1, 'min_samples_leaf' : 3 }
    #params         = {'kernel' : 'linear' }


    #Preprocessing
    #scaled_features = preprocessing.scale(features);
    scaled_features  = features;

    # Set the parameters by cross-validation
    paramaters_search = {'max_depth': [2,3,4,5,6]};

    # Set the parameters by cross-validation
    #paramaters_search = {'C': [0.0000001, 0.001, 0.005, 0.008, 0.01, 0.02, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 10, 100, 0.004]};

    best_rmse         = sys.float_info.max;
    best_params       = None;

    for ps in paramaters_search.keys():
        for param in paramaters_search[ps]:
            params[str(ps)] = param;

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



    print("Best RMSE : " + str(best_rmse));
    print("Best Params : " + str(best_params));


    #Train the model on the entire set
    estimator                = RandomForestRegressor(**params)
    #estimator               = SVR(**best_params)
    estimator.fit(scaled_features, label);

    return  estimator;

def predict_and_save(model, test_features, test_restaurant_ids):
    predictions = model.predict(test_features);
    f  = open("./data/submission.csv", "w");
    f.write("Id,Prediction\n");
    for i in range(0, len(test_features)):
        f.write(str(test_restaurant_ids[i]) + ","  + str(predictions[i]) + "\n");
    f.close();

if __name__ == '__main__':
    train_cities = Set();
    test_cities  = Set();

    print("Reading Training data");
    computeFeatures("./data/train.csv", train_cities, 0, "train");
    print("Reading Test data");
    computeFeatures("./data/test.csv", test_cities, 0, "test");


    cities_in_train_test = list(test_cities.intersection(train_cities));

    print("Generating Features Training data");
    train_features, train_labels, train_restaurant_ids = computeFeatures("./data/train.csv", cities_in_train_test, 1, "train");
    print("Generating Features Test data");
    test_features, test_restaurant_ids                 = computeFeatures("./data/test.csv", cities_in_train_test, 1, "test");

    #train_features = SelectKBest(f_regression, k = 30).fit(train_features, train_labels);

    
    model                                              = train_model(train_features, train_labels);

    print("Writing Output");
    predict_and_save(model, test_features, test_restaurant_ids);