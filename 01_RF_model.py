
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import (GridSearchCV, KFold,ShuffleSplit,
                                     cross_val_score,cross_validate)


#〈H,θ_6,h_a 〉 MODEL 1
#〈H,θ_100,h_a 〉 MODEL 2
#〈H,θ_6,θ_100 〉 MODEL 3
#〈θ_6,θ_100,h_a 〉 MODEL 4

input_file='RF_dataset.csv'
features=['H','F','L','S6','S100']
Ylabel=['DS100','DS100H','DS200','DS200H']

data=pd.read_csv(input_file,delimiter=';')

####################### MODEL 1 ##############################
matX=np.column_stack([data.Rainfall.to_numpy(),
                        data.Falda_prev.to_numpy(),
                        data.Storage_prev_6.to_numpy()])
                       
####################### MODEL 2 ##############################
# matX=np.column_stack([data.Rainfall.to_numpy(),
#                         data.Falda_prev.to_numpy(),
#                         data.Storage_prev_100.to_numpy()])

####################### MODEL 3 ##############################
# matX=np.column_stack([data.Rainfall.to_numpy(),
#                        data.Storage_prev_6.to_numpy(),
#                        data.Storage_prev_100.to_numpy()])
                       
####################### MODEL 4 ##############################
# matX=np.column_stack([data.Storage_prev_6.to_numpy(),
#                         data.Falda_prev.to_numpy(),
#                         data.Storage_prev_100.to_numpy()])

#############################################################
matY=data.DS_200H.to_numpy() 


####################### RANDOM FOREST##############################
Xtrain0, Xtest0, Ytrain0, Ytest0 = train_test_split(matX,matY,train_size=0.8,test_size=0.2,shuffle=True, random_state=42)

X=Xtrain0
y=Ytrain0

skf= KFold(n_splits=10, shuffle=True, random_state=42)

rfc = RandomForestRegressor(n_estimators=25,random_state=42, n_jobs=-1)


scores = cross_val_score(rfc,X, y, cv=skf, scoring='neg_root_mean_squared_error')

print("%0.3f RMSE with a standard deviation of %0.3f" % (scores.mean(), scores.std()))

cv_score = cross_validate(rfc,X, y, cv=skf, scoring='neg_root_mean_squared_error',return_estimator =True)


# improve by modify the defult parameters:
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []

trees_grid = [1,2,3,4,5,6,7,8,9, 10, 15, 20, 25, 30,50,75,100]

for ntrees in trees_grid:
    rfc = RandomForestRegressor(n_estimators=ntrees, random_state=42, n_jobs=-1)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        y_train_p  =rfc.predict(X_train)
        y_test_p =rfc.predict(X_test)
        MSE1=mean_squared_error(y_train_p, y_train)
        MSE2=mean_squared_error(y_test_p, y_test)
        temp_train_acc.append(MSE1**0.5)
        temp_test_acc.append(MSE2**0.5)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    
train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV Metric is {:.2f}% with {} trees".format(min(test_acc.mean(axis=1))*1, 
    trees_grid[np.argmax(test_acc.mean(axis=1))]))

plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv_test')
ax.legend(loc='best')
ax.set_ylabel("Metric")
ax.set_xlabel("N_estimators");

#################################### max depth #################

train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
max_depth_grid = [2,3, 5, 7, 9, 10,11, 12,13, 15, 17, 20, 22, 24]

for max_depth in max_depth_grid:
    rfc = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1, max_depth=max_depth)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        y_train_p  =rfc.predict(X_train)
        y_test_p =rfc.predict(X_test)
        MSE1=mean_squared_error(y_train_p, y_train)
        MSE2=mean_squared_error(y_test_p, y_test)
        temp_train_acc.append(MSE1**0.5)
        temp_test_acc.append(MSE2**0.5)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    
train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV Metric is {:.2f}% with {} max_depth".format(min(test_acc.mean(axis=1))*1, 
                                                        max_depth_grid[np.argmax(test_acc.mean(axis=1))]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(max_depth_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(max_depth_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv_test')
ax.legend(loc='best')
ax.set_ylabel("Metric")
ax.set_xlabel("Max_depth");

####################################### min sample leaf grid #################

train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
min_samples_leaf_grid = [1, 3,4, 5, 6,7, 9, 11, 13, 15, 17, 20, 22, 24]

for min_samples_leaf in min_samples_leaf_grid:
    rfc = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1, 
                                 min_samples_leaf=min_samples_leaf)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        y_train_p  =rfc.predict(X_train)
        y_test_p =rfc.predict(X_test)
        MSE1=mean_squared_error(y_train_p, y_train)
        MSE2=mean_squared_error(y_test_p, y_test)
        temp_train_acc.append(MSE1**0.5)
        temp_test_acc.append(MSE2**0.5)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    
train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV Metric is {:.2f}% with {} min_samples_leaf".format(min(test_acc.mean(axis=1))*1, 
                min_samples_leaf_grid[np.argmax(test_acc.mean(axis=1))]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(min_samples_leaf_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(min_samples_leaf_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv_test')
ax.legend(loc='best')
ax.set_ylabel("Metric")
ax.set_xlabel("Min_samples_leaf");

####################################### max feature #################

train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []

max_features_grid = [1,2,3]

for max_features in max_features_grid:
    rfc = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1, 
                                 max_features=max_features)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        y_train_p  =rfc.predict(X_train)
        y_test_p =rfc.predict(X_test)
        MSE1=mean_squared_error(y_train_p, y_train)
        MSE2=mean_squared_error(y_test_p, y_test)
        temp_train_acc.append(MSE1**0.5)
        temp_test_acc.append(MSE2**0.5)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    
train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV Metric is {:.2f}% with {} max_features".format(min(test_acc.mean(axis=1))*1, 
                                                        max_features_grid[np.argmax(test_acc.mean(axis=1))]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(max_features_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(max_features_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv_test')
ax.legend(loc='best')
ax.set_ylabel("Metric")
ax.set_xlabel("Max_features");

################################################################################

parameters = {'n_estimators':[5,10,20,30],
              'max_features': [1,2,3], 
              'min_samples_leaf': [15,20,25], 
              'max_depth': [3,4, 5, 6,7]}
rfc =  RandomForestRegressor(random_state=42, n_jobs=-1)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X, y)

gcv.best_params_, gcv.best_score_

################################################################################    

rfc = RandomForestRegressor(n_estimators=gcv.best_params_['n_estimators'],
                            max_features=gcv.best_params_['max_features'],
                            min_samples_leaf=gcv.best_params_['min_samples_leaf'], 
                            max_depth=gcv.best_params_['max_depth'],
                            random_state=42, n_jobs=-1)

cv_score_update = cross_validate(rfc,X, y, cv=skf, scoring='neg_root_mean_squared_error',return_estimator =True)

RMSE=[]
for i, (train_index, test_index) in enumerate(skf.split(X)):
    
    Ypred=cv_score_update['estimator'][i].predict(Xtest0)
    
    MSE=mean_squared_error(Ytest0, Ypred)
    RMSE_test=MSE**0.5
    RMSE.append(RMSE_test)
    
print("%0.4f RMSE with a standard deviation of %0.5f" % (np.mean(RMSE), np.std(RMSE)))
        


