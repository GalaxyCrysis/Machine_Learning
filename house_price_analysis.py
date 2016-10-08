from urllib.request import urlopen
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor




"""
house data for suburbs from Boston. My goal is to predict the value of the houses based on the data we have with Regression.
NOTE: I downloaded the data with urllib and added the column names to the text file
The data are not prepared yet, since there are more than 1 seperator. We need to fix that first
"""


with open("house_prices.txt","r") as file:
    data = file.read().split("\n")
for i in range(0,len(data)):
    line = data[i][1:]
    line = line.replace("   ",",")
    line = line.replace("  ",",")
    line = line.replace(" ",",")
    data[i] = line


"""
To create the dataframe we first have to create another file and write data onto that file
"""

if not os.path.exists("house_price_data.txt"):
    f = open("house_price_data.txt","w")
    f.write("")
    f.close()
with open("house_price_data.txt","a") as file:
    for line in data:
        file.write(line + "\n")


"""
The data are prepared now. Time to get the data with pandas and split it into data and target
"""
data = pd.read_table("house_price_data.txt",sep=",")

target = data["MEDV"]
del(data["MEDV"])
"""
We split the data into train and test scaled data
After that we first begin with Linear Regression
"""
xtrain,xtest,ytrain,ytest = train_test_split(scale(data),target,test_size=0.3,random_state=5)

lin_regression = LinearRegression()
lin_regression.fit(xtrain,ytrain)
print("Linear Regression Score: %.3f"% lin_regression.score(xtest,ytest))
print("Mean squared error: %.2f" % mean_squared_error(y_true=ytest, y_pred=lin_regression.predict(xtest)))
"""
we got 0.687 score and 29.74 as mean squared error which is not so great
Lets try optimizing the mean squared error with crossvalidation
"""
for i in range(2,15):
    crossvalidation = KFold(n=data.shape[0], n_folds=i, shuffle=True, random_state=1)
    score = cross_val_score(lin_regression,scale(data),target,scoring="mean_squared_error",cv=crossvalidation,
                            n_jobs=1)
    print("Folds: %i, mean squared error: %.2f std: %.2f" % (len(score), np.mean(np.abs(score)), np.std(score)))

"""
we could lower the mean squared error to 24.01, a much better MSE.
Lets see if we can get better results with L1 and L2
"""
ridge = Ridge(normalize=True)
search = GridSearchCV(estimator=ridge,
                      param_grid={"alpha": np.logspace(-5,2,8)},
                      scoring="mean_squared_error",n_jobs=1, refit=True, cv=10)
search.fit(scale(data),target)
print("Ridge best param: %s" % search.best_params_)
print("Ridge CV MSE of best param: %.2f" % abs(search.best_score_))

"""
30.99 is much more worse than the MSE of our linear regression, lets try lasso
"""
lasso = Lasso(normalize=True)
search = GridSearchCV(estimator=lasso,
                      param_grid={"alpha": np.logspace(-5, 2, 8)},
                      scoring="mean_squared_error", n_jobs=1, refit=True, cv=10)
search.fit(scale(data), target)
print("Lasso best param: %s" % search.best_params_)
print("Lasso CV MSE of best param: %.2f" % abs(search.best_score_))
"""
Lasso is with a MSE of 33,26 even worse
Lets try ElasticNet as our last regulator
"""
elastic = ElasticNet(normalize=True)
search = GridSearchCV(estimator=elastic,
                      param_grid={"alpha": np.logspace(-5, 2, 8), "l1_ratio": [0.25,0.5,0.75]},
                      scoring="mean_squared_error", n_jobs=1, refit=True, cv=10)
search.fit(scale(data), target)
print("ElasticNet best param: %s" % search.best_params_)
print("ElasticNet CV MSE of best param: %.2f" % abs(search.best_score_))
"""
With 30.8 our best regulator, but not good enough. Its time to change the classifier
We begin with support vector machine
"""

svr = SVR()
search = [
    {"kernel":["linear"], "C": np.logspace(-3,2,6),
     "epsilon": [0,0.01,0.1,0.5,1,2,4]},
    {"kernel": ["rbf"], "degree": [2,3],
     "C": np.logspace(-3,2,6),
     "epsilon": [0,0.01,0.1,0.5,1,2,4]}]
gridsearch = GridSearchCV(estimator=svr, param_grid=search, refit=True, scoring="r2",cv=10)
gridsearch.fit(xtrain,ytrain)
print("SVR best score: %s" % gridsearch.best_score_)
print("SVR test performance: %.2f" % gridsearch.score(xtest,ytest))
print("SVR MSE : %.3f" %mean_squared_error(y_true=ytest, y_pred=gridsearch.predict(xtest)))
print("SVR best params: %s"%gridsearch.best_params_)

"""
with 'epsilon': 1, 'degree': 2, 'kernel': 'rbf', 'C': 100.0
Our accuracy was 0.85 and MSE 13.919. Really good results! Much better than all other tests before
However i just want to test if we can get better results with Random Forest 
"""

random_forst = RandomForestRegressor(n_estimators=250, random_state=1)
for i in range(2,10):
    crossvalidation = KFold(n=data.shape[0],n_folds=i,shuffle=True, random_state=1)
    score = np.mean(cross_val_score(random_forst,data,target,
                                    scoring="mean_squared_error",cv=i,n_jobs=1))
    print("Folds : %i MSE: %.3f"%(i,abs(score)))

"""
we wont get better results than SVR. Lets try with optimized random forest
"""
searchgrid = {
    "max_features":[0.1,"sqrt", "log2","auto"],
    "min_samples_leaf": [1,2,10,30]}
search_algo = GridSearchCV(estimator=random_forst,
                           param_grid=searchgrid,scoring="mean_squared_error",n_jobs=1, cv = 9)
search_algo.fit(data,target)
print("Random Forest best parameter: %s"%search_algo.best_params_)
print("Random Forest best MSE: %.3f" % search_algo.best_score_)

"""
So in the end SVR was the best regressor. So we gonna use SVR to predict something
"""
svr = SVR(epsilon=1, degree=2,kernel="rbf",C=100.0)
svr.fit(data,target)
sample = [0.03731,0.01,8.070,0,0.3690,5.3610,79.90,4.9371,3,252.0,18.83,397.90,9.10]
print("Prediction: %.2f" % svr.predict(sample))
