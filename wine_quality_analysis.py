import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split,cross_val_score,KFold
from sklearn.feature_selection import SelectPercentile,f_regression
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

"""
Note: You can get the data from the separated file
"""
#import dataframe vom white wine table
wine = pd.read_table("winequality-white.csv",sep=";")


#first let see the outlier quality wines
box = wine.boxplot("quality")
#now lets see all other attributes by quality
box2 = wine.boxplot(by="quality", return_type = "axes")
plt.show()

#now get our x and y for linear regression. We have to split the dataframe
quality = wine["quality"]
del(wine["quality"])

#init regression
regression = LinearRegression()
#first lets split the data into training and text data
x_train, x_test, y_train, y_test = train_test_split(scale(wine),quality, test_size=0.3, random_state=6)

#train the classifier
regression.fit(x_train,y_train)

#get score
print(regression.score(x_test,y_test))

#lets see which features are most responsible for the quality
print([name+":"+str(round(coef,1)) for name, coef in zip(
wine.columns.values, regression.coef_,)])

#the score was bad, what about the mean squared error?
print("Mean Squared Error: %.2f" % mean_squared_error(y_true=y_test, y_pred=regression.predict(x_test)))

#the mean squared error was 0.56. lets try it again with cross validation
crossvalidation = KFold(n=wine.shape[0], n_folds=13, shuffle=True, random_state=1)
#the mean squared error was 0.56. Not really god, lets try it again with cross validation
for i in range(2,12):
    crossvalidation = KFold(n=wine.shape[0],n_folds=i,shuffle=True,random_state=1)
    scores = cross_val_score(regression, wine, quality, scoring="mean_squared_error", cv=crossvalidation, n_jobs=1)
    print("Folds: %i, mean squared error: %.2f std: %.2f" % (len(scores), np.mean(np.abs(scores)), np.std(scores)))



#the mean quared error is still the same. we need feature seletion  to see if we can get better results

#print all f_scores for each feature
f_selector = SelectPercentile(f_regression, percentile=25)
f_selector.fit(wine,quality)
for feature, score in zip(wine.columns.values, f_selector.scores_):
    print("F-Score: %3.2f\t for feature %s" %(score,feature))

"""
we can see that some features are not important for the regression
with a greedy search we can get the optimal number of features
"""
greedy = RFECV(estimator=regression, cv=13, scoring="mean_squared_error")
greedy.fit(wine,quality)
print("Optimal number of features: %d" % greedy.n_features_)

#however i wanna test logistic regression now because y data look like data to be classified. We might get better results
logistic = LogisticRegression()
ovr = OneVsRestClassifier(LogisticRegression()).fit(x_train,y_train)
ovo = OneVsOneClassifier(LogisticRegression()).fit(x_train,y_train)
#print accuracy
print("OnevsRest Score: %.3f"% ovr.score(x_test,y_test))
print("OneVsOne Score: %.3f"% ovo.score(x_test,y_test))
#print MSE
print("OneVsRest Mean Squared Error: %.2f" % mean_squared_error(y_true=y_test,y_pred=ovr.predict(x_test)))
print("OneVsOne Mean Squared Error: %.2f" % mean_squared_error(y_true=y_test,y_pred=ovo.predict(x_test)))

#an accuracy of 0.539 is still not optimal but much better than linear regression

#finally lets predict a sample wine
sample = [3,0.22,0.21,4.2,32,34,74,0.9733,3.19,0.42,7.7]
print("Sample: %.3f" %(regression.predict(sample)))

"""
the outcome is not really optimal
we should analyze the data with another classifier like nearest neighbor
we gonna start a grid search
"""
range = [1,2,3,4,5,6,7,8,9,10,11]
classifier = KNeighborsClassifier(n_neighbors=3, metric="minkowski",p=2, weights="uniform")
grid = {"n_neighbors": range, "weights":["uniform","distance"], "p": [1,2]}

search = GridSearchCV(estimator=classifier, param_grid=grid, scoring="accuracy", n_jobs=1, refit=True, cv=13)
search.fit(wine,quality)
print("Best parameter: %s" %search.best_params_)
print("CV- accuracy: %.3f" %search.best_score_)

#an accuracy of 0.49 is not optimal neither which means some of the features have nothing to do with wine quality. 



