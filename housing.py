#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:,:13].values
y = dataset.iloc[:,13].values

#Evaluating the co-relation matrix
corr_matrix = dataset.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

#Trying Attribute Combinations
housing["TAXRM"] = housing['TAX']/housing['RM']
housing.head()



#Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#Wvaluating accuracy using K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=regressor, X=X_train,y=y_train,cv=10)
accuracies.mean()

#Saving the model
from joblib import dump, load
dump(model, 'Dragon.joblib')
