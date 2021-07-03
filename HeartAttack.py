import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble

#Importing data
data = pd.read_csv (r'heart.csv')

inputCols = set(data.columns)
inputCols.remove('output')

#Input data
inputs = data[inputCols]
inp = inputs.to_numpy()
#Output data
outcome = pd.DataFrame(data,columns=['output'])
out = outcome.to_numpy()

#Create Training and Test sets
nums = np.linspace(0,302,303).astype("int")
trainind = np.random.choice(nums,250,replace=False)
testind = np.delete(nums,trainind)

#Training data
trinp = inp[trainind,:]
trout = np.ravel(out[trainind,:])

#Test data
teinp = inp[testind,:]
teout = np.ravel(out[testind,:])

#Fit Logistic Regression
LogReg = linear_model.LogisticRegression()
LogReg.fit(trinp,trout)

#Score Logistic Regression model
scoreLogReg = LogReg.score(teinp,teout)

#Fit Classification Tree
clTr = tree.DecisionTreeClassifier()
clTr.fit(trinp,trout)

#Score Classification Tree model
scoreClassTree = clTr.score(teinp,teout)

#Fit AdaBoost model
AdB = ensemble.AdaBoostClassifier()
AdB.fit(trinp,trout)

#Score AdaBoost model
scoreAdaBoost = AdB.score(teinp,teout)

#Fit Random Forest model
RdF = ensemble.RandomForestClassifier()
RdF.fit(trinp,trout)

#Score Random Forest model
scoreRandomForest = RdF.score(teinp,teout)