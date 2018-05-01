
# coding: utf-8

# In[19]:

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#importing libraies
headers = ["Id", "MSSubClass", "MSZoning", "LotArea", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual",
           "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle" , "MasVnrType", "MasVnrArea", "1stFlrSF", "2ndFlrSF", 
           "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
          "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "SaleCondition", "SoldStatus"]


# In[20]:

df_train = pd.read_csv("trainSold.csv")
df_train.head()
df_test = pd.read_csv("testSold.csv")
#Importing the file
y_train = df_train["SaleStatus"]
#Making a seperate column of salestatus
x_train = df_train.drop('SaleStatus', 1)
#Dropping salestatus from training data to merge both training and testing data 
x_test = df_test
data = pd.concat([x_train,x_test],axis = 0)
#merging both data now row wise


# In[21]:

data.dtypes
#Getting data types


# In[22]:

A = data.loc[:, data.dtypes == object]
#selecting only object types from merged file
A = list(A.columns.values)
print(A)
#getting only object type to apply one hotting 


# In[23]:

data2 = data[A]
#Creating a backup
data = data.drop(A,1)
data = pd.concat([data, pd.get_dummies(data2, columns=A)],axis = 1)
#Applying one hotting to all the object type data
data.head()


# In[24]:

B = data.columns[data.isnull().any()].tolist()
#Findind the NaN
for i in B:
    data[i].fillna(data[i].mean(),inplace = True)
    #Replacing them with average values of the columns


# In[25]:

A = A = df_train.loc[:, df_train.dtypes != object]
#Selectig all the non object in data 
A = list(A.columns.values)


# In[26]:

from sklearn import preprocessing
x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)
data.head()
#normalising all the data


# In[27]:

x_train = data.iloc[:1460,:]
x_test = data.iloc[1460:,:]
#Now splitting up both train and test data


# In[28]:

clf = DecisionTreeClassifier()
#Applying classifiers to get the accuracy
score = cross_val_score(clf,x_train,y_train,cv = 5)
#print(dec_tree_cv.best_params_)
print(score.mean())


# In[29]:

import matplotlib.pyplot as plt
#Hypertuining the decision tree classifer wth max depth
score_all = []
for i in range(1,40):
    clf = DecisionTreeClassifier(max_depth = i)
    score = cross_val_score(clf,x_train,y_train,cv = 5)
    score_all.append(score.mean())
    print(score.mean())
plt.scatter(range(1,40),score_all)
plt.show()


# In[30]:

score_all = []
#Hypertuining the decision tree classifer wth max feature while taking depth 13
for i in range(2,106):
    clf = DecisionTreeClassifier(max_depth = 13 , max_features = i)
    score = cross_val_score(clf,x_train,y_train,cv = 5)
    score_all.append(score.mean())
    print(score.mean(),i)
plt.scatter(range(2,106),score_all)
plt.show()


# In[31]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#another classifier to get accuracy of training data
rf = RandomForestClassifier()
score = cross_val_score(rf,x_train,y_train,cv = 5)
print(score.mean())


# In[32]:

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
score = cross_val_score(rf,x_train,y_train,cv = 5)
print (score.mean())
#Another classifier to get accuracy


# In[33]:

clf = DecisionTreeClassifier()
clf.fit (x_train, y_train)
y_train = clf.predict(x_train)
#getting the answers for traing data with classifier that gives max axxuracy
print (y_train)


# In[34]:

rf.fit (x_train, y_train)
y_train = clf.predict(x_train)
#getting the answers for training data with another classifier
print (y_train)


# In[35]:

clf = DecisionTreeClassifier()
clf.fit (x_train, y_train)
y_test = clf.predict(x_test)
#getting the answers for test data with classifier that gives max axxuracy
print (y_test)


# In[36]:


rf.fit (x_train, y_train)
y_test = clf.predict(x_test)
#getting the answers for test data with another classifier
print (y_test)

