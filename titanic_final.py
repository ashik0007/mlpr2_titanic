# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:41:17 2018

@author: ASHIK
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
titanic = train.append(test,ignore_index='True') #merged train and test data for preprocessing
PassengerId = test.PassengerId #kept the PassengerId column for test results

titanic.drop(['PassengerId','Name','Ticket','Survived'],1,inplace=True) #dropped these columns as they have the least impact
                                                                        #their title from name have impact but as this is a 
                                                                        #beginner approach it was not considered
titanic['Familymember'] = titanic.SibSp + titanic.Parch + 1 #SibSp and Parch colums were merged to create one single column

#all the missing values were filled for better accuracy
titanic.Pclass.fillna(titanic.Pclass.value_counts().index[0] , inplace=True)
titanic.Age.fillna(titanic.Age.median(),inplace=True)
titanic.Embarked.fillna(titanic.Embarked.value_counts().index[0] , inplace=True)
titanic.Fare.fillna(titanic.Fare.median(),inplace=True)
titanic.Cabin.fillna(titanic.Cabin.value_counts().index[0],inplace=True)

titanic.Cabin = titanic.Cabin.map(lambda x: x[0]) #Cabin number's first letter is significant and the rest is not

#Used LabelEncoder for transforming catagorical values into numbers
le = LabelEncoder()
encoded_Sex = le.fit_transform(titanic.Sex)
encoded_Cabin = le.fit_transform(titanic.Cabin)
encoded_Embarked = le.fit_transform(titanic.Embarked)

dummy = np.matrix.transpose(np.stack((encoded_Sex,encoded_Cabin,encoded_Embarked))) #these 3 columns are merged together
titanic.drop(['Sex','Cabin','Embarked'],1,inplace=True) #old catagorical columns are to be dropped
titanic_dummy = np.concatenate((titanic,dummy),axis=1) #new numerical columns are joined in place of the catagorical columns

#dividing into train and test samples
X_train = titanic_dummy[0:891]
X_test = titanic_dummy[891:1310]
y_train = train.Survived

#AdaBoostClassifier is used for a better approximation
clf = AdaBoostClassifier(n_estimators=90,learning_rate=.9)

clf.fit(X_train,y_train) #trained the model
pred = clf.predict(X_test) #predicted for unseen data
print(pred)

#storing the results from the prediction to a .csv file
submission = pd.DataFrame({'PassengerId':PassengerId , 'Survived':pred})
submission.to_csv("C:/Users/ASHIK/Desktop/Projects/Titanic/Titanic_prediction.csv", index=False) 
