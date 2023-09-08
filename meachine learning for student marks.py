# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:42:59 2023

@author: my pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the file 
df=pd.read_csv(r"D:\notes\4. Students mark prediction\student_info.csv")
df

df.head()

df.tail()

df.describe()

df.info()

df.shape

#visualize the data
plt.scatter(x=df["study_hours"],y=df["student_marks"])
plt.xlabel("no.of how student study")
plt.ylabel("marks of the students")
plt.title("no.of hours study vs marks of the student")
plt.show()

#Test the dataset contain missing or not
df.isnull().sum()

#replace the null value by using mean 
df["study_hours"].mean()

#fill the missing value with mean of the features
df2=df.fillna(df.mean())

#again check wheather dataset contain the nullvalues or not 
df2.isnull().sum()

df2.head()

#identify the dependent varibule and indepedent varibules from the dataset
X=df2.iloc[:,0:1].values  #independent varibule

y=df2.iloc[:,1].values    #Depedent varibule

#split the data into the train and test dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#crate a regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

regressor.coef_

regressor.intercept_


m=3.93
c=50.48
y=m*4+c
y

regressor.predict([[4]]).round(2)

y_pred=regressor.predict(X_test)
y_pred

#Display the actual marks and predicted marks
pd.DataFrame(np.c_[X_test,y_test,y_pred],columns=["no.of study hours","actual marks","predicted marks"])

#plot the training dataset
plt.scatter(X_train,y_train)

#plot the test dataset and predict data
plt.scatter(X_test,y_test,color="red")
plt.plot(X_test,regressor.predict(X_test))


#Save the machine learning model

import joblib
joblib.dump(regressor,"student_marks_pred.pkl")


#import the ml model
model=joblib.load("student_marks_pred.pkl")

model.predict([[5]])
