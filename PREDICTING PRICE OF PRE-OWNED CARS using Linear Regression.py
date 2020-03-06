# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:39:39 2020

@author: Ravishankar
"""

##############################################
##PREDICTING PRICE OF PRE-OWNED CARS
##############################################

#To work with Data Frames
import pandas as pd
#To perform numerical operations
import numpy as np
#To visualize Data
import matplotlib.pyplot as plt
import seaborn as sns

################################
##Setting Dimention for Plot
#################################
sns.set(rc={'figure.figsize':(11.7,8.27)})

#################
#Importing Data 
#################
cars_data=pd.read_csv('cars_sampled.csv')

###################################
#creating a copy of original data
###################################
cars=cars_data.copy()

##################################
#Structure of data
###################################
cars.info()

##################################
#Summerizing Data
##################################

cars.describe()

#To display maximum set of  columns and 
pd.set_option('display.float_format',lambda x:'%.3f'% x)
pd.set_option('display.max_columns',20)
cars.describe()

##################################
#Droping unwanted columns
##################################
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)


##################################
# Removing duplicate records
##################################
cars.drop_duplicates(keep='first',inplace=True)

#Note: 470 duplicate records

#++++++++++++++++++++++++++++
#Data cleaning
#+++++++++++++++++++++++++++++

#No.of missing values in each columns
cars.isnull().sum()

#variable yearofRegistration
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)

#Working range - 1950 and 2018

#Variable price

price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])

cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)

#working range 100 and 150000

#variable powerPS

power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#working range- 10 and 500

#++++++++++++++++++++++++++
# Working range of data
#+++++++++++++++++++++++++++

cars=cars[
        (cars['yearOfRegistration']<=2018)
        &(cars['yearOfRegistration']>=1950)
        &(cars['price']<=150000)
        &(cars['price']>=100)
        &(cars['powerPS']<=500)
        &(cars['powerPS']>=10)]

#Note: 6700 records are dropped

#further simplify- variable reduction
#combining yearOfRegistration and monthOfRegistration

cars['monthOfRegistration']/=12
#Creating new variable by adding yearOfRegistration and monthOfRegistration
cars['age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['age']=round(cars['age'],2)
cars['age'].describe()

#Dropping yearOfRegistration and monthOfRegistration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1) 

#+++++++++++++++++++++++++++++++++++++
# Data Visualizing 
#+++++++++++++++++++++++++++++++++++++

#visualiing parameters

#Age
sns.distplot(cars['age'])
sns.boxplot(y=cars['age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#visualizing parameter after narrowing working range

#age vs price
sns.regplot(x='age',y='price',scatter=True,fit_reg=False,data=cars)

#Note: with age increse the price decrease

#powerPS vs price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)

#Note: with power increase , price also increase

#++++++++++++++++++++++++++++++++++++
#visualizing categorical variables
#++++++++++++++++++++++++++++++++++++

#variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
#Note: Fewer cars have commercial 
#we can remove record with commercial data because its contain only 1 value

#variable offerType
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'],columns='count',normalize=True)
sns.countplot(x='offerType',data=cars)
#Note: All cars have "offer"

#variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
#Note: both are almost equally distributed
sns.boxplot(x='abtest',y='price', data=cars)
#Note:price value also almost 50-50 distributed , not much affect on price

#variable vehicalType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
#Note: 
sns.boxplot(x='vehicleType',y='price', data=cars)

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
#Note: 
sns.boxplot(x='gearbox',y='price', data=cars)

#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
#Note: 
sns.boxplot(x='model',y='price', data=cars)

#variable fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
#Note: 
sns.boxplot(x='fuelType',y='price', data=cars)

#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data =cars)
#Note: 
sns.boxplot(x='brand',y='price', data=cars)

#variable notRepairedDamage
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data =cars)
#Note: 
sns.boxplot(x='notRepairedDamage',y='price', data=cars)

#+++++++++++++++++++++++++++++++++++++
#Removing insignificant values
#+++++++++++++++++++++++++++++++++++++

col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#++++++++++++++++++++++++++++++++++++++++
#Correlation
#++++++++++++++++++++++++++++++++++++++++

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

############################################
# Omitting Missing Values
##############################################
cars_omit=cars.dropna(axis=0)

#Converting categorical variable to dummy variables
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

#++++++++++++++++++++++++++++++++++++++++++
# Importing Necessary Libraries
#++++++++++++++++++++++++++++++++++++++++++

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#+++++++++++++++++++++++++++++++++++++++++
# MOdel Building with omitted Data
#++++++++++++++++++++++++++++++++++++++++++++

#separating input and output features
x1=cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']

#Plotting the variable price
prices=pd.DataFrame({"1.Before":y1,"2. After":np.log(y1)})
prices.hist()

#Transforming price as a Logarithmic values 
y1=np.log(y1)

#Spliting data into test and train
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#++++++++++++++++++++++++++++++++++++++++++++
# Baseline Model for Omitted Data
#++++++++++++++++++++++++++++++++++++++++++++

#Finding the mean for test data value
base_pred=np.mean(y_test)
print(base_pred)

#Repeating same value till length of test data

base_pred=np.repeat(base_pred,len(y_test))

#Finding the RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)

#+++++++++++++++++++++++++++++++++++++++++++++
#Linear Regression with Omitted Data
#+++++++++++++++++++++++++++++++++++++++++++++

#Setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#Model
model_lin1=lgr.fit(x_train,y_train)

#Predicting model on test set
cars_prediction_lin1=lgr.predict(x_test)

#computing MSE and RMSE
lin_mse1=mean_squared_error(y_test,cars_prediction_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R Squared Value
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regressionn diagnostics- residual plot analysis
residuals1=y_test-cars_prediction_lin1
sns.regplot(x=cars_prediction_lin1,y=residuals1,scatter=True,fit_reg=False)
residuals1.describe()

#++++++++++++++++++++++++++++++++++++++++++++
# Random Forest with omitted Data
#++++++++++++++++++++++++++++++++++++++++++++

#Model parameters
rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4, random_state=1)

#model
model_rf1=rf.fit(x_train,y_train)

#Predicting model on test set
cars_prediction_rf1=rf.predict(x_test)

#computing MSE and RMSE

rf_mse1=mean_squared_error(y_test,cars_prediction_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

#R Squared Value
r2_rf_test1=model_rf1.score(x_test,y_test)
r2_rf_train1=model_rf1.score(x_train,y_train)
print(r2_rf_test1,r2_rf_train1)

#++++++++++++++++++++++++++++++++++++++++++++++++++
# MOdel Building with imputed data
#++++++++++++++++++++++++++++++++++++++++++++++++++

cars_imputed=cars.apply(lambda x:x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

#converting categorical variable to dummy variables
cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)

#separating input and output features
x2=cars_imputed.drop(['price'],axis='columns',inplace=False)
y2=cars_imputed['price']

#Plotting the variable price
prices=pd.DataFrame({"1.Before":y2,"2. After":np.log(y2)})
prices.hist()

#Transforming price as a Logarithmic values 
y1=np.log(y2)

#Spliting data into test and train
x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(x_train1.shape,x_test1.shape,y_train1.shape,y_test1.shape)

#++++++++++++++++++++++++++++++++++++++++++++
# Baseline Model for Imputed Data
#++++++++++++++++++++++++++++++++++++++++++++

#Finding the mean for test data value
base_pred1=np.mean(y_test1)
print(base_pred1)

#Repeating same value till length of test data

base_pred1=np.repeat(base_pred1,len(y_test1))

#Finding the RMSE
base_root_mean_square_error_imputed=np.sqrt(mean_squared_error(y_test1,base_pred1))
print(base_root_mean_square_error_imputed)

#+++++++++++++++++++++++++++++++++++++++++++++
#Linear Regression with Imputed Data
#+++++++++++++++++++++++++++++++++++++++++++++

#Setting intercept as true
lgr2=LinearRegression(fit_intercept=True)

#Model
model_lin2=lgr2.fit(x_train1,y_train1)

#Predicting model on test set
cars_prediction_lin2=lgr2.predict(x_test1)

#computing MSE and RMSE
lin_mse2=mean_squared_error(y_test1,cars_prediction_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)

#R Squared Value
r2_lin_test2=model_lin2.score(x_test1,y_test1)
r2_lin_train2=model_lin2.score(x_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

#++++++++++++++++++++++++++++++++++++++++++++
# Random Forest with Imputed Data
#++++++++++++++++++++++++++++++++++++++++++++

#Model parameters
rf2=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4, random_state=1)

#model
model_rf2=rf2.fit(x_train1,y_train1)

#Predicting model on test set
cars_prediction_rf2=rf2.predict(x_test1)

#computing MSE and RMSE

rf_mse2=mean_squared_error(y_test1,cars_prediction_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

#R Squared Value
r2_rf_test2=model_rf2.score(x_test1,y_test1)
r2_rf_train2=model_rf2.score(x_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)


#+++++++++++++++++++++++++++++++++++++++++++++++
# END of script
#+++++++++++++++++++++++++++++++++++++++++++++++









