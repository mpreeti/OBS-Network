# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:47:14 2018

@author: hp
"""

import pandas as pd
import numpy as np
#%%
network_data_df=pd.read_csv(r'C:\Users\hp\Desktop\python\OBS_Network_data.csv',header=None,engine='python')
network_data_df

network_data_df.head()
network_data_df.shape

network_data_df.columns=["Node","Utilised Bandwith Rate","Packet Drop Rate","Full_Bandwidth","Average_Delay_Time_Per_Sec",
"Percentage_Of_Lost_Pcaket_Rate","Percentage_Of_Lost_Byte_Rate","Packet Received Rate","of Used_Bandwidth",
"Lost_Bandwidth","Packet Size_Byte","Packet_Transmitted","Packet_Received","Packet_lost","Transmitted_Byte",
"Received_Byte","10-Run-AVG-Drop-Rate","10-Run-AVG-Bandwith-Use","10-Run-Delay","Node Status","Flood Status","Class"]

network_data_df.head()

#%%Preprocessing data
network_data_df.isnull().sum()

pd.set_option("display.max_columns",None)#if col is hide  then use it
#%%copy of dataframe
network_data_rev=pd.DataFrame.copy(network_data_df)
network_data_rev.head()

#%%removing the column which was having  only 1 value
network_data_rev=network_data_rev.drop('Packet Size_Byte',axis=1)#axis=1:search in  column   header
network_data_rev.head()

#%%convert cat to   num
colname=['Node','Full_Bandwidth','Node Status','Class']
colname

from sklearn import preprocessing
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()#convert cat to num by label encoding
    
for x in colname:
    network_data_rev[x]=le[x].fit_transform(network_data_rev.__getattr__(x))
    
network_data_rev.head()

#%%Separate dep and indep    var
X=network_data_rev.values[:,:-1]
Y=network_data_rev.values[:,-1]
#%%Normalizing data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() #to avoid biasing in model
scaler.fit(X)
X=scaler.transform(X)
Y=Y.astype(int)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)

#%%Running Decision Tree Model
#prediction using Decision_Tree_Classifier
from sklearn.tree import  DecisionTreeClassifier

model_DecisionTree=DecisionTreeClassifier()
model_DecisionTree.fit(X_train,Y_train)
#%%fit the model on the data n predict values
Y_pred=model_DecisionTree.predict(X_test)
print(Y_pred)
print(list(zip(Y_test,Y_pred)))

#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred) #for comparison
print(cfm)

print("Classification report:")
print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",accuracy_score)#100%accuracy chk for overfitting

#%%Cross  validation(for rechecking our accuracy)
classifier=(DecisionTreeClassifier())

from sklearn import cross_validation
#performing k fold cross val
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
                                                 y=Y_train,scoring="accuracy",cv=kfold_cv)
print(kfold_cv_result)  #gives list of 10 values of accuracy[......]

#finding mean

print(kfold_cv_result.mean())#stop in case to evaluate

"""for train_value,test_value in kfold_cv:
    classifier.fit(X_train[train_value],Y_train[train_value]).predict(X_train[test_value])#predict wala will be 10th part n previous wala 9 parts train,train_value is training data
Y_pred=classifier.predict(X_test)

"""
#%%Using Logistic  Reg
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())
#fitting training data tothe model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred) #for comparison
print(cfm)

print("Classification report:")
print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",accuracy_score)

#accuracy   isless  coz  we hv 4 class n logistic reg is good for binary classification
#%%Using SVM
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
#from sklearn.linear_model import LogisticRegression
#svc_model=LogisticRegression()
svc_model.fit(X_train,Y_train)#Y_train is loan status
Y_pred=svc_model.predict(X_test)#using X_test we r predicting Y_pred of testing data
print(list(Y_pred))
Y_pred_col=list(Y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred) #for comparison
print(cfm)

print("Classification report:")
print(classification_report(Y_test,Y_pred))

accuracy_score=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",accuracy_score)

#%%Graph--store in   working dir n go online webgraphicviz.com...copy txt on  web it shows graph

from sklearn import tree
with open("model_DecisionTree.txt","w") as f:
    f=tree.export_graphviz(model_DecisionTree,out_file=f)
#%%Running ExtraTree classifier

#predicting using the Bagging_Classifier
from sklearn.ensemble import ExtraTreesClassifier

model=(ExtraTreesClassifier(21))#no.of bags  is  21  i.e,21   Dec tree
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#%%
#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(501)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#%%
#predicting using the AdaBoost_Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
#fit the model on the data and predict the values
model_AdaBoost.fit(X_train,Y_train)

Y_pred=model_AdaBoost.predict(X_test)

 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#%%
#predicting using the Gradient_Boost_Classifier
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier()
#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)

 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


#%%  Ensemble  Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('log', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)

 
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred)) #coz of  LR   as  it is used for binary classification as example is of multiclass classification













