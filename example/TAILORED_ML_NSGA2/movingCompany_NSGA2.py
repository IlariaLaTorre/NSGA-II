#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 03:07:32 2021

@author: ilaria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:29:47 2021

@author: ilaria
"""


import random
from random import (
    seed,
    randrange
) 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
from pyitlib import discrete_random_variable as drv

import shap



n_Ind = 200
var = ["age","gender","marital_status","education","lift_heavy_weight"]
var_2 = ["age","Gender:Female","Gender:Male", "marital_status","education","lift_heavy_weight"]
var_values = [[18,50],['F','M'],['single', 'married'],['primary', 'secondary', 'further', 'higher'],[5,50]]
gender_values = [[1,0]]

rows, cols = (n_Ind, var) 
applicants = [[]for j in range(rows)] 
applicants_test = [[]for j in range(rows)] 

genders = ['F','M']
marital_statuses = ['single', 'married']
educations = ['primary', 'secondary', 'further', 'higher']

age_vector = []
gender_vector = []
marital_status_vector = []
education_vector = []
lift_heavy_weight_vector = []
hiring_vector = []
tot_infl = []

change_position = []
change_position_F = []
l = 0
l_F = 0
num_F = []

y = []






def decide(applicant):
    gender = 1
    education = 3
    if applicant[education] == 'primary':
        return True
        if applicant[gender] == 'F':
            r = randrange(0, 1)
            if r > 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

#classification logic
def decide2(applicant):
    gender = 1
    heavy_weight = 4
    if  applicant[heavy_weight] >= 20:
        return True
        if applicant[1] == 'F':
            r = random.randrange(0, 1)
            if r > 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def decide3(applicant):
    new_record = [applicant]
    #print("APPLICANT")
    #print(new_record[0])
    #print("CLASSIFIER PREDICTION BEFORE INTERVENTION")
    #print(logistic_regression.predict(new_record)[0])
    return (logistic_regression.predict(new_record)[0])

#random.seed(3)

#Feature vectors generation
for n in range (n_Ind):
    age = (random.randint(18,50))
    gender = (genders[random.randint(0,1)])
    marital_status = (marital_statuses[random.randint(0,1)])
    education = (educations[random.randint(0,3)])
    
    if gender == 'F':
        #lift_heavy_weight.append(random.randint(5,15))
        lift_heavy_weight = (np.random.choice([random.randint(5,15),random.randint(20,50)],p=[0.8,0.2]))
    else:
        #lift_heavy_weight.append(random.randint(20,50))
        lift_heavy_weight = (np.random.choice([random.randint(5,15),random.randint(20,50)],p=[0.2,0.8]))

    applicants[n].append(age)
    applicants[n].append(gender)
    applicants[n].append(marital_status) 
    applicants[n].append(education)
    applicants[n].append(lift_heavy_weight)

    age_vector.append(age)
    gender_vector.append(gender)
    marital_status_vector.append(marital_status)
    education_vector.append(education)
    lift_heavy_weight_vector.append(lift_heavy_weight)

    features = [age_vector,gender_vector,marital_status_vector,education_vector,lift_heavy_weight_vector]


#Ground Truth Generation Logic (with bias embedded in the generation of the dataset=all individuals with LHW < 20 kg = NO/ all men with LHW >= 20 kg = YES / Penalize Women that lift more thatn 20 kg)
gender = 1
heavy_weight = 4
for n in range (n_Ind):
    if  features[heavy_weight][n] >= 20:
        if features[gender][n] == 'F':
            r = random.randrange(0, 1)
            if r > 0:
            #if features[heavy_weight][n] >= 40:
                hiring_vector.append('Yes')
            else:
                hiring_vector.append('No')
        else:
            hiring_vector.append('Yes')
    else:
        hiring_vector.append('No')
        


features = [age_vector,gender_vector,marital_status_vector,education_vector,lift_heavy_weight_vector,hiring_vector]

dataSet = pd.DataFrame(features).transpose()
dataSet.columns=['Age','Gender','Marital_Status','Education','Lift_Heavy_Weight','Hiring_Decision']
print("DATASET")
print(dataSet)
    


#----Classifier----

# Check for Null Data
dataSet.isnull().sum()

# Replace All Null Data in NaN
dataSet = dataSet.fillna(np.nan)

# Reformat Categorical Variables
dataSet['Gender']=dataSet['Gender'].map({'F': 1, 'M': 0})
dataSet['Marital_Status']=dataSet['Marital_Status'].map({'married': 0, 'single': 1})
dataSet['Education']=dataSet['Education'].map({'primary': 0, 'secondary': 1,'further':2,'higher':3})
dataSet['Hiring_Decision']=dataSet['Hiring_Decision'].map({'Yes': 1, 'No': 0})

# Confirm All Missing Data is Handled
dataSet.isnull().sum()

print(dataSet.head(4))

# Split-out Validation Dataset and Create Test Variables
array = dataSet.values
X = array[:,0:5]
Y = array[:,5]
print('Split Data: X')
print(X)
print('Split Data: Y')
print(Y)

validation_size = 0.20
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)

results = []

kfold = KFold(n_splits=10)#, random_state=seed)
cv_results = cross_val_score(LogisticRegression(), X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % ("LR", cv_results.mean(), cv_results.std())
print(msg)

# Finalize Model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
predictions = logistic_regression.predict(X_validation)
#print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#Create Initial Population
print("INITIAL POPULATION")
for i in range(self.nInd):
    individual = self.generate_individual(i)
    self.problem.calculate_objectives(individual)
    population.append(individual)


#Test Data Generation
for n in range (n_Ind):
    age_test = (random.randint(18,50))
    gender_test = (random.randint(0,1))
    marital_status_test = (random.randint(0,1))
    education_test = (random.randint(0,3))
    
    if gender_test == 'F':
        #lift_heavy_weight.append(random.randint(5,15))
        lift_heavy_weight_test = (np.random.choice([random.randint(5,15),random.randint(20,50)],p=[0.8,0.2]))
    else:
        #lift_heavy_weight.append(random.randint(20,50))
        lift_heavy_weight_test = (np.random.choice([random.randint(5,15),random.randint(20,50)],p=[0.2,0.8]))

    applicants_test[n].append(age_test)
    applicants_test[n].append(gender_test)
    applicants_test[n].append(marital_status_test) 
    applicants_test[n].append(education_test)
    applicants_test[n].append(lift_heavy_weight_test)

    y.append(logistic_regression.predict(applicants_test[n])[0])

        
        
print("\nLift Heavy Weight")
print(drv.information_mutual(lift_heavy_weight_vector, hiring_vector, cartesian_product=True))
print("Gender")
print(drv.information_mutual(gender_vector, hiring_vector, cartesian_product=True))
print("Marital Status")
print(drv.information_mutual(marital_status_vector, hiring_vector, cartesian_product=True))
print("Age")
print(drv.information_mutual(age_vector, hiring_vector, cartesian_product=True))
print("Education")
print(drv.information_mutual(education_vector, hiring_vector, cartesian_product=True))

#MI on Test Set and Classifier decision
HD = []

for i in range(len(X_validation)):
    HD.append(logistic_regression.predict(X_validation)[i])


print("\nLift Heavy Weight")
print(drv.information_mutual(HD, X_validation[:,0], cartesian_product=True))
print("Gender")
print(drv.information_mutual(HD, X_validation[:,1], cartesian_product=True))
print("Marital Status")
print(drv.information_mutual(HD, X_validation[:,2], cartesian_product=True))
print("Age")
print(drv.information_mutual(HD, X_validation[:,3], cartesian_product=True))
print("Education")
print(drv.information_mutual(HD, X_validation[:,4], cartesian_product=True))

print(drv.information_mutual_conditional(HD, X_validation[:,1], X_validation[:,0]))
print(drv.information_mutual_conditional(HD, X_validation[:,0], X_validation[:,1]))