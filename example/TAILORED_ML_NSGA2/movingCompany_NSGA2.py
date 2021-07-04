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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics


import numpy as np
from pyitlib import discrete_random_variable as drv

import shap


from population import Population
import utils
from problem import Problem
from utils import NSGA2Utils
from evolution import Evolution
import sys


from datetime import datetime
now = datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = open('example/TAILORED_ML_NSGA2/data/output/movingCompany_NSGA2_'+now, 'w')


n_Ind = 2000
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




featuresRange = [(18,50),(0,1),(0,1),(0,3),(10,50)]

#Feature vectors generation
for n in range (n_Ind):
    age = (random.randint(18,50))
    gender = (genders[random.randint(0,1)])
    marital_status = (marital_statuses[random.randint(0,1)])
    education = (educations[random.randint(0,3)])
    lift_heavy_weight = (random.randint(10,50))

    
    # if gender == 'F':
    #     #lift_heavy_weight.append(random.randint(5,15))
    #     lift_heavy_weight = (np.random.choice([random.randint(5,15),random.randint(20,50)],p=[0.8,0.2]))
    # else:
    #     #lift_heavy_weight.append(random.randint(20,50))
    #     lift_heavy_weight = (np.random.choice([random.randint(5,15),random.randint(20,50)],p=[0.2,0.8]))

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
    if  features[heavy_weight][n] >= 30:
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

#model = "logisticRegression"
model = "randomForest"

if (model == "logisticRegression"):
    # Finalize Model
    # Logistic Regression
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, Y_train)
    predictions = logistic_regression.predict(X_validation)
    #print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    #['Age','Gender','Marital_Status','Education','Lift_Heavy_Weight']
    #print(logistic_regression.predict([[25,1,0,2,40]])[0])

if (model == "randomForest"):
    # Random Forest
    random_forest=RandomForestClassifier(n_estimators=100)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    random_forest.fit(X_train,Y_train)

    Y_pred=random_forest.predict(X_validation)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(Y_validation, Y_pred))

    print(random_forest.predict([[25,1,0,2,40]]))

#MUTUAL INFO DATASET VARIABLES
print("DATASET - Mutual Info")
for i in range(len(dataSet.values[0])-1):
    mi = drv.information_mutual(dataSet.values[:,i],dataSet.values[:,-1],cartesian_product=True)
    print(var[i])
    print(mi)

#FITNESS FUNCTIONS


bestFeatures_MI = []
length_gene = 2000
num_genes = len(applicants[0])
pop_size = 10
num_generations = 10


#c = logistic_regression
c = random_forest

def f(chromosome):

    y=[]
    #predict
    #len(chromosome) = 30 (observations)
    for i in range(length_gene):
        y.append(c.predict([chromosome[i]])[0])
        
    #compute mutual array using y
    M=[]
    Var = []
    ch = np.array(chromosome)
        
    for i in range(num_genes):
        M.append(drv.information_mutual(ch[:,i],np.array(y),cartesian_product=True))
        Var.append(dataSet.columns[i])

    #dictionary Variables-MI
  
    
    d = {"".join(Var[0]):M[0]}
    #print(d)

    for i in range(len(Var)):
        d["".join(Var[i])] = M[i]
    #print(d)

    sort_orders = sorted(d.items(), key=lambda x: x[1], reverse=True)
    #print(sort_orders)

    best = []

    den = 0
    num = 0
    res = 0
    summary = 0
    summary = sum(M)
    threshold = summary * 0.5
    
    M.sort(reverse=True)
    #print(M)

    #den = 0
    while num < threshold:
        num=num+M[den]
        den=den+1

    for i in range(den):
        best.append(sort_orders[i])
    bestFeatures_MI = [best]

    if (den == 0):
        den = len(Var)

    return [-(summary),den,bestFeatures_MI,sort_orders,summary]
        

def f1(chromosome):
    return f(chromosome)[0]

def f2(chromosome):
    return f(chromosome)[1]

def bestFeatures(chromosome):
    return f(chromosome)[2]

def allFeatures(chromosome):
    return f(chromosome)[3]

def totFF(chromosome):
    return f(chromosome)[4]



problem = Problem(objectives=[f1, f2], num_of_variables=num_genes, num_obs=length_gene, variables_range=featuresRange, meaningful_features=bestFeatures, allFeatures=allFeatures, totFF=totFF, same_range=False, expand=False)
evo = Evolution(problem, num_of_generations=num_generations, num_of_individuals=pop_size, mutation_param=20)

#Create Initial Population
initialPopulation = evo.createInitialPopulation()

#Population Evolution
generations = evo.evolve()
func = [i.objectives for i in generations[-1].fronts[0]] #N of fronts[0] after all the generations

#front 0 for each generation
for i in range(num_generations):   
    print("FRONT 0 GENERATION ", i)
    print([i.objectives for i in generations[i].fronts[0]])
    print([i.best_features for i in generations[i].fronts[0]])

sys.stdout.close()

#SHAPLEY VALUES
#shap_values.shape[1]

expl = shap.LinearExplainer(logistic_regression,X_train)
shap_values = expl.shap_values(X_validation)

#BAR PLOT
shap.summary_plot(shap_values, X_validation, var, plot_type="bar")
#shap.plots.bar(shap_values[0],show=True)
#print(shap_values.base_values)

#SUMMARY PLOT
shap.summary_plot(shap_values,
                  X_validation,
                  feature_names=var)



# print("\nLift Heavy Weight")
# print(drv.information_mutual(HD, X_validation[:,0], cartesian_product=True))
# print("Gender")
# print(drv.information_mutual(HD, X_validation[:,1], cartesian_product=True))
# print("Marital Status")
# print(drv.information_mutual(HD, X_validation[:,2], cartesian_product=True))
# print("Age")
# print(drv.information_mutual(HD, X_validation[:,3], cartesian_product=True))
# print("Education")
# print(drv.information_mutual(HD, X_validation[:,4], cartesian_product=True))

# print(drv.information_mutual_conditional(HD, X_validation[:,1], X_validation[:,0]))
# print(drv.information_mutual_conditional(HD, X_validation[:,0], X_validation[:,1]))