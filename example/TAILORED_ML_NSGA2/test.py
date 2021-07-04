import problem
from movingCompany_NSGA2 import logistic_regression, dataSet
import numpy as np
from pyitlib import discrete_random_variable as drv


individual = problem.generate_individual(1)

c = logistic_regression

def f(chromosome):

    y=[]

    for i in range(100):
        y.append(c.predict([chromosome[i]])[0])
        
    M=[]
    Var = []
    ch = np.array(chromosome)
        
    for i in range(100):
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

    print(sort_orders)

f(individual)