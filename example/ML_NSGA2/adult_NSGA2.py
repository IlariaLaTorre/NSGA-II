from problem import Problem
from evolution import Evolution
import matplotlib.pyplot as plt
import math
from classifier import binaryClassifier
import numpy as np
from pyitlib import discrete_random_variable as drv
import pickle
import sys


sys.stdout = open('example/ML_NSGA2/output2', 'w')


c = binaryClassifier()
bestFeatures_MI = []
classifier = c
length_gene = 10
num_genes = 8
pop_size = 10



def f(chromosome):

    y=[]
    #predict
    #len(chromosome) = 30 (observations)
    for i in range(length_gene):
        y.append(c.predict([chromosome[i]])[0])
        
    #compute mutual array using y
    M=[]
    ch = np.array(chromosome)
        
    for i in range(num_genes):
        M.append(drv.information_mutual(ch[:,i],np.array(y),cartesian_product=True))
   
    #dictionary Variables-MI
    Var = ["age","education.num","marital.status","race","sex","capital.gain","capital.loss","hours.per.week"]

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
    threshold = summary * 0.8
    
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

    return [-num,den,bestFeatures_MI,sort_orders,summary]
        

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
    

problem = Problem(objectives=[f1, f2], num_of_variables=num_genes, num_obs=length_gene, variables_range=[(17,90),(1,16),(0,1),(0,4),(0,1),(0,99999),(0,4356),(1,99)], meaningful_features=bestFeatures, allFeatures=allFeatures, totFF=totFF, same_range=False, expand=False)
evo = Evolution(problem, num_of_generations=5, num_of_individuals=pop_size, mutation_param=20)
#func = [i.objectives for i in evo.evolve()] #N of fronts[0] after all the generations

generations = evo.evolve()
func = [i.objectives for i in generations[-1].fronts[0]] #N of fronts[0] after all the generations

#
# for pop in generations:
#     func = [i.objectives for i in pop.fronts[0]] 
#     function1 = [i[0] for i in func]
#     function2 = [i[1] for i in func]
#     plt.xlabel('Function 1', fontsize=15)
#     plt.ylabel('Function 2', fontsize=15)
#     plt.scatter(function1, function2)
#     plt.show()


#func = [i.objectives for i in evo.evolve()] #N of fronts[0] after all the generations

# function1 = [i[0] for i in func]
# function2 = [i[1] for i in func]
# plt.xlabel('Function 1', fontsize=15)
# plt.ylabel('Function 2', fontsize=15)
# plt.scatter(function1, function2)
# plt.show()



with open('generations.pickle', 'wb') as f:
    pickle.dump(generations, f)

g = 0
for pop in generations:
    print("\n")
    print("GENERATION: ", g)
    for i in range(len(pop.fronts)):
        pf = [i.objectives for i in pop.fronts[i]] 
        print("Pareto Front ", i)
        print(pf)
    g = g+1

sys.stdout.close()