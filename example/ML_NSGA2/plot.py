import pickle
import matplotlib.pyplot as plt
from population import Population
import random
from random import randint

colors = []
plots = []
paretoLabels = []

random.seed(10)


with open('example/ML_NSGA2/generations.pickle','rb') as f:
    #f.encode('utf-8').strip()
    #text=f.read().decode(errors='replace')
    generations = pickle.load(f)


for pop in generations:
    #for i in range(len(pop.fronts)):
    if (len(pop.fronts)) > 4:
        paretoLabels = ["Pareto Front 1","Pareto Front 2","Pareto Front 3","Pareto Front 4"]
        for i in range(4):
            for j in range(4):
                colors.append('#%06X' % randint(0, 0xFFFFFF))
            func = [i.objectives for i in pop.fronts[i]] 
            function1 = [i[0] for i in func]
            function2 = [i[1] for i in func]
            plt.xlabel('Function 1', fontsize=15)
            plt.ylabel('Function 2', fontsize=15)
            if i==0:
                plots.append(plt.scatter(function1, function2,color="black"))
            else:
                plots.append(plt.scatter(function1, function2,color=colors[i]))
        colors.append("black")
        plt.legend(plots,paretoLabels)    
        plt.show()
    else:
        for i in range(len(pop.fronts)):
            paretoLabels.append(("Pareto"+(i+1)))
            for j in range(len(pop.fronts)):
                colors.append('#%06X' % randint(0, 0xFFFFFF))
            func = [i.objectives for i in pop.fronts[i]] 
            function1 = [i[0] for i in func]
            function2 = [i[1] for i in func]
            plt.xlabel('Function 1', fontsize=15)
            plt.ylabel('Function 2', fontsize=15)
            if i==0:
                plots.append(plt.scatter(function1, function2,color="black"))
            else:
                plots.append(plt.scatter(function1, function2,color=colors[i]))
        colors.append("black")
        plt.legend(plots,paretoLabels)    
        plt.show()

