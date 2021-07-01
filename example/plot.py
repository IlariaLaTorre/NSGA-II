import pickle
import matplotlib.pyplot as plt

with open('generations.pickle') as f:
    generations = pickle.load(f)

for pop in generations:
    func = [i.objectives for i in pop.fronts[0]] 
    function1 = [i[0] for i in func]
    function2 = [i[1] for i in func]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()