from individual import Individual
import random
import numpy as np

class Problem:

    def __init__(self, objectives, num_of_variables, num_obs, variables_range, meaningful_features, allFeatures, totFF, same_range=False, expand=True):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.num_obs = num_obs
        self.objectives = objectives
        self.expand = expand
        self.variables_range = []
        self.meaningful_features = meaningful_features
        self.allFeatures = allFeatures
        self.totFF = totFF
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

    def generate_individual(self, num_individual):
        individual = Individual()
        individual.num_individual = num_individual

        X = np.full((self.num_obs, self.num_of_variables), 0, int)

        for i in range(len(X)):
            for j in  range(len(X[i])):
                X[i][j] = random.randint(self.variables_range[j][0],self.variables_range[j][1])
                
        individual.features = X

        if (num_individual != None):
            print("\n")
            print("Individual ", num_individual)
            print(individual.features)

        return individual

    def calculate_objectives(self, individual):
        if self.expand:
            individual.objectives = [f(*individual.features) for f in self.objectives]
        else:
            individual.objectives = [f(individual.features) for f in self.objectives]

        individual.best_features = self.meaningful_features(individual.features)
        
        if (individual.num_individual != None):
            print("Fitness values :")
            print(individual.objectives)

            print(self.meaningful_features(individual.features))
            print(self.totFF(individual.features))
            print(self.allFeatures(individual.features))
