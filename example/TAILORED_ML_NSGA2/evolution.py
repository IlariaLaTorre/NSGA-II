from utils import NSGA2Utils
from population import Population
from copy import copy

class Evolution:

    def __init__(self, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param)
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals

    def createInitialPopulation(self):
        self.population = self.utils.create_initial_population()
        return self.population

    def evolve(self):
        #self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population) #create fronts
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None
        #
        generations = []
        for i in range(self.num_of_generations):
            print("GENERATION N: ", i)
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals: #new_population will contain all the fronts until the num_of_individuals is reached
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals-len(new_population)]) #original size pop with best fronts
            returned_population = self.population #doubled pop
            self.population = new_population
            self.utils.fast_nondominated_sort(self.population)
            for i in range(len(self.population.population)):
                print("\n")
                print("Individual ", i)
                print("Fitness values :")
                print(self.population.population[i].objectives)
                print(self.population.population[i].best_features)
            print("BEST INDIVIDUAL: ")
            print(self.population.fronts[0][0].objectives)
            print(self.population.fronts[0][0].best_features)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)
            #
            returned_population = self.population
            generations.append(copy(returned_population))
        #return returned_population.fronts[0]  #last population fronts[0]
        return generations 
