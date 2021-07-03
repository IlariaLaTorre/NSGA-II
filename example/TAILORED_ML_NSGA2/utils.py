from population import Population
import random
import numpy as np

class NSGA2Utils:

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.mutation_param = mutation_param

    def create_initial_population(self):
        population = Population()

        print("INITIAL POPULATION")
        for i in range(self.num_of_individuals):
            individual = self.problem.generate_individual(i)
            self.problem.calculate_objectives(individual)
            population.append(individual)
        return population

    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    #if individual dominates
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    #if individual is dominated
                    individual.domination_count += 1
            #if the individual is not dominated by anyone
            if individual.domination_count == 0:
                #first pareto front (most meaningful features in absolute = on the same level)
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            #check each individual in the current front
            for individual in population.fronts[i]:
                #check individuals dominated by the current individual
                for other_individual in individual.dominated_solutions:
                    #reduce step by step as each individual in the pareto front 0 is checked
                    other_individual.domination_count -= 1
                    #if other individual is dominated only by one individual (or by all or some the indivuduals in the pareto front 0)
                    if other_individual.domination_count == 0:
                        other_individual.rank = i+1
                        temp.append(other_individual)
            i = i+1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0
            #len = n of fitness functions
            #front[0] = first individual in the current parento front
            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m]) #the individual is the last one in the current parento front #sort (in an ascending order -3,-2,-1 / 1,2,3) individuals in the current front by the m-th objective values
                front[0].crowding_distance = 10**9 #first (gratest m-th ff value)
                front[solutions_num-1].crowding_distance = 10**9 #last (smallest m-th ff value)
                m_values = [individual.objectives[m] for individual in front] #list of values of the m-th fitness function for all the individuals in the current pareto front
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num-1): #compute crowding_distance for the remaing individuals (all - first and last)
                    front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m])/scale

    def crowding_operator(self, individual, other_individual): #if individual is better that other_individual
        if (individual.rank < other_individual.rank) or \
            ((individual.rank == other_individual.rank) and (individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        children = []
        #double the population with new childrens
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            comparison = parent1 == parent2
            equal_arrays = comparison.all()
            while equal_arrays:
                parent2 = self.__tournament(population)
                comparison = parent1 == parent2
                equal_arrays = comparison.all()
            child1, child2 = self.__crossover(parent1, parent2)
            self.__mutate(child1)
            self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)

        return children

    def __crossover(self, individual1, individual2):
        # child1 = self.problem.generate_individual(None)
        # child2 = self.problem.generate_individual(None)
        # num_of_features = len(child1.features)
        # genes_indexes = range(num_of_features)
        # for i in genes_indexes:
        #     beta = self.__get_beta()
        #     x1 = (individual1.features[i] + individual2.features[i])/2
        #     x2 = abs((individual1.features[i] - individual2.features[i])/2)
        #     child1.features[i] = x1 + beta*x2
        #     child2.features[i] = x1 - beta*x2
        # return child1, child2
        num_of_features = len(individual1.features[0])
        length_gene = len(individual1.features)
        crossover_pos = random.randint(1,num_of_features-2)

        #child1 = np.empty([length_gene, num_of_features], type = int)
        #child2 = np.empty([length_gene, num_of_features], type = int)

        child1 = self.problem.generate_individual(None)
        child2 = self.problem.generate_individual(None)

        #child1 = np.array([[0 for _ in range(num_of_features)] for _ in range(length_gene)])
        #child2 = np.array([[0 for _ in range(num_of_features)] for _ in range(length_gene)])

        child1.features[:,:crossover_pos] = individual1.features[:,:crossover_pos]
        child2.features[:,:crossover_pos] = individual2.features[:,:crossover_pos]
        for i in range(crossover_pos, num_of_features):
            child1.features[:,i], child2.features[:,i] = individual2.features[:,i], individual1.features[:,i]
        #individual1.features = child1
        #individual2.features = child2

        return child1, child2
            

    # def __get_beta(self):
    #     u = random.random()
    #     if u <= 0.5:
    #         return (2*u)**(1/(self.crossover_param+1))
    #     return (2*(1-u))**(-1/(self.crossover_param+1))

    def __mutate(self, child):
        # num_of_features = len(child.features)
        # for gene in range(num_of_features):
        #     u, delta = self.__get_delta()
        #     if u < 0.5:
        #         child.features[gene] += delta*(child.features[gene] - self.problem.variables_range[gene][0])
        #     else:
        #         child.features[gene] += delta*(self.problem.variables_range[gene][1] - child.features[gene])
        #     if child.features[gene] < self.problem.variables_range[gene][0]:
        #         child.features[gene] = self.problem.variables_range[gene][0]
        #     elif child.features[gene] > self.problem.variables_range[gene][1]:
        #         child.features[gene] = self.problem.variables_range[gene][1]
        num_of_features = len(child.features[0])
        length_gene = len(child.features)

        for gene in range(num_of_features):
            u, delta = self.__get_delta()
            if u < 0.5:
                for i in range(length_gene):
                    child.features[i][gene] += delta*(child.features[i][gene] - self.problem.variables_range[gene][0])
                    if child.features[i][gene] < self.problem.variables_range[gene][0]:
                        child.features[i][gene] = self.problem.variables_range[gene][0]
                    elif child.features[i][gene] > self.problem.variables_range[gene][1]:
                        child.features[i][gene] = self.problem.variables_range[gene][1]




    def __get_delta(self):
        u = random.random()
        if u < 0.5:
            return u, (2*u)**(1/(self.mutation_param + 1)) - 1
        return u, 1 - (2*(1-u))**(1/(self.mutation_param + 1))

    #select num_of_tour_particips chromosomes - then check the best one - the keep the best one only if rnd <= tournament_prob
    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False
