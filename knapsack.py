#COMP131 Assignment 3: The Knapsack Problem
#Author: Thomas McNulty
#Date: 23 March 2021
#This program performs a genetic algorithm on a number of hikers.
#Constants are shown below that modify perameters of the algorithm.
#Otherwise the program is run in command line.

import random
import numpy as np
from copy import copy



#Maximum backpack weight
MAX_WEIGHT = 250

#Starting population quantity
START_POP = 200

#Chance of simply mutating (out of 100)
MUTATION_PERCENTAGE = 10

#Chance of multi-point mutation if already mutating (out of 100)
MULTI_MUTATION_PERCENTAGE = 50

#Chance of multi point crossover (out of 100)
MULTI_CROSSOVER_PERCENTAGE = 20

#Percentage of population removed each iteration (out of 100)
CULL_PERCENTAGE = 50


#hiker class: initialized with a starting backpack and boxes (a dictionary
#of box weights and values). Other functions are defined for the 
#population set used later, so collisions can be resolved in hashing.
class hiker:

    #create hiker with backpack. references boxes values/weights
    def __init__(self, backpack, boxes):
        self.backpack = backpack

        #evaluate fitness using fitness function and weights
        self.fitness = self.__fitness_function(boxes)

    #hash function for later set converts backpack to a decimal representation
    #of the binary number its backpack represents
    def __hash__(self):
        return (int(''.join(map(str, self.backpack)), 2) << 1)

    #equivalent function returns true if the two hikers backpacks are the same
    def __eq__(self, other):
        if self.backpack == other.backpack:
            return True
        return False

    #fitness function evacuates how fit the hiker is using its backpack and
    #the designated weights in boxes. If the weight exceeds the MAX_WEIGHT,
    #the hikers fitness is 0
    def __fitness_function(self, boxes):
        total_value = 0
        total_weight = 0
        for i in range(len(self.backpack)):
            if self.backpack[i] == 1:
                total_weight = total_weight + boxes[i][0]
                total_value = total_value + boxes[i][1]
        if total_weight > MAX_WEIGHT:
            total_value = 0
        return total_value
        

#make_initial_pop accepts the weights/values of the boxes, and creates
#an initial population set of hikers with random backpacks. It creates
#as many hikers as START_POP.
def make_initial_pop(boxes):
    initial_pop = set()
    while len(initial_pop) < START_POP:
        backpack = [random.randint(0,1) for i in range(len(boxes))]
        individual = hiker(backpack, boxes)
        initial_pop.add(individual)
    return initial_pop


#genetic_algorithm accepts the weights/values of the boxes and a number of
#iterations to perform of the algorithm. 
def genetic_algorithm(boxes, iterations):

    population = make_initial_pop(boxes)
    
    for i in range(iterations):

        child_pop = set()
        for j in population:

            individuals = random.sample(population, 2)

            ind1 = individuals[0]
            ind2 = individuals[1]

            if np.random.binomial(1, MULTI_CROSSOVER_PERCENTAGE/100, 1)[0]:
                child = hiker(multi_point_crossover(ind1.backpack.copy(), ind2.backpack.copy()), boxes)
            else:
                child = hiker(point_crossover(ind1.backpack.copy(), ind2.backpack.copy()), boxes)

            if np.random.binomial(1, MUTATION_PERCENTAGE/100, 1)[0]:

                if np.random.binomial(1, MULTI_MUTATION_PERCENTAGE/100, 1)[0]:
                    multi_point_mutation(child.backpack)
                else:
                    point_mutation(child.backpack)
            
            child_pop.add(child)

        population.union(child_pop)
        
        sorted_set = sorted(population, key=lambda x: x.fitness)
        cull_index = int((len(sorted_set) - 1) * (CULL_PERCENTAGE / 100))
        
        sorted_set = sorted_set[cull_index:]
        population = set(sorted_set)

    best_ind = hiker([random.randint(0,1) for i in range(len(boxes))], boxes)
    for ind in population:
        if ind.fitness >= best_ind.fitness:
            best_ind = ind

    print(best_ind.fitness)
    print(best_ind.backpack)



#point_mutation accepts a backpack and changes the value at a random index
#from 0 to 1 or 1 to 0
def point_mutation(backpack):
    index = random.randint(0, len(backpack)-1)
    backpack[index] = 1 - backpack[index]

#multi_point_mutation accepts a backpack and changes multiple values at
#multiple random indices from 0 to 1 or 1 to 0
def multi_point_mutation(backpack):
    indices = random.sample(range(len(backpack)), random.randint(2,len(backpack)))
    for index in indices:
        backpack[index] = 1 - backpack[index]


#point_crossover accepts 2 backpacks and swaps the values in the two
#backpacks after a random index. It returns 1 of the backpacks at random.
def point_crossover(backpack1, backpack2):
    index = random.randint(1, len(backpack1)-1)
    for i in range(index, len(backpack1)):
        backpack1[i], backpack2[i] = backpack2[i], backpack1[i]

    return random.choice([backpack1, backpack2])


#multi_point_crossover accepts 2 backpacks and swaps the values in the two 
#backpacks after multiple random indices. It does this by checking the
#amount of indices, and only swapping if on even indices. It returns
#one of the backpacks at random.
def multi_point_crossover(backpack1, backpack2):
    indices = random.sample(range(len(backpack1)), random.randint(2,len(backpack1)))
    indices.sort()

    if len(indices) % 2 == 1:
        indices.append(len(backpack1)-1)

    for i in range(len(indices)):
        if i != len(backpack1) - 1:
            if i % 2 == 0:
                for j in range(indices[i],indices[i+1]):
                    backpack1[j], backpack2[j] = backpack2[j], backpack1[j]

    return random.choice([backpack1, backpack2])
    

#main function combines the box weights and values into boxes,
#and passes the iterations into the genetic_algorithm
def main(weights, values, iterations):

    #Index[box i] = [weight, value]
    boxes = []
    for i in range(len(weights)):
        boxes.append([weights[i],values[i]])
    genetic_algorithm(boxes, iterations)




if __name__ == "__main__":
    # weights = [int(i) for i in input("What are the boxes weights('e.g. 1 2 3 10 15' for 5 entered)?: ").rstrip().split(" ")]
    # values = [int(i) for i in input("What are the boxes values('e.g. 1 2 3 4 5' for 5 entered)?: ").rstrip().split(" ")]
    # iterations = int(input("How many iterations?: "))

    weights = [1,2,3,4,5,10,20,230]
    values = [1,2,3,4,5,10,20,200]
    iterations = 20
    assert(len(values) == len(weights))

    main(weights, values, iterations)
