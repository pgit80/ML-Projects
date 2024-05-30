# install and import necessary libraries
import random
import numpy
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms # deap is a library which allows us to implement a wide range of GAs


# we have a maze a start point and a end point
# the maze is defined as a matrix of 0s and 1s

# define the maze
maze=[
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
]

# start and end points
start, end = (0,0), (len(maze)-1, len(maze[0])-1)

# here as the maze is not changing so every solution will be a series of instruction like- up, down, left or right

# so an example solution can be like a large list of following commands:- [d,l,l,r,u,r,l,d,l,l,r,r,d,d,u,r]

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
# above ^ are two classes:
# class FitnessMin is inherited from the base.Fitness class and with `weights` parameter
# class Individual is type of list and with a `Fitness` attribute `FitnessMin`

# now we will create a function toolbox and toolbox is box of functions
# register is used to add functions in the toolbox

# now further steps are ------------------------->
# while(condition):
#     a. evaluate the solutions using fitness or evaluation function
#     b. Select- parents who has good genes- solutions that did well
#     c. Crossover- combining genes or good solutions to create new solutions or individuals
#     d. Mutation- making slight changes to the resulting offspring soluion to add randomness and explore 
#     e. Back to step(a) with the new population of these offsprings.

# a. Evaluation function:->
def evaluate(individual):
    x,y = start
    steps=0
    for move in individual:
        steps+=1
        # Move up, down, left, right while checking boundaries
        if move=='U':
            y=max(0, y-1)
        elif move== 'D':
            y=min(len(maze)-1, y+1)
        elif move=='L':
            x=max(0,x-1)
        elif move=='R':
            x=min(len(maze[0])-1, x+1)

        # check if the current position is the end goal
        if(x,y)==end:
            return(steps,) # perfect score, we reached the end
        
        # check if the current position is a wall
        if maze[y][x]==1:
            break
    # didn't reach the goal, return 100 plus the Manhattan distance to the end
    return(100 + abs(end[0]-x) + abs(end[1]-y), )

# mutation function->
def custom_mutate(individual, indpb=0.2):
    directions = ['U', 'D', 'L', 'R']
    for i in range(len(individual)):
        if random.random()<indpb:
            # exclude the current direction to ensure mutation changes the gene
            possible_directions = [d for d in directions if d != individual[i]]
            individual[i] =  random.choice(possible_directions)
    return individual,


# function to visualize the maze and the path
def plot_path(individual):
    x,y = start
    plt.plot(x, y, "go") # start point
    for move in individual:
        if move == 'U':
            y=max(0,y-1)
        elif move == 'D':
            y=min(len(maze)-1, y+1)
        elif move=='L':
            x=max(0, x-1)
        elif move=='R':
            x=min(len(maze[0])-1, x+1)
        plt.plot(x,y,"bo")
        if maze[y][x]==1:   break
    plt.plot(end[0], end[1], "ro")
    plt.imshow(maze, cmap="binary")
    plt.show()

# main function
def run_ga(generations=2000, pop_size=50):
    pop = toolbox.population(n=pop_size)
    best_individuals = []
    for gen in range(generations):
        offspring=algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits=toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values=fit
        pop = toolbox.select(offspring, k=len(pop))
        top_individual=tools.selBest(pop, k=1)[0]
        best_individuals.append(top_individual)

        if gen in [2,10,50,100,500] or gen == generations-1:
            print(f"Generation {gen}:")
            plot_path(top_individual)


toolbox=base.Toolbox()
toolbox.register("attr_direction", random.choice, ['U', 'D', 'L', 'R'])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_direction, n=100) # this will return a random choice of 100 values from the   
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # this will give us a random population
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniform, indpb=0.5)# uniform crossover
toolbox.register("mutate", custom_mutate, indpb=0.2)

run_ga()
