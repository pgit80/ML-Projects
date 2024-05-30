# import required modules
import random
import numpy
from matplotlib import pyplot as plt
import matplotlib
from deap import base, creator, tools, algorithms


# TODO: Task 1- Define the number of locations and vehicles
num_locations = 10 # Define the number of locations
locations = [(random.randint(0,100), random.randint(0,100)) for _ in range(num_locations)] # Create a list of tuples representing location coordinates-#try to use a random number generator


depot = (50, 50) # define the coordinated for the depot
num_vehicles = 3 #Define the number of vehicles

# Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)


# Task 2- Finish setting up the individuals and population
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_locations), num_locations)# function to generate a list of unique, randomly ordered location indices
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)# function to create an individual as a shuffled list of location indices
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # function to create a population of individuals


# Fitness function
def evalVRP(individual):
    # task 3- write the fitness evaluation function
    # calculate the total distance of routes and return it
    total_distance = 0
    distances = [] # track distance traveled by each vehicle for balance calculation
    #split the list of locations among vehicles, ensuring each starts and ends at the depot
    for i in range(num_vehicles):
        vehicle_route = [depot] + [locations[individual[j]] for j in range(i, len(individual), num_vehicles)]+[depot]
        # calculate total distance covered by this vehicle
        vehicle_distance = sum(numpy.linalg.norm(numpy.array(vehicle_route[k+1]) - numpy.array(vehicle_route[k])) for k in range(len(vehicle_route)-1))
        total_distance+=vehicle_distance
        distances.append(vehicle_distance)
    balance_penalty = numpy.std(distances)#use standard deviation of disances as a penalty for imbalance among vehicles
    return total_distance, balance_penalty

toolbox.register("evaluate", evalVRP) # VRP- vehicle routing problem
toolbox.register("mate", tools.cxPartialyMatched) # register the crossover function suitable for permutation-based representation
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05) # Register the mutation function to shuffle indices with a 5% chance per index
toolbox.register("select", tools.selTournament, tournsize=3) # Register the selection function using tournament selection


# Plotting function
def plot_routes(individual, title="Routes"):
    plt.figure()
    # Plotting locations as blue dot and depot as red dot
    for (x, y) in locations:
        plt.plot(x, y, 'bo')
    plt.plot(depot[0], depot[1], 'rs')

    # Draw routes for each vehicle
    for i in range(num_vehicles):
        vehicle_route = [depot] + [locations[individual[j]] for j in range(i, len(individual), num_vehicles)] + [depot]
        plt.plot(*zip(*vehicle_route), '-')

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# running the GA
def main():
    random.seed(42) # seed for reproducibility
    pop = toolbox.population(n=300) # generating the initial population
    hof = tools.HallOfFame(1) # hall of fame to store the best individual

    # setup statistics to track
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)

    # run the genetic algorithm
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 300, stats=stats, halloffame=hof)

    # plot the best found route
    plot_routes(hof[0], "Optimal Route")
    return pop, stats, hof

if __name__ == "__main__":
    main()