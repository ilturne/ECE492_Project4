import random

from graphing import save_frames, plot_optimization_history, calculate_distance


def load_cities(filename):
    loaded_cities = []
    with open(filename, 'r') as file:
        for line in file:
            if len(loaded_cities) >= 80:
                break
            parts = line.split()
            loaded_cities.append((float(parts[1]), float(parts[2])))  # Assuming the format is ID X Y
    return loaded_cities


def create_initial_population(size, city_count):
    population = []
    for _ in range(size):
        individual = list(range(city_count))
        random.shuffle(individual)
        population.append(individual)
    return population


def select_parents(population, cities):
    distances = [calculate_distance(individual, cities) for individual in population]
    total_fitness = sum(1 / d for d in distances)
    probabilities = [(1 / d) / total_fitness for d in distances]
    parents = random.choices(population, probabilities, k=len(population))
    return parents


def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = parent1[:], parent2[:]
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        p1[parent1[i]] = i
        p2[parent2[i]] = i

    cxpoint1, cxpoint2 = sorted([random.randint(0, size) for _ in range(2)])

    for i in range(cxpoint1, cxpoint2):
        temp1 = child1[i]
        temp2 = child2[i]

        child1[i], child1[p1[temp2]] = temp2, temp1
        child2[i], child2[p2[temp1]] = temp1, temp2

        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return child1, child2


def mutate(individual):
    return [swap_mutation(individual[:]), insert_mutation(individual[:]), scramble_mutation(individual[:])]


def swap_mutation(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def insert_mutation(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    if idx1 < idx2:
        individual.insert(idx2, individual.pop(idx1))
    else:
        individual.insert(idx1, individual.pop(idx2))
    return individual


def scramble_mutation(individual):
    start, end = sorted(random.sample(range(len(individual)), 2))
    individual[start:end] = random.sample(individual[start:end], len(individual[start:end]))
    return individual


def genetic_algorithm(cities, population_size=80, generations=100):
    population = create_initial_population(population_size, len(cities))
    best_distance_gen = float('inf')
    best_route_gen = None
    history_gen = []

    for generation in range(generations):
        parents = select_parents(population, cities)
        next_population = []

        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = pmx_crossover(parent1, parent2)
            next_population.extend([child1, child2])

        next_population = [mut for child in next_population for mut in mutate(child)]  # Apply all mutations

        distances = [calculate_distance(individual, cities) for individual in next_population]
        min_distance = min(distances)
        min_distance_index = distances.index(min_distance)

        if min_distance < best_distance_gen:
            best_distance_gen = min_distance
            best_route_gen = next_population[min_distance_index]

        population = next_population
        history_gen.append(best_route_gen)

    # This is for debugging and more detailed view of generation and distance.
    print(f'Best Distance = {best_distance_gen}')
    return best_route_gen, history_gen


if __name__ == "__main__":
    cities = load_cities('uy734.dat')
    best_route, history = genetic_algorithm(cities)

    plot_optimization_history(history, cities)
    # You comment out this next line if you don't want to save any frames or generate the .gif
    # save_frames(cities, best_route)
