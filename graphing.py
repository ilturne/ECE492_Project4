import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def save_frames(cities, route, filename='optimization.gif'):
    if not os.path.exists('../Images'):
        os.makedirs('../Images')

    frames = []
    for i in range(len(route) + 1):
        plt.figure(figsize=(10, 6))
        if i > 0:
            x = [cities[j][0] for j in route[:i]]
            y = [cities[j][1] for j in route[:i]]
            plt.plot(x, y, 'o-', markersize=5)
        else:
            plt.plot([cities[route[0]][0]], [cities[route[0]][1]], 'o', markersize=5)
        plt.title(f'Visiting city {i}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'Images/frame_{i}.png')
        frames.append(imageio.imread(f'Images/frame_{i}.png'))
        plt.close()

    imageio.mimsave(f'Images/{filename}', frames, duration=0.5)


def plot_optimization_history(history, cities):
    plt.plot([calculate_distance(route, cities) for route in history])
    plt.title('Optimization History')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.show()


def calculate_distance(route, cities):
    distance = 0
    for i in range(len(route) - 1):
        city1 = cities[route[i]]
        city2 = cities[route[i + 1]]
        distance += np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
    return distance
