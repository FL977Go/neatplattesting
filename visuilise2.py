
import matplotlib.pyplot as plt
import warnings

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg', title='Processing Fitness'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = statistics.get_fitness_stat(lambda x: sum(x) / len(x))
    stdev_fitness = statistics.get_fitness_stat(lambda x: (sum((xi - (sum(x) / len(x))) ** 2 for xi in x) / len(x)) ** 0.5)

    plt.figure(figsize=(10, 6))  # Bigger figure
    plt.plot(generation, avg_fitness, 'b-', label="average")
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, [a + s for a, s in zip(avg_fitness, stdev_fitness)], 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()

def draw_net(config, genome, view=False, fmt='png', show_disabled=False, prune_unused=False, node_names=None):
    # Safe no-op/placeholder as before, keeping existing signatures
    try:
        pass 
        # print(f"visuilise2.draw_net called for genome id={genome.key}")
    except Exception:
        pass
