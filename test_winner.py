"""Test the final evolved network from the NEAT training checkpoints.
This script loads the best genome from neat-checkpoint-99 and runs it
on multiple game episodes to evaluate how well it learned.

Run with:
    python test_winner.py
"""
import os
import neat
import neatplat
import stats as sp

# Define MyReporter class (needed for unpickling checkpoint)
filename = 'NeatTextOutput.txt'
class MyReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        pass  # no-op

# Load the final checkpoint
checkpoint_path = 'neat-checkpoint-99'
p = neat.Checkpointer.restore_checkpoint(checkpoint_path)

# Get the configuration
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'configneatNew.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

print(f"Loaded checkpoint: {checkpoint_path}")
print(f"Population size: {len(p.population)}")

# Find the best genome from the final population (filter out None fitness)
valid_genomes = [(k, v) for k, v in p.population.items() if v.fitness is not None]
if not valid_genomes:
    # If no fitness values, just use the first genome
    best_id = list(p.population.keys())[0]
    best_genome = p.population[best_id]
    print("Warning: No fitness values in checkpoint, using first genome")
else:
    best_id, best_genome = max(valid_genomes, key=lambda x: x[1].fitness)

print(f"\nBest genome ID: {best_id}")
print(f"Best genome fitness: {best_genome.fitness if best_genome.fitness is not None else 'N/A'}")

# Create the network from the best genome
try:
    net = neat.nn.RecurrentNetwork.create(best_genome, config)
except Exception:
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

print(f"Network created (recurrent or feed-forward)")

# Run multiple test episodes
num_episodes = 5
wins = 0
total_fitness = 0.0

print(f"\n{'='*60}")
print(f"Running {num_episodes} test episodes...")
print(f"{'='*60}\n")

for ep in range(num_episodes):
    neatplat.reset()
    episode_fitness = 0.0
    steps = 0
    max_steps = 3500
    death = False
    won = False

    while not death and steps < max_steps:
        # Get inputs from neatplat
        fitness, enemyX, enemyY, X, Y, emenmydis, goaldis, Y_vel, on_ground, enemy_dx, death = neatplat.sim(False, False, False)

        # Prepare network inputs
        inputs = [
            enemyX / neatplat.SCwidth,
            enemyY / neatplat.SChight,
            X / neatplat.SCwidth,
            Y / neatplat.SChight,
            emenmydis / (neatplat.SCwidth * 1.5 + 1e-6),
            goaldis / (neatplat.SCwidth * 1.5 + 1e-6),
            Y_vel / (neatplat.SChight + 1e-6),
            on_ground,
            enemy_dx / (neatplat.SCwidth + 1e-6),
        ]

        # Get network output
        output = net.activate(inputs)
        decision = int(max(range(len(output)), key=lambda i: output[i]))

        # Map decision to actions
        action1 = (decision == 0)  # left
        action2 = (decision == 1)  # right
        action3 = (decision == 2)  # jump

        # Step simulation with network's actions
        fitness, enemyX, enemyY, X, Y, emenmydis, goaldis, Y_vel, on_ground, enemy_dx, death = neatplat.sim(action1, action2, action3)

        episode_fitness += fitness
        steps += 1

        # Check if won
        if fitness >= 600:
            won = True

    total_fitness += episode_fitness
    status = "WON!" if won else "LOST"
    print(f"Episode {ep+1}: Fitness={episode_fitness:.2f}, Steps={steps}, Status={status}")
    if won:
        wins += 1

print(f"\n{'='*60}")
print(f"Results Summary")
print(f"{'='*60}")
print(f"Wins: {wins}/{num_episodes} ({100*wins/num_episodes:.1f}%)")
print(f"Average fitness: {total_fitness/num_episodes:.2f}")
print(f"Best genome fitness from training: {best_genome.fitness:.2f}")
print(f"{'='*60}")
