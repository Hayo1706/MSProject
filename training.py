import pickle
import random
import neat
import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

class Strategy:
    def __init__(self):
        self.fitness = 0

    def make_move(self, ownHistory, oppHistory):
        pass


class TitForTat(Strategy):
    def make_move(self, ownHistory, oppHistory):
        if len(oppHistory) == 0:
            return "C"
        return oppHistory[-1]


class AlwaysDefect(Strategy):

    def make_move(self, ownHistory, oppHistory):
        return "D"


class Genome(Strategy):
    def __init__(self, genome, config):
        super().__init__()
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.config = config
        self.genome = genome

    def make_move(self, ownHistory, oppHistory):
        input = [0] * 40

        for idx, (opp,own) in enumerate(zip(oppHistory, ownHistory)):
            if own == "C":
                input[idx * 2] = 1
            else:
                input[idx * 2] = -1

            if opp == "C":
                input[idx * 2 + 1] = 1
            else:
                input[idx * 2 + 1] = -1

        move = self.net.activate(input)
        return "C" if move[0] < 0.5 else "D"


class GameStats:
    def __init__(self):
        self.total_score = 0
        self.average_score = 0
        self.best_score = 0

        self.amount_both_cooperate = 0
        self.amount_both_defect = 0
        self.amount_one_cooperate = 0

        self.amount_start_defect = 0
        self.amount_start_cooperate = 0

        self.history1 = []
        self.history2 = []


# Define the reward system for the Prisoner's Dilemma
REWARD_MATRIX = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1)
}

subset_size = 20
gamestats_per_generation = {}


def play_game(genome1: Strategy, genome2: Strategy, num_rounds=20):
    score1, score2 = 0, 0
    play1_history = []
    play2_history = []
    gamestats = GameStats()

    # Simulate a series of rounds in the game
    for ind in range(num_rounds):
        # Decide the moves
        move1 = genome1.make_move(play1_history, play2_history)
        move2 = genome2.make_move(play2_history, play1_history)

        if ind == 0:
            if move1 == "C":
                gamestats.amount_start_cooperate += 1
            else:
                gamestats.amount_start_defect += 1

            if move2 == "C":
                gamestats.amount_start_cooperate += 1
            else:
                gamestats.amount_start_defect += 1

        if move1 == "C" and move2 == "C":
            gamestats.amount_both_cooperate += 1
        elif move1 == "D" and move2 == "D":
            gamestats.amount_both_defect += 1
        else:
            gamestats.amount_one_cooperate += 1

        play1_history.append(move1)
        play2_history.append(move2)

        # Update scores based on moves
        round_score1, round_score2 = REWARD_MATRIX[(move1, move2)]
        score1 += round_score1
        score2 += round_score2

        gamestats.history1 = play1_history
        gamestats.history2 = play2_history

    return score1, score2, gamestats


generation = 0


def evaluate_genomes(genomes, config):
    global generation
    global gamestats_per_generation

    print("Evaluating genomes")
    # Initialize fitness for all genomes
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome.amount_of_games = 0

    # Create a list of all genome pairs to be evaluated
    genome_pairs = create_genome_pairs(genomes)
    results = []
    with mp.Pool() as pool:
        results = pool.starmap(play_game, genome_pairs)
    # for genome1, genome2 in genome_pairs:
    #     results.append(play_game(genome1, genome2))


    # Update the fitness of each genome based on the results
    for (genome1, genome2), (score1, score2, gamestats) in zip(genome_pairs, results):
        genome1.genome.fitness += score1
        genome1.genome.amount_of_games += 1

        if generation not in gamestats_per_generation:
            gamestats_per_generation[generation] = []

        gamestats_per_generation[generation].append(gamestats)

        if isinstance(genome2, Genome):
            genome2.genome.fitness += score2
            genome2.genome.amount_of_games += 1

    # Calculate average fitness for each genome
    for genome_id, genome in genomes:
        genome.fitness = (genome.fitness / genome.amount_of_games)  # Average fitness squared

    generation += 1


def create_genome_pairs_random(genomes, subset_size):
    # Create a list of all genome pairs to be evaluated
    genome_pairs = []
    for (genome_id1, genome1) in genomes:
        opponents = random.sample(genomes, subset_size)
        for genome_id2, genome2 in opponents:
            genome_pairs.append((
                Genome(genome1, config), Genome(genome2, config)
            ))
    return genome_pairs

def create_genome_pairs(genomes):
    # Create a list of all genome pairs to be evaluated
    genome_pairs = []
    for idx, (genome_id1, genome1) in enumerate(genomes):
        # opponents = random.sample(genomes, subset_size)
        for genome_id2, genome2 in genomes[idx+1:]:
            genome_pairs.append((
                Genome(genome1, config), Genome(genome2, config)
            ))
    return genome_pairs


def create_genome_pairs_tft(genomes, subset_size):
    genome_pairs = []
    for genome_id1, genome1 in genomes:
        genome_pairs.append((
            Genome(genome1, config), TitForTat()))
        genome_pairs.append((
            Genome(genome1, config), AlwaysDefect()))
        genome1.amount_of_games += 2

    return genome_pairs


def calculate_percentage_both_cooperate(gamestats_per_generation):
    percentages = {}

    for generation, stats_list in gamestats_per_generation.items():
        total_both_cooperate = sum(stat.amount_both_cooperate for stat in stats_list)
        total_both_defect = sum(stat.amount_both_defect for stat in stats_list)
        total_one_cooperate = sum(stat.amount_one_cooperate for stat in stats_list)

        # Calculate total moves
        total_moves = total_both_cooperate + total_both_defect + total_one_cooperate

        # Avoid division by zero
        if total_moves > 0:
            percentage = (total_both_cooperate / total_moves) * 100
        else:
            percentage = 0

        percentages[generation] = percentage

    return percentages


def plot_percentage_both_cooperate(percentages):
    generations = list(percentages.keys())
    percentage_values = list(percentages.values())

    plt.figure(figsize=(10, 5))
    plt.plot(generations, percentage_values, marker='o', linestyle='-', color='b')
    plt.title('Percentage of Both Cooperate per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Percentage of Both Cooperate (%)')
    plt.xticks(generations)
    plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
    plt.grid()
    plt.show()


def calculate_percentage_niceness(gamestats_per_generation):
    percentages = {}

    for generation, stats_list in gamestats_per_generation.items():
        total_start_cooperate = sum(stat.amount_start_cooperate for stat in stats_list)
        total_start_defect = sum(stat.amount_start_defect for stat in stats_list)

        # Calculate total first moves
        total_first_moves = total_start_cooperate + total_start_defect

        # Avoid division by zero
        if total_first_moves > 0:
            percentage = (total_start_cooperate / total_first_moves) * 100
        else:
            percentage = 0

        percentages[generation] = percentage

    return percentages


def plot_percentage_niceness(percentages):
    generations = list(percentages.keys())
    percentage_values = list(percentages.values())

    plt.figure(figsize=(10, 5))
    plt.plot(generations, percentage_values, marker='o', linestyle='-', color='g')
    plt.title('Percentage of Niceness (First Move Cooperation) per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Percentage of First Move Cooperation (%)')
    plt.xticks(generations)
    plt.ylim(0, 100)  # Set y-axis limits from 0 to 100
    plt.grid()
    plt.show()

# Calculate percentages and plot


if __name__ == '__main__':
    # Load the NEAT config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-rnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(evaluate_genomes, 1000)

    percentages = calculate_percentage_both_cooperate(gamestats_per_generation)
    plot_percentage_both_cooperate(percentages)

    percentages = calculate_percentage_niceness(gamestats_per_generation)
    plot_percentage_niceness(percentages)

    with open("WinnerGenome", 'wb') as f:
        # Save the genome and neural network state
        pickle.dump(winner, f)
