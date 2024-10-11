import math
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


class CooperateUntilThreeDefections(Strategy):
    def make_move(self, ownHistory, oppHistory):
        # Cooperate if no previous history
        if len(oppHistory) < 3:
            return "C"

        # Check if the opponent has defected three times in a row at any point
        for i in range(len(oppHistory) - 2):
            if oppHistory[i:i + 3] == ["D", "D", "D"]:
                return "D"

        # Otherwise, continue cooperating
        return "C"


class Genome(Strategy):
    def __init__(self, genome, config):
        super().__init__()
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.config = config
        self.genome = genome

    def make_move(self, ownHistory, oppHistory):
        input = [0] * 40

        for idx, (opp, own) in enumerate(zip(oppHistory, ownHistory)):
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

        self.amount_forgiveness = 0

        self.history1 = []
        self.history2 = []

        self.complexity1 = 0
        self.complexity2 = 0


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

        if not len(play1_history) == 0 and play1_history[-1] == "D" and play2_history[-1] == "D":
            # calculate forgiveness, can only be done if there is a previous round
            if move1 == "C":
                gamestats.amount_forgiveness += 1
            if move2 == "C":
                gamestats.amount_forgiveness += 1

        play1_history.append(move1)
        play2_history.append(move2)

        # Update scores based on moves
        round_score1, round_score2 = REWARD_MATRIX[(move1, move2)]
        score1 += round_score1
        score2 += round_score2

        gamestats.history1 = play1_history
        gamestats.history2 = play2_history

        # measure genetic complexity
        gamestats.complexity1 = genome1.genome.size()
        if isinstance(genome2, Genome):
            gamestats.complexity2 = genome2.genome.size()

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
    genome_pairs = create_genome_pairs_predefined(genomes)
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
    # Create a list of all genome pairs randomly, with a subset of the population
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
        for genome_id2, genome2 in genomes[idx + 1:]:
            genome_pairs.append((
                Genome(genome1, config), Genome(genome2, config)
            ))
    return genome_pairs


def create_genome_pairs_predefined(genomes):
    genome_pairs = []
    for idx, (genome_id1, genome1) in enumerate(genomes):
        # opponents = random.sample(genomes, subset_size)
        for genome_id2, genome2 in genomes[idx + 1:]:
            genome_pairs.append((
                Genome(genome1, config), Genome(genome2, config)
            ))
        # for i in range(30):
        #     genome_pairs.append((
        #         Genome(genome1, config), CooperateUntilThreeDefections()))
        for i in range(30):
            genome_pairs.append((
                Genome(genome1, config), AlwaysDefect()))

    return genome_pairs


def calculate_percentage_stats_per_five_generations(gamestats_per_generation):
    average_cooperate = {}
    average_first_cooperate = {}
    average_complexity = {}
    average_forgiveness = {}

    # Calculate for every 5 rounds
    for i in range(math.floor(len(gamestats_per_generation.items()) / 5)):
        total_both_cooperate = 0
        total_both_defect = 0
        total_one_cooperate = 0

        total_start_cooperate = 0
        total_start_defect = 0

        total_complexity = 0
        total_genomes = 0

        total_forgiveness = 0
        total_games = 0

        for stats_list in list(gamestats_per_generation.values())[i:i + 5]:
            total_both_cooperate += sum(stat.amount_both_cooperate for stat in stats_list)
            total_both_defect += sum(stat.amount_both_defect for stat in stats_list)
            total_one_cooperate += sum(stat.amount_one_cooperate for stat in stats_list)

            total_start_cooperate += sum(stat.amount_start_cooperate for stat in stats_list)
            total_start_defect += sum(stat.amount_start_defect for stat in stats_list)

            total_forgiveness += sum(stat.amount_forgiveness for stat in stats_list)
            total_games += len(stats_list)

            for stat in stats_list:
                total_complexity += stat.complexity1[0] * stat.complexity1[1]
                total_genomes += 1
                if stat.complexity2 != 0:
                    total_complexity += stat.complexity2[0] * stat.complexity2[1]
                    total_genomes += 1

        total_moves = total_both_cooperate + total_both_defect + total_one_cooperate
        total_first_moves = total_start_cooperate + total_start_defect

        average_forgiveness[i * 5] = (total_forgiveness / total_games)
        average_cooperate[i * 5] = (total_both_cooperate / total_moves) * 100
        average_first_cooperate[i * 5] = (total_start_cooperate / total_first_moves) * 100
        average_complexity[i * 5] = total_complexity / total_genomes

    return average_cooperate, average_first_cooperate, average_complexity, average_forgiveness


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

def plot_complexity(percentage):
    generations = list(percentage.keys())
    percentage_values = list(percentage.values())

    plt.figure(figsize=(10, 5))
    plt.plot(generations, percentage_values, marker='o', linestyle='-', color='r')
    plt.title('Average Complexity per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Complexity')
    plt.xticks(generations)
    plt.grid()
    plt.show()


def plot_combined(percentages_both_cooperate, percentages_niceness, complexities, average_forgiveness):
    generations = list(percentages_both_cooperate.keys())

    # Create the first y-axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot percentage of both cooperate
    ax1.plot(generations, list(percentages_both_cooperate.values()), marker='o', linestyle='-', color='b',
             label='Both Cooperate (%)')
    ax1.set_title('Strategies evolution per Generation')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Percentage of Both Cooperate (%)', color='b')
    ax1.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for percentage of niceness
    ax2 = ax1.twinx()
    ax2.plot(generations, list(percentages_niceness.values()), marker='o', linestyle='-', color='g',
             label='First Move Cooperation (%)')
    ax2.set_ylabel('Percentage of First Move Cooperation (%)', color='g')
    ax2.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    ax2.tick_params(axis='y', labelcolor='g')

    # Create a third y-axis for average complexity
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward
    ax3.plot(generations, list(complexities.values()), marker='o', linestyle='-', color='r', label='Average Complexity')
    ax3.set_ylabel('Average Complexity', color='r')
    ax3.tick_params(axis='y', labelcolor='r')

    # Create a fourth y-axis for average forgiveness
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))  # Move the fourth y-axis further outward
    ax4.plot(generations, list(average_forgiveness.values()), marker='o', linestyle='-', color='y', label='Average Forgiveness per game')
    ax4.set_ylabel('Average Forgiveness per game', color='y')
    ax4.tick_params(axis='y', labelcolor='y')

    # Set x-ticks
    ax1.set_xticks([gen for gen in generations if gen % 100 == 0])

    # Adding a grid
    ax1.grid()

    # Show the plot
    plt.show()

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

    average_cooperate, average_first_cooperate, average_complexity, average_forgiveness = calculate_percentage_stats_per_five_generations(
        gamestats_per_generation)

    plot_combined(average_cooperate, average_first_cooperate, average_complexity, average_forgiveness)

    # plot_percentage_niceness(average_first_cooperate)
    # plot_percentage_both_cooperate(average_cooperate)
    # plot_complexity(average_complexity)

    with open("WinnerGenome", 'wb') as f:
        # Save the genome and neural network state
        pickle.dump(winner, f)
