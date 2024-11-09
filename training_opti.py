import pickle
from collections import defaultdict
import neat
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd


class Strategy:
    def __init__(self, ID=-1):
        self.fitness = 0
        self.ID = ID

    def make_move(self, ownHistory, oppHistory):
        pass


class AlwaysDefect(Strategy):
    def make_move(self, ownHistory, oppHistory):
        return "D"


class Genome(Strategy):
    def __init__(self, genome, config, ID):
        super().__init__(ID)
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.config = config
        self.genome = genome

    def make_move(self, ownHistory, oppHistory):
        ownHistory, oppHistory = ownHistory[-20:], oppHistory[-20:]

        input_CD = [
            (1 if own == "C" else -1 if own == "D" else 0,
             1 if opp == "C" else -1 if opp == "D" else 0)
            for own, opp in zip(ownHistory, oppHistory)
        ]

        input_CD = [item for history in input_CD for item in history]
        input_CD.extend([0] * (40 - len(input_CD)))

        move = self.net.activate(input_CD)
        return "C" if move[0] < 0.5 else "D"


class GameStats:
    def __init__(self):
        self.amount_both_cooperate = 0
        self.amount_both_defect = 0
        self.amount_one_cooperate = 0

        self.amount_start_defect = 0
        self.amount_start_cooperate = 0

        self.amount_forgiveness = 0


class PopulationStats:
    def __init__(self):
        self.average_complexity_per_generation = defaultdict(int)

        self.amount_both_cooperate_per_generation = defaultdict(int)
        self.amount_one_cooperate_per_generation = defaultdict(int)
        self.amount_both_defect_per_generation = defaultdict(int)

        self.amount_start_cooperate_per_generation = defaultdict(int)
        self.amount_start_defect_per_generation = defaultdict(int)

        self.amount_forgiveness_per_generation = defaultdict(int)

        self.amount_of_games_per_generation = defaultdict(int)

    def add_stats(self, generation, stats: GameStats):
        self.amount_both_cooperate_per_generation[generation] += stats.amount_both_cooperate
        self.amount_one_cooperate_per_generation[generation] += stats.amount_one_cooperate
        self.amount_both_defect_per_generation[generation] += stats.amount_both_defect

        self.amount_start_cooperate_per_generation[generation] += stats.amount_start_cooperate
        self.amount_start_defect_per_generation[generation] += stats.amount_start_defect

        self.amount_forgiveness_per_generation[generation] += stats.amount_forgiveness

        self.amount_of_games_per_generation[generation] += 1

    def add_average_complexity(self, generation, complexity):
        self.average_complexity_per_generation[generation] = complexity

    def compute_averages(self):
        for generation in self.amount_of_games_per_generation.keys():
            self.amount_both_cooperate_per_generation[generation] /= (
                        (self.amount_of_games_per_generation[generation] * num_rounds) / 100)
            self.amount_one_cooperate_per_generation[generation] /= self.amount_of_games_per_generation[generation]
            self.amount_both_defect_per_generation[generation] /= self.amount_of_games_per_generation[generation]

            self.amount_start_cooperate_per_generation[generation] /= (
                        self.amount_of_games_per_generation[generation] / 100)
            self.amount_start_defect_per_generation[generation] /= self.amount_of_games_per_generation[generation]

            self.amount_forgiveness_per_generation[generation] /= self.amount_of_games_per_generation[generation]


def play_game(genome1: Strategy, genome2: Strategy):
    score1, score2 = 0, 0
    play1_history, play2_history = [], []
    game_stats = GameStats()
    reward_matrix = REWARD_MATRIX

    # round 1
    move1 = genome1.make_move(play1_history, play2_history)
    move2 = genome2.make_move(play2_history, play1_history)
    play1_history.append(move1)
    play2_history.append(move2)

    # start stats (measured for round 1 only)
    if move1 == "C":
        game_stats.amount_start_cooperate += 1
    else:
        game_stats.amount_start_defect += 1

    # stats measured for all rounds, including round 1
    if move2 == "C":
        game_stats.amount_start_cooperate += 1
    else:
        game_stats.amount_start_defect += 1

    if move1 == "C" and move2 == "C":
        game_stats.amount_both_cooperate += 1
    elif move1 == "D" and move2 == "D":
        game_stats.amount_both_defect += 1
    else:
        game_stats.amount_one_cooperate += 1

    round_score1, round_score2 = reward_matrix[(move1, move2)]
    score1 += round_score1
    score2 += round_score2

    # for loop excluding round 1 (evaluated separately, see above)
    for ind in range(1, num_rounds):
        move1 = genome1.make_move(play1_history, play2_history)
        move2 = genome2.make_move(play2_history, play1_history)
        play1_history.append(move1)
        play2_history.append(move2)

        if move1 == "C" and move2 == "C":
            game_stats.amount_both_cooperate += 1
        elif move1 == "D" and move2 == "D":
            game_stats.amount_both_defect += 1
        else:
            game_stats.amount_one_cooperate += 1

        if play1_history[ind - 1] == "D" and play2_history[ind - 1] == "D":
            # calculate forgiveness, can only be done if there is a previous round
            if move1 == "C":
                game_stats.amount_forgiveness += 1
            if move2 == "C":
                game_stats.amount_forgiveness += 1

        round_score1, round_score2 = reward_matrix[(move1, move2)]
        score1 += round_score1
        score2 += round_score2

    return genome1.ID, score1, genome2.ID, score2, game_stats


generation = 0
# Define the reward system for the Prisoner's Dilemma
REWARD_MATRIX = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1)
}
num_rounds = 20
subset_size = 20
statistics = PopulationStats()


def evaluate_genomes(genomes, config):
    global generation
    genome_dict = dict(genomes)

    # Initialize fitness for all genomes
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome.amount_of_games = 0

    # Create a list of all genome pairs to be evaluated
    genome_pairs = create_genome_pairs(genomes)

    async_results = [pool.apply_async(play_game, (g1, g2)) for g1, g2 in genome_pairs]
    for result in async_results:
        genome1_id, score1, genome2_id, score2, game_stats = result.get()

        if genome1_id != -1:
            genome_dict[genome1_id].fitness += score1
            genome_dict[genome1_id].amount_of_games += 1

        if genome2_id != -1:
            genome_dict[genome2_id].fitness += score2
            genome_dict[genome2_id].amount_of_games += 1
        statistics.add_stats(generation, game_stats)

    total_complexity = 0

    for genome_id, genome in genomes:
        genome.fitness /= genome.amount_of_games  # Average fitness squared
        total_complexity += genome.size()[0] * genome.size()[1]
    statistics.add_average_complexity(generation, total_complexity / len(genomes))

    generation += 1


def create_genome_pairs(genomes):
    # Create a list of all genome pairs to be evaluated
    genome_pairs = []
    for idx, (genome_id1, genome1) in enumerate(genomes):
        # opponents = random.sample(genomes, subset_size)
        for genome_id2, genome2 in genomes[idx + 1:]:
            genome_pairs.append((
                Genome(genome1, config, genome_id1), Genome(genome2, config, genome_id2)
            ))
    return genome_pairs


def create_genome_pairs_predefined_strategies(genomes):
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


def plot_combined(both_cooperate, niceness, complexities, forgiveness, window):
    generations = list(both_cooperate.keys())

    # Convert values to pandas Series to calculate the rolling mean
    both_cooperate_series = pd.Series(list(both_cooperate.values())).rolling(window=window).mean()
    niceness_series = pd.Series(list(niceness.values())).rolling(window=window).mean()
    complexities_series = pd.Series(list(complexities.values())).rolling(window=window).mean()
    forgiveness_series = pd.Series(list(forgiveness.values())).rolling(window=window).mean()

    # Create the first y-axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot percentage of both cooperate
    ax1.plot(generations, list(both_cooperate_series), linestyle='-', color='b',
             label='Both Cooperate (%)')
    ax1.set_title('Strategies evolution per Generation', fontsize=24)
    ax1.set_xlabel('Generation', fontsize=20)
    ax1.set_ylabel('Percentage of Both Cooperate Moves Per Game(%)', color='b', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=16)
    ax1.set_ylim(0, 100)

    # Create a second y-axis for percentage of niceness
    ax2 = ax1.twinx()
    ax2.plot(generations, list(niceness_series), linestyle='-', color='g',
             label='First Move Cooperation (%)')
    ax2.set_ylabel('Percentage of First Move Cooperation (%)', color='g', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='g', labelsize=16)
    ax2.set_ylim(0, 200)

    # Create a third y-axis for average complexity
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 120))  # Move the third y-axis outward
    ax3.plot(generations, list(complexities_series), linestyle='-', color='r', label='Average Complexity')
    ax3.set_ylabel('Average Complexity', color='r', fontsize=20)
    ax3.tick_params(axis='y', labelcolor='r', labelsize=16)

    # Create a fourth y-axis for average forgiveness
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 240))  # Move the fourth y-axis further outward
    ax4.plot(generations, list(forgiveness_series), linestyle='-', color='y',
             label='Average Forgiveness moves per game')
    ax4.set_ylabel('Average Forgiveness moves per game', color='y', fontsize=20)
    ax4.tick_params(axis='y', labelcolor='y', labelsize=16)

    # Set x-ticks
    ax1.set_xticks([gen for gen in generations if gen % 200 == 0])
    ax1.tick_params(axis='x', labelsize=16)

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

    # Define the folder for checkpoints
    checkpoint_folder = "checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Initialize the Checkpointer with your desired settings
    checkpoint_interval = 10  # Save every 10 generations (adjust as needed)
    checkpoint_prefix = os.path.join(checkpoint_folder, "neat-checkpoint-")

    population.add_reporter(neat.Checkpointer(generation_interval=500, time_interval_seconds=None, filename_prefix=checkpoint_folder))
    pool = mp.Pool(mp.cpu_count())

    num_generations = 5000

    winner = population.run(evaluate_genomes, num_generations)

    statistics.compute_averages()

    plot_combined(statistics.amount_both_cooperate_per_generation, statistics.amount_start_cooperate_per_generation,
                  statistics.average_complexity_per_generation, statistics.amount_forgiveness_per_generation,
                  min(1, round(num_generations / 20)))

    with open("WinnerGenome", 'wb') as f:
        # Save the genome and neural network state
        pickle.dump(winner, f)
