import pickle
from collections import defaultdict
import neat
import os
import multiprocessing as mp
import matplotlib.pyplot as plt


class Strategy:
    def __init__(self, ID=-1):
        self.fitness = 0
        self.ID = ID

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
    def __init__(self, genome, config, ID):
        super().__init__(ID)
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
            self.amount_both_cooperate_per_generation[generation] /= self.amount_of_games_per_generation[generation]
            self.amount_one_cooperate_per_generation[generation] /= self.amount_of_games_per_generation[generation]
            self.amount_both_defect_per_generation[generation] /= self.amount_of_games_per_generation[generation]

            self.amount_start_cooperate_per_generation[generation] /= self.amount_of_games_per_generation[generation]
            self.amount_start_defect_per_generation[generation] /= self.amount_of_games_per_generation[generation]

            self.amount_forgiveness_per_generation[generation] /= self.amount_of_games_per_generation[generation]


def play_game(genome1: Strategy, genome2: Strategy, num_rounds=20):
    score1, score2 = 0, 0
    play1_history = []
    play2_history = []
    game_stats = GameStats()

    # Simulate a series of rounds in the game
    for ind in range(num_rounds):
        # Decide the moves
        move1 = genome1.make_move(play1_history, play2_history)
        move2 = genome2.make_move(play2_history, play1_history)

        if ind == 0:
            if move1 == "C":
                game_stats.amount_start_cooperate += 1
            else:
                game_stats.amount_start_defect += 1

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

        if not len(play1_history) == 0 and play1_history[-1] == "D" and play2_history[-1] == "D":
            # calculate forgiveness, can only be done if there is a previous round
            if move1 == "C":
                game_stats.amount_forgiveness += 1
            if move2 == "C":
                game_stats.amount_forgiveness += 1

        play1_history.append(move1)
        play2_history.append(move2)

        # Update scores based on moves
        round_score1, round_score2 = REWARD_MATRIX[(move1, move2)]
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
        if not genome1_id == -1:
            genome_dict[genome1_id].fitness += score1
            genome_dict[genome1_id].amount_of_games += 1
        if not genome2_id == -1:
            genome_dict[genome2_id].fitness += score2
            genome_dict[genome2_id].amount_of_games += 1
        statistics.add_stats(generation, game_stats)

    # Calculate average fitness for each genome
    total_complexity = 0

    for genome_id, genome in genomes:
        genome.fitness = (genome.fitness / genome.amount_of_games)  # Average fitness squared
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


def plot_combined(both_cooperate, niceness, complexities, forgiveness):
    generations = list(both_cooperate.keys())

    # Create the first y-axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot percentage of both cooperate
    ax1.plot(generations, list(both_cooperate.values()), marker='o', linestyle='-', color='b',
             label='Both Cooperate (%)')
    ax1.set_title('Strategies evolution per Generation')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Percentage of Both Cooperate (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for percentage of niceness
    ax2 = ax1.twinx()
    ax2.plot(generations, list(niceness.values()), marker='o', linestyle='-', color='g',
             label='First Move Cooperation (%)')
    ax2.set_ylabel('Percentage of First Move Cooperation (%)', color='g')
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
    ax4.plot(generations, list(forgiveness.values()), marker='o', linestyle='-', color='y',
             label='Average Forgiveness per game')
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

    pool = mp.Pool(mp.cpu_count())
    winner = population.run(evaluate_genomes, 10)

    statistics.compute_averages()

    plot_combined(statistics.amount_both_cooperate_per_generation, statistics.amount_start_cooperate_per_generation,
                  statistics.average_complexity_per_generation, statistics.amount_forgiveness_per_generation)

    with open("WinnerGenome", 'wb') as f:
        # Save the genome and neural network state
        pickle.dump(winner, f)
