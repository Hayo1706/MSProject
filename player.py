import pickle
import random
import neat
import os
import multiprocessing

import numpy as np


class Player:
    def __init__(self, name):
        self.name = name
        self.ownHistory = []
        self.opponentHistory = []
        self.ownScore = 0
        self.opponentScore = 0

    def make_move(self):
        pass

    def update_history(self, own_move, opponent_move, own_score, opponent_score):
        self.ownScore += own_score
        self.opponentScore += opponent_score
        self.ownHistory.append(own_move)
        self.opponentHistory.append(opponent_move)

    def reset_history(self):
        self.opponentHistory = []
        self.ownHistory = []
        self.ownScore = 0
        self.opponentScore = 0


class AlwaysCooperate(Player):
    def __init__(self):
        super().__init__("Always Cooperate")

    def make_move(self):
        return "C"


class AlwaysDefect(Player):
    def __init__(self):
        super().__init__("Always Defect")

    def make_move(self):
        return "D"


class TitForTat(Player):
    def __init__(self):
        super().__init__("Tit for Tat")

    def make_move(self):
        if not self.opponentHistory:
            return "C"
        return self.opponentHistory[-1]


class RandomPlayer(Player):
    def __init__(self):
        super().__init__("Random")

    def make_move(self):
        return random.choice(["C", "D"])


class GrimTrigger(Player):
    def __init__(self):
        super().__init__("Grim Trigger")

    def make_move(self):
        if "D" in self.opponentHistory:
            return "D"
        return "C"


class HumanPlayer(Player):
    def __init__(self, queue):
        super().__init__("Human Player")
        self.queue = queue

    def make_move(self):
        return self.queue.get()


class LearningPlayer(Player):
    # Create the population
    population = None
    genomes = None
    scores = {}
    last_winner = None
    simulation_round = 0
    instances = 0

    # Load the NEAT config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-rnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    @classmethod
    def initialize_population(cls):
        if cls.population is None:
            cls.population = neat.Population(cls.config)
            cls.population.run(cls.evaluate_genomes, 1)
            # Return the amount of genomes in the population
            cls.instances = len(cls.genomes)

    def __init__(self, index):
        super().__init__("Learning Player")
        self.genome_index, self.current_genome = self.genomes[index]
        self.net = neat.nn.RecurrentNetwork.create(self.current_genome, self.config)

    def make_move(self):
        # Use the last two moves as input for the RNN (modify if necessary)
        input = create_input(self.opponentHistory, self.ownHistory, self.ownScore, self.opponentScore)

        # Get the output from the neural network
        output = self.net.activate(input)
        # Decide to cooperate or defect
        move = 'C' if output[0] > output[1] else 'D'
        return move

    def add_score(self, score):
        # Check if key exists in scores and add score
        if self.genome_index in LearningPlayer.scores:
            LearningPlayer.scores[self.genome_index] += score
        else:
            LearningPlayer.scores[self.genome_index] = score

    @classmethod
    def evaluate_genomes(cls, genomes, config):
        # Initialize fitness for all genomes
        for genome_id, genome in genomes:
            if cls.genomes is None:
                genome.fitness = cls.scores.get(genome_id, 0)
            else:
                genome.fitness = cls.scores.get(genome_id, random.randint(0, 10))
        cls.genomes = genomes

    @classmethod
    def reset_rounds(cls, amount_of_games):
        # Divide all scores by the amount of games played to get the average score
        # Consideration: Maybe average is not a right approach
        for key in LearningPlayer.scores:
            cls.scores[key] /= amount_of_games

        # Save the winner genome and run evolution for the next generation
        cls.last_winner = cls.population.run(cls.evaluate_genomes, 1)

        # Reset the scores and increase the simulation round
        cls.scores = {}
        cls.simulation_round += 1
        cls.instances = len(cls.genomes)

    @classmethod
    def get_population_size(cls):
        return cls.genomes.__len__()

    @classmethod
    def save_winner(cls):
        with open("WinnerGenome", 'wb') as f:
            # Save the genome and neural network state
            pickle.dump(cls.last_winner, f)
        print(f"Network winner saved")


class LastWinner(Player):
    def __init__(self):

        super().__init__("Last Winner")
        # Load the winner genome from the file and create a neural network
        with open("WinnerGenome", 'rb') as f:
            self.genome = pickle.load(f)

        # Load the NEAT config file
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-rnn')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        self.net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, config)

    def make_move(self):
        # Use the last two moves as input for the RNN (modify if necessary)
        input = create_input(self.opponentHistory, self.ownHistory, self.ownScore, self.opponentScore)

        # Get the output from the neural network
        output = self.net.activate(input)
        # Decide to cooperate or defect
        move = 'C' if output[0] < 0.5 else 'D'
        return move


def create_input(oppHistory, ownHistory, own_score, opponent_score):

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

    return input

