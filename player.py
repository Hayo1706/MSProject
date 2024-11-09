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

        self.net = neat.nn.FeedForwardNetwork.create(self.genome, config)

    def make_move(self):
        # Use the last two moves as input for the RNN (modify if necessary)
        input = create_input(self.opponentHistory, self.ownHistory, self.ownScore, self.opponentScore)

        # Get the output from the neural network
        output = self.net.activate(input)
        # Decide to cooperate or defect
        move = 'C' if output[0] < 0.5 else 'D'
        return move


class LastPopulation(Player):
    def __init__(self):
        super().__init__("Last Population")
        # Load the specified population from the file and create a neural network
        p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-999")
        genome = self.get_random_genome(p, 2)

        self.net = neat.nn.FeedForwardNetwork.create(genome, p.config)

    def make_move(self):
        # Use the last two moves as input for the RNN (modify if necessary)
        input = create_input(self.opponentHistory, self.ownHistory, self.ownScore, self.opponentScore)

        # Get the output from the neural network
        output = self.net.activate(input)
        print(output)
        # Decide to cooperate or defect
        move = 'C' if output[0] < 0.5 else 'D'
        return move

    def get_random_genome(self, population, genome_index):
        return random.choice(list(population.population.values()))


def create_input(oppHistory2, ownHistory2, own_score, opponent_score):
    ownHistory, oppHistory = ownHistory2[-20:], oppHistory2[-20:]

    input_CD = [
        (1 if own == "C" else -1 if own == "D" else 0,
         1 if opp == "C" else -1 if opp == "D" else 0)
        for own, opp in zip(ownHistory, oppHistory)
    ]

    input_CD = [item for history in input_CD for item in history]
    input_CD.extend([0] * (40 - len(input_CD)))

    return input_CD
