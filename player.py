import random
import neat
import os
import multiprocessing


class Player:
    def __init__(self, name):
        self.name = name
        self.ownHistory = []
        self.opponentHistory = []

    def make_move(self):
        pass

    def update_history(self, own_move, opponent_move):
        self.ownHistory.append(own_move)
        self.opponentHistory.append(opponent_move)

    def reset_history(self):
        self.opponentHistory = []
        self.ownHistory = []


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
    def __init__(self):
        super().__init__("Learning Player")
        # Load the NEAT config file
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-rnn')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)
        # Create the population
        self.population = neat.Population(self.config)
        self.current_genome = None
        self.genome_index = 0
        self.genomes = None
        self.score = 0
        winner = self.population.run(self.evaluate_genomes, 1)  # Run for one generation

        self.net = neat.nn.RecurrentNetwork.create(winner, self.config)

    def evaluate_genomes(self, genomes, config):
        self.genomes = genomes
        # Initialize fitness for all genomes
        for genome_id, genome in genomes:
            genome.fitness = self.score

    def make_move(self):
        # Use the last two moves as input for the RNN (modify if necessary)
        if len(self.opponentHistory) > 1:
            inputs = [self.opponentHistory[0], self.ownHistory[0]]
        else:
            inputs = [0, 0]

        inputs = [1 if move == "D" else 0 for move in inputs]

        # Get the output from the neural network
        output = self.net.activate(inputs)[0]
        # Decide to cooperate or defect
        move = 'C' if output > 0.5 else 'D'
        return move

    def reset_rounds(self):
        winner = self.net.run(self.evaluate_genomes, 1)
        self.net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        self.score = 0

    def add_score(self, score):
        self.score = self.score + score
