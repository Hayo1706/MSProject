import random
import neat
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

    def make_move(self):
        return "C"


