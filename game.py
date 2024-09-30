import multiprocessing
import sys
from typing import List

from player import Player, HumanPlayer


class GameSettings:
    def __init__(self, player1: Player, player2: Player, rounds=10):
        self.player1 = player1
        self.player2 = player2
        self.rounds = rounds


class PrisonersDilemmaSimulation:
    def __init__(self, game_list: List[GameSettings]):
        self.game_list = game_list

    def _run_game(self, game_settings: GameSettings, q):
        logic = Logic(game_settings, q)
        return logic.loop()

    def run(self, q=None):
        print("Starting simulation")
        total_games = len(self.game_list)
        pool_size = multiprocessing.cpu_count() * 2
        if total_games < pool_size:
            pool_size = total_games

        args = [(game_settings, q) for game_settings in self.game_list]
        with multiprocessing.Pool(processes=pool_size) as pool:
            return pool.starmap(
                self._run_game, args
            )


class Logic:
    def __init__(self, game_settings: GameSettings, q=None):
        self.history = [[] for _ in range(2)]
        self.player1 = game_settings.player1
        self.player2 = game_settings.player2
        self.rounds = game_settings.rounds
        self.scores = {self.player1.name: 0, self.player2.name: 0}

        self.q = q

    def loop(self):
        for i in range(self.rounds):
            move1 = self.player1.make_move()
            move2 = self.player2.make_move()

            self.history[0].append(move1)
            self.history[1].append(move2)

            self.player1.update_history(move1, move2)
            self.player2.update_history(move2, move1)

            self.update_scores(move1, move2)
            if self.q is not None:
                self.q.put([self.history, self.scores])
        if self.q is not None:
            self.q.put(None)
        return self.scores

    def update_scores(self, move1, move2):
        payoff_matrix = {
            ("C", "C"): (3, 3),
            ("C", "D"): (0, 5),
            ("D", "C"): (5, 0),
            ("D", "D"): (1, 1),
        }
        score1, score2 = payoff_matrix[(move1, move2)]
        self.scores[self.player1.name] += score1
        self.scores[self.player2.name] += score2

        return self.scores
