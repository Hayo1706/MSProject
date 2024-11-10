import multiprocessing
import threading
import tkinter
from tkinter import messagebox
from tkinter import ttk as tk
from player import *
from game import *
import itertools

from game import PrisonersDilemma
import sv_ttk


class PrisonersDilemmaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prisoner's Dilemma Game")
        self.root.geometry("1600x800")

        # Main frame divided into left (controls) and right (status)
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        # Left LabelFrame for game controls and strategy selection
        self.left_frame = tkinter.LabelFrame(main_frame, text="Game Controls", width=400, padx=10, pady=10,
                                             relief="groove")
        self.left_frame.pack(side="left", fill="y", padx=(10, 5), pady=(10, 10))
        self.left_frame.pack_propagate(0)

        # Right LabelFrame for game/simulation status
        self.right_frame = tkinter.LabelFrame(main_frame, text="Game Status", padx=10, pady=10,
                                              relief="groove")
        self.right_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=(10, 10))
        self.table = None

        # Mode selection label and dropdown
        self.mode_label = tk.Label(self.left_frame, text="Select Mode:")
        self.mode_label.pack(pady=(10, 5))

        self.mode_var = tkinter.StringVar()
        self.mode_dropdown = tk.Combobox(self.left_frame, textvariable=self.mode_var, state="readonly", width=20)
        self.mode_dropdown["values"] = ["Play vs AI"]
        self.mode_dropdown.current(0)  # Default to "Play vs AI"
        self.mode_dropdown.pack(pady=(5, 10))

        # Strategy selection label for "Play vs AI"
        self.strategy_label = tk.Label(self.left_frame, text="Select AI Strategy to Play Against:")
        self.strategy_label.pack(pady=(10, 5))

        # Strategy dropdown for "Play vs AI" mode
        self.strategy_var = tkinter.StringVar()
        self.strategy_dropdown = tk.Combobox(self.left_frame, textvariable=self.strategy_var, state="readonly",
                                             width=20)
        self.strategy_dropdown.pack(pady=(5, 10))

        # Play Game button
        self.play_button = tk.Button(self.left_frame, text="Play Game", command=self.run_game)
        self.simulation_checkboxes = []

        self.current_mode = "Play vs AI"  # Default mode
        # Dynamically populate strategy dropdown
        self.populate_strategy_dropdown()
        self.update_ui()


    def update_ui(self):
        """Update UI elements based on the selected mode."""
        if self.current_mode == "Play vs AI":
            # Show play vs AI elements
            self.strategy_label.config(text="Select AI Strategy to Play Against:")
            self.strategy_dropdown.pack(pady=(5, 10))
            self.play_button.pack(pady=(10, 10))


    def run_game(self):
        """Start a manual game between the human and the selected AI."""
        print("Running game")
        rounds = 20

        strategy_name = self.strategy_var.get()
        # Create a Manager
        manager = multiprocessing.Manager()
        # Create a Queue
        update_q = manager.Queue()
        input_q = manager.Queue()

        human_player = HumanPlayer(input_q)
        ai_strategy = self.get_strategy(strategy_name)()

        game_settings = GameSettings(human_player, ai_strategy, rounds)

        self.table = ColorTable(self.right_frame, 2, rounds, input_q)

        # Start the game
        game = PrisonersDilemma([game_settings])
        listener_thread = threading.Thread(target=self.update_game_status, args=(update_q,))
        listener_thread.start()

        game_thread = threading.Thread(target=game.run, args=(update_q,))
        game_thread.start()

    def update_game_status(self, queue):
        last_state = None
        while True:
            state = queue.get()
            if state is None:
                break
            self.table.update_cell(0, len(state[0][0]) - 1, state[0][0][-1])
            self.table.update_cell(1, len(state[0][1]) - 1, state[0][1][-1])
            self.table.update_score(state[1])

            last_state = state

        self.show_game_results(last_state[1])

    def show_game_results(self, scores):
        """Display game results."""

        # Unpack the dictionary into keys and values
        (key1, val1), (key2, val2) = scores.items()

        # Compare the values and return the corresponding key
        if val1 > val2:
            winner = key1
        else:
            winner = key2

        game_status = f"Game Over! {winner} wins!\n"
        messagebox.showinfo("Game Over", game_status)

    def populate_strategy_dropdown(self):
        """Populate the strategy dropdown dynamically based on available subclasses of Player."""
        strategies = self.get_all_strategies()
        strategy_names = [strategy.__name__ for strategy in strategies]
        self.strategy_dropdown["values"] = strategy_names
        self.strategy_dropdown.current(0)  # Default to the first strategy

    def get_all_strategies(self):
        """Return all subclasses of Player dynamically."""
        return [cls for cls in Player.__subclasses__() if cls.__name__ != "HumanPlayer"]

    def get_strategy(self, strategy_name):
        """Return the strategy class based on the selected strategy name."""
        strategies = self.get_all_strategies()
        for strategy in strategies:
            if strategy.__name__ == strategy_name:
                return strategy


class ColorTable:
    def __init__(self, parent, rows, cols, output_queue: multiprocessing.Queue):
        self.rows = rows
        self.cols = cols
        self.cells = []
        self.queue = output_queue

        # Create a frame to hold the table and the row labels
        self.table_frame = tk.Frame(parent, borderwidth=1, relief="solid")
        self.table_frame.pack(padx=10, pady=20)

        # Create a frame for the labels ("You", "Player") and the grid
        label_frame = tk.Frame(self.table_frame)
        label_frame.pack(side="left", padx=10)

        grid_frame = tk.Frame(self.table_frame)
        grid_frame.pack(padx=10, side="left")

        # Add row labels ("You", "Player") on the left side, bigger and centered
        row_labels = ["You", "AI"]
        for row in range(rows):
            label = tkinter.Label(label_frame, text=row_labels[row], font=("Arial", 12), width=10, anchor="e", height=2)
            label.pack(pady=4)  # Adjust pady to center the label vertically

        square_width = int(80 / self.cols)
        # Create the grid of labels inside the grid_frame
        for row in range(rows):
            row_frame = tk.Frame(grid_frame)  # A frame for each row
            row_frame.pack()
            row_cells = []
            for col in range(cols):
                label = tkinter.Label(row_frame, width=min(square_width, 6), height=3, bg="white", borderwidth=1,
                                      relief="solid")
                label.pack(side="left", padx=1, pady=1)
                row_cells.append(label)
            self.cells.append(row_cells)

        # Add score labels for both players
        self.score_label = tk.Label(parent, text="Make a move to start the game", font=("Arial", 12))
        self.score_label.pack(pady=5)

        # Add buttons below the table
        button_frame = tk.Frame(self.table_frame)
        button_frame.pack(pady=10, side="right", fill="x", anchor="center")

        self.cooperate_button = tk.Button(button_frame, width=15, padding=5, text="Cooperate",
                                          command=lambda: self.put_in_queue("C"))
        self.cooperate_button.pack(side="top", padx=5, pady=5)

        self.defect_button = tk.Button(button_frame, width=15, padding=5, text="Defect",
                                       command=lambda: self.put_in_queue("D"))
        self.defect_button.pack(side="bottom", padx=5, pady=5)

    def update_cell(self, row, col, type):
        if type == "C":
            self.cells[row][col].config(bg="green")
        else:
            self.cells[row][col].config(bg="red")

    def update_score(self, scores):
        self.score_label.config(text=str(scores).replace("{", "").replace("}", "").replace("'", ""))

    def put_in_queue(self, move):
        self.queue.put(move)

    def unpack(self):
        self.score_label.pack_forget()
        self.table_frame.pack_forget()


# Run the GUI
if __name__ == "__main__":
    root = tkinter.Tk()
    sv_ttk.set_theme("dark")
    app = PrisonersDilemmaGUI(root)
    root.mainloop()