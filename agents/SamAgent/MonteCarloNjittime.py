from copy import deepcopy
import random
from math import sqrt, log
import time

import numpy as np
from numba import njit, prange

from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Board import Board
from src.Move import Move

# Helper functions for Colour conversions

def colour_to_int(colour: Colour) -> int:
    """Converts a Colour enum to its integer representation."""
    if colour == Colour.RED:
        return 0
    elif colour == Colour.BLUE:
        return 1
    else:
        return -1  # For EMPTY or None

def int_to_colour(value: int) -> Colour | None:
    """Converts an integer to its corresponding Colour enum."""
    if value == 0:
        return Colour.RED
    elif value == 1:
        return Colour.BLUE
    else:
        return None  # For EMPTY or invalid values

# Constants for Colours
EMPTY = -1
RED = 0
BLUE = 1

# Board helper functions for a 1D board representation
def board_to_1d(board: Board) -> np.ndarray:
    """Converts Board object to a 1D NumPy array."""
    return np.array([colour_to_int(tile.colour) for row in board.tiles for tile in row], dtype=np.int32)

def board_from_1d(board_1d: np.ndarray, board_size) -> Board:
    """Converts 1D NumPy array back into Board object."""
    new_board = Board(board_size)
    for idx, colour_int in enumerate(board_1d):
        x, y = divmod(idx, board_size)
        colour = int_to_colour(colour_int)
        new_board.set_tile_colour(x, y, colour)
    return new_board

# Numba-Optimized get_available_moves_1d
@njit
def get_available_moves_1d_numba(board_1d, board_size):
    """Generate available move indices from the 1D board using Numba."""
    count = 0
    total = board_size * board_size
    for idx in range(total):
        if board_1d[idx] == EMPTY:
            count += 1
    moves = np.empty(count, dtype=np.int32)
    current = 0
    for idx in range(total):
        if board_1d[idx] == EMPTY:
            moves[current] = idx
            current += 1
    return moves

# Original get_available_moves_1d function now calls the Numba-optimized version
def get_available_moves_1d(board_1d: np.ndarray, board_size):
    """Generate available moves from the 1D board."""
    available_indices = get_available_moves_1d_numba(board_1d, board_size)
    return available_indices

NEIGHBOUR_DISPLACEMENTS = np.array([
    [-1, 0], [-1, 1], [0, 1],
    [1, 0], [1, -1], [0, -1]
], dtype=np.int32)

@njit
def initialize_union_find(n):
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    return parent, rank

@njit
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # Path compression
        x = parent[x]
    return x

@njit
def union(parent, rank, x, y):
    rx = find(parent, x)
    ry = find(parent, y)
    if rx != ry:
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        else:
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

@njit
def connected(parent, x, y):
    return find(parent, x) == find(parent, y)

@njit
def simulate_random_playoff_numba(simulation_board, board_size, player):
    # Initialize Union-Find
    total_tiles = board_size * board_size
    parent, rank = initialize_union_find(total_tiles + 4)  # Extra for virtual nodes

    # Virtual nodes indices
    red_top = total_tiles
    red_bottom = total_tiles + 1
    blue_left = total_tiles + 2
    blue_right = total_tiles + 3

    # Add existing stones to Union-Find
    for idx in prange(total_tiles):
        c = simulation_board[idx]
        if c != EMPTY:
            x = idx // board_size
            y = idx % board_size
            if c == RED:
                if x == 0:
                    union(parent, rank, idx, red_top)
                if x == board_size - 1:
                    union(parent, rank, idx, red_bottom)
            elif c == BLUE:
                if y == 0:
                    union(parent, rank, idx, blue_left)
                if y == board_size - 1:
                    union(parent, rank, idx, blue_right)
            # Union with same-color neighbors
            for d in range(6):
                nx = x + NEIGHBOUR_DISPLACEMENTS[d, 0]
                ny = y + NEIGHBOUR_DISPLACEMENTS[d, 1]
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    n_idx = nx * board_size + ny
                    if simulation_board[n_idx] == c:
                        union(parent, rank, idx, n_idx)

    # Check if any player has already won
    if connected(parent, red_top, red_bottom):
        return RED
    if connected(parent, blue_left, blue_right):
        return BLUE

    current_player = player

    while True:
        # Find available moves
        available_moves = get_available_moves_1d_numba(simulation_board, board_size)
        if available_moves.size == 0:
            return 1 - current_player  # Opposite player as fallback

        # Randomly select a move
        move_idx = available_moves[np.random.randint(available_moves.size)]
        simulation_board[move_idx] = current_player
        x = move_idx // board_size
        y = move_idx % board_size

        # Union with virtual edges if needed
        if current_player == RED:
            if x == 0:
                union(parent, rank, move_idx, red_top)
            if x == board_size - 1:
                union(parent, rank, move_idx, red_bottom)
        elif current_player == BLUE:
            if y == 0:
                union(parent, rank, move_idx, blue_left)
            if y == board_size - 1:
                union(parent, rank, move_idx, blue_right)

        # Union with same-color neighbors
        for d in range(6):
            nx = x + NEIGHBOUR_DISPLACEMENTS[d, 0]
            ny = y + NEIGHBOUR_DISPLACEMENTS[d, 1]
            if 0 <= nx < board_size and 0 <= ny < board_size:
                n_idx = nx * board_size + ny
                if simulation_board[n_idx] == current_player:
                    union(parent, rank, move_idx, n_idx)

        # Check for a win
        if current_player == RED and connected(parent, red_top, red_bottom):
            return RED
        if current_player == BLUE and connected(parent, blue_left, blue_right):
            return BLUE

        # Switch player
        current_player = 1 - current_player

# Numba-Optimized Function to Select Best Child
@njit
def select_best_child_numba(parent_visits, child_visits, child_wins, c_param):
    """
    Selects the best child index based on the UCB1 formula.

    Parameters:
    - parent_visits (int): Number of visits to the parent node.
    - child_visits (array): Array of visits for each child.
    - child_wins (array): Array of wins for each child.
    - c_param (float): Exploration parameter.

    Returns:
    - best_index (int): Index of the best child.
    """
    max_ucb1 = -1e10
    best_index = -1
    for i in range(child_visits.shape[0]):
        if child_visits[i] == 0:
            ucb1 = 1e10  # Represents infinity
        else:
            exploitation = child_wins[i] / child_visits[i]
            exploration = c_param * sqrt((2 * log(parent_visits)) / child_visits[i])
            ucb1 = exploitation + exploration
        if ucb1 > max_ucb1:
            max_ucb1 = ucb1
            best_index = i
    return best_index

class TreeNode:
    def __init__(self, board_1d: np.ndarray, board_size: int, parent=None, move: Move = None, player: Colour = Colour.RED):
        self.board_1d = board_1d.copy()  # Ensure separate copy
        self.board_size = board_size
        self.parent = parent
        self.move = move  # The move which led to this node
        self.player = colour_to_int(player)  # Convert enum to int for internal use
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(get_available_moves_1d(self.board_1d, self.board_size))

    def best_child(self, c_param):
        """
        Selects the best child node based on the UCB1 formula using a Numba-optimized function.
        """
        if self.visits == 0:
            # Avoid division by zero and log(0) issues
            return None

        num_children = len(self.children)
        child_visits = np.empty(num_children, dtype=np.int32)
        child_wins = np.empty(num_children, dtype=np.float32)
        for i, child in enumerate(self.children):
            child_visits[i] = child.visits
            child_wins[i] = child.wins

        best_index = select_best_child_numba(self.visits, child_visits, child_wins, c_param)
        if best_index == -1 or best_index >= num_children:
            return None  # No valid children found
        return self.children[best_index]

    def expand(self):
        tried_moves = set((child.move.x, child.move.y) for child in self.children)
        available_indices = get_available_moves_1d(self.board_1d, self.board_size)
        for idx in available_indices:
            x, y = divmod(idx, self.board_size)
            if (x, y) not in tried_moves:
                new_board_1d = self.board_1d.copy()
                new_board_1d[idx] = self.player

                next_player_int = 1 - self.player  # Switch player
                next_player = int_to_colour(next_player_int)

                child_node = TreeNode(
                    board_1d=new_board_1d,
                    board_size=self.board_size,
                    parent=self,
                    move=Move(x, y),
                    player=next_player
                )

                self.children.append(child_node)
                return child_node
        return self

    def simulate_random_playoff(self):
        # Call the Numba-optimized simulation
        result_int = simulate_random_playoff_numba(self.board_1d.copy(), self.board_size, self.player)
        return result_int

    def backpropagate(self, result: int):
        self.visits += 1
        if (1 - self.player) == result:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, iterations: int = 200000, c_param: float = 0.5, time_limit: float = 5.0, confidence_margin: float = 0.05):
        self.iterations = iterations
        self.c_param = c_param
        self.time_limit = time_limit
        self.iterations_run = 0
        self.confidence_margin = confidence_margin

        # Warm-up calls to compile Numba functions
        dummy_board = np.full(1, EMPTY, dtype=np.int32)
        simulate_random_playoff_numba(dummy_board.copy(), 1, RED)
        select_best_child_numba(0, np.array([0], dtype=np.int32), np.array([0.0], dtype=np.float32), 0.5)

    def search(self, initial_board: Board, player: Colour):
        board_size = initial_board.size
        initial_board_1d = board_to_1d(initial_board)
        root = TreeNode(board_1d=initial_board_1d, board_size=board_size, player=player)
        start_time = time.time()

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.time_limit:
                print(f"Time limit {self.time_limit} seconds reached, completed {self.iterations_run} iterations")
                break

            node = self._select(root)
            if not self.check_terminal(node):
                node = node.expand()

            result = node.simulate_random_playoff()
            if result is not None:
                node.backpropagate(result)

            self.iterations_run += 1

            # every 10,000 iterations check if an obvious best move has been found
            if len(root.children) >= 2 and self.iterations_run % 10000 == 0:
                # Get win rates of the top two moves
                sorted_children = sorted(root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else 0, reverse=True)
                best_win_rate = sorted_children[0].wins / sorted_children[0].visits if sorted_children[0].visits > 0 else 0
                second_best_win_rate = sorted_children[1].wins / sorted_children[1].visits if sorted_children[1].visits > 0 else 0
                print(f"confidence margin {best_win_rate - second_best_win_rate}")
                if (best_win_rate - second_best_win_rate) >= self.confidence_margin:
                    print(f"Confidence threshold met after {self.iterations_run} iterations")
                    break

        best_child = root.best_child(c_param=0)
        # Convert integer result back to Colour enum if needed
        if best_child is not None and best_child.visits > 0:
            selected_move = best_child.move
            win_ratio = best_child.wins / best_child.visits
            print(f"Selected Move ({selected_move.x}, {selected_move.y}) - Wins: {best_child.wins}, Visits: {best_child.visits}, Win Ratio: {win_ratio:.2f}")
        else:
            # Fallback if no best child found
            available_moves = get_available_moves_1d(initial_board_1d, board_size)
            if available_moves.size > 0:
                move_idx = available_moves[0]
                x, y = divmod(move_idx, board_size)
                selected_move = Move(x, y)
                print(f"No visits yet. Selecting first available move ({x}, {y})")
            else:
                selected_move = Move(-1, -1)  # No moves available
                print("No available moves.")
        return selected_move

    def _select(self, node: TreeNode):
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c_param)
            if node is None:
                break
        return node

    def check_terminal(self, node: TreeNode) -> bool:
        # Terminal check is handled in simulation; can be optimized further if needed
        return False

def cloneBoard(board: Board) -> Board:
    new_board = Board(board.size)
    new_board.winner = None
    new_board._tiles = deepcopy(board.tiles)
    return new_board

def get_available_moves(board: Board) -> list[Move]:
    available_moves = []
    for i in range(board.size):
        for j in range(board.size):
            if board.tiles[i][j].colour is None:
                available_moves.append(Move(i, j))
    return available_moves

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour, iterations: int = 20000, c_param: float = 0.5, time_limit: float = 5.0, max_moves: int = 30, confidence_margin: float = 0.05):
        super().__init__(colour)
        self.iterations = iterations
        self.c_param = c_param
        self.time_limit = time_limit
        self.confidence_margin = confidence_margin

        self.max_moves = max_moves
        self.time_used = 0
        self.time_limit = 300

        self.fair_opens = [(0, 10), (1,2), (1, 8), (2,0)]
        self.swaps = [(0,10), (1,9), (1,10), (9,0), (9,1), (10,0)]

    def make_move(self, turn: int, board: Board, opp_move: Move | None):
        if turn // 2 > self.max_moves - 5:
            self.max_moves += 10
        # Clone the board to avoid side effects
        if turn == 1:
            # use dict of known fair first moves
            x, y = random.choice(self.fair_opens)
            return Move(x, y)

        elif turn == 2:
            # swap if move is part of self.swaps or 2 <= x <= 8 
            if opp_move and (2 <= opp_move.x <= 8 or (opp_move.x, opp_move.y) in self.swaps):
                return Move(-1, -1)
        time_budget = (self.time_limit - self.time_used) / (self.max_moves - (turn // 2))
        
        start = time.time()
        board_clone = cloneBoard(board)

        # Handle opponent move (if any)
        if opp_move and opp_move.x != -1 and opp_move.y != -1:
            opp_colour = Colour.opposite(self.colour)
            board_clone.set_tile_colour(opp_move.x, opp_move.y, opp_colour)

        # Initialize MCTS
        mcts = MCTS(iterations=self.iterations, c_param=self.c_param, time_limit=time_budget, confidence_margin=self.confidence_margin)
        best_move = mcts.search(initial_board=board_clone, player=self.colour)
        print(f"MCTS selected move at ({best_move.x}, {best_move.y})")
        end = time.time()
        self.time_used += end - start
        return Move(best_move.x, best_move.y)
