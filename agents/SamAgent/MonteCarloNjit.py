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

def get_available_moves_1d(board_1d: np.ndarray, board_size):
    """Generate available moves from the 1D board."""
    available_indices = np.where(board_1d == EMPTY)[0]
    return [Move(idx // board_size, idx % board_size) for idx in available_indices]

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
        available_moves = np.where(simulation_board == EMPTY)[0]
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
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                ucb1 = float('inf')
            else:
                exploitation = child.wins / child.visits
                exploration = c_param * sqrt((2 * log(self.visits)) / child.visits)
                ucb1 = exploitation + exploration
            choices_weights.append(ucb1)
        best_index = choices_weights.index(max(choices_weights))
        return self.children[best_index]

    def expand(self):
        tried_moves = set((child.move.x, child.move.y) for child in self.children)
        possible_moves = get_available_moves_1d(self.board_1d, self.board_size)
        for move in possible_moves:
            if (move.x, move.y) not in tried_moves:
                new_board_1d = self.board_1d.copy()
                idx = move.x * self.board_size + move.y
                new_board_1d[idx] = self.player

                next_player_int = 1 - self.player  # Switch player
                next_player = int_to_colour(next_player_int)

                child_node = TreeNode(
                    board_1d=new_board_1d,
                    board_size=self.board_size,
                    parent=self,
                    move=move,
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
    def __init__(self, iterations: int = 20000, c_param: float = 0.5, time_limit: float = 5.0):
        self.iterations = iterations
        self.c_param = c_param
        self.time_limit = time_limit
        self.iterations_run = 0

        # Warm-up calls to compile Numba functions
        dummy_board = np.full(1, EMPTY, dtype=np.int32)
        simulate_random_playoff_numba(dummy_board.copy(), 1, RED)

    def search(self, initial_board: Board, player: Colour):
        board_size = initial_board.size
        initial_board_1d = board_to_1d(initial_board)
        root = TreeNode(board_1d=initial_board_1d, board_size=board_size, player=player)
        start_time = time.time()

        for _ in range(self.iterations):
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

        best_child = root.best_child(c_param=0)
        # Convert integer result back to Colour enum if needed
        selected_move = best_child.move
        win_ratio = best_child.wins / best_child.visits if best_child.visits > 0 else 0
        print(f"Selected Move ({selected_move.x}, {selected_move.y}) - Wins: {best_child.wins}, Visits: {best_child.visits}, Win Ratio: {win_ratio:.2f}")
        return selected_move

    def _select(self, node: TreeNode):
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c_param)
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
    def __init__(self, colour: Colour, iterations: int = 20000, c_param: float = 0.5, time_limit: float = 5.0):
        super().__init__(colour)
        self.iterations = iterations
        self.c_param = c_param
        self.time_limit = time_limit

    def make_move(self, turn: int, board: Board, opp_move: Move | None):
        # Clone the board to avoid side effects
        board_clone = cloneBoard(board)

        # Handle opponent move (if any)
        if opp_move and opp_move.x != -1 and opp_move.y != -1:
            opp_colour = Colour.opposite(self.colour)
            board_clone.set_tile_colour(opp_move.x, opp_move.y, opp_colour)

        # Initialize MCTS
        mcts = MCTS(iterations=self.iterations, c_param=self.c_param, time_limit=self.time_limit)
        best_move = mcts.search(initial_board=board_clone, player=self.colour)
        print(f"MCTS selected move at ({best_move.x}, {best_move.y})")
        return Move(best_move.x, best_move.y)
