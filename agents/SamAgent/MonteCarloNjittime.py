from copy import deepcopy  # Used to create deep copies of objects
import random  # For random move selection
from math import sqrt, log  # Mathematical functions used in UCB1
import time  # To track time for move decisions

import numpy as np  # For efficient numerical operations
from numba import njit, prange  # For just-in-time compilation to speed up functions

from src.AgentBase import AgentBase  # Base class for agents
from src.Colour import Colour  # Enum for player colours
from src.Board import Board  # Represents the game board
from src.Move import Move  # Represents a move on the board

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
EMPTY = -1  # Represents an empty tile
RED = 0     # Represents the red player
BLUE = 1    # Represents the blue player

# Board helper functions for a 1D board representation
def board_to_1d(board: Board) -> np.ndarray:
    """Converts Board object to a 1D NumPy array."""
    return np.array([colour_to_int(tile.colour) for row in board.tiles for tile in row], dtype=np.int32)

def board_from_1d(board_1d: np.ndarray, board_size) -> Board:
    """Converts 1D NumPy array back into Board object."""
    new_board = Board(board_size)
    for idx, colour_int in enumerate(board_1d):
        x, y = divmod(idx, board_size)  # Convert index back to 2D coordinates
        colour = int_to_colour(colour_int)  # Get the Colour enum from integer
        new_board.set_tile_colour(x, y, colour)  # Set the tile colour on the new board
    return new_board

# Numba-Optimized get_available_moves_1d
@njit
def get_available_moves_1d_numba(board_1d, board_size):
    """Generate available move indices from the 1D board using Numba."""
    count = 0
    total = board_size * board_size  # Total number of tiles
    for idx in range(total):
        if board_1d[idx] == EMPTY:
            count += 1  # Count the number of empty tiles
    moves = np.empty(count, dtype=np.int32)  # Preallocate array for available moves
    current = 0
    for idx in range(total):
        if board_1d[idx] == EMPTY:
            moves[current] = idx  # Store the index of the empty tile
            current += 1
    return moves  # Return the array of available move indices

# Original get_available_moves_1d function now calls the Numba-optimized version
def get_available_moves_1d(board_1d: np.ndarray, board_size):
    """Generate available moves from the 1D board."""
    available_indices = get_available_moves_1d_numba(board_1d, board_size)  # Get available moves using optimized function
    return available_indices

# Neighbor positions for a hexagonal grid (assuming Hex game)
NEIGHBOUR_DISPLACEMENTS = np.array([
    [-1, 0], [-1, 1], [0, 1],
    [1, 0], [1, -1], [0, -1]
], dtype=np.int32)

@njit
def initialize_union_find(n):
    parent = np.arange(n, dtype=np.int32)  # Initialize each node's parent to itself
    rank = np.zeros(n, dtype=np.int32)  # Initialize ranks for union by rank optimization
    return parent, rank

@njit
def find(parent, x):
    """Finds the root of x with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # Path compression step
        x = parent[x]
    return x  # Return the root of x

@njit
def union(parent, rank, x, y):
    """Unites the sets containing x and y."""
    rx = find(parent, x)  # Find root of x
    ry = find(parent, y)  # Find root of y
    if rx != ry:
        if rank[rx] < rank[ry]:
            parent[rx] = ry  # Attach smaller rank tree under root of higher rank
        else:
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1  # Increase rank if both have same rank

@njit
def connected(parent, x, y):
    """Checks if x and y are connected."""
    return find(parent, x) == find(parent, y)  # True if both have the same root

@njit
def simulate_random_playoff_numba(simulation_board, board_size, player):
    # Initialize Union-Find
    total_tiles = board_size * board_size
    parent, rank = initialize_union_find(total_tiles + 4)  # Extra for virtual nodes

    # Virtual nodes indices for connecting edges
    red_top = total_tiles
    red_bottom = total_tiles + 1
    blue_left = total_tiles + 2
    blue_right = total_tiles + 3

    # Add existing stones to Union-Find
    for idx in prange(total_tiles):
        c = simulation_board[idx]
        if c != EMPTY:
            x = idx // board_size  # Row coordinate
            y = idx % board_size   # Column coordinate
            if c == RED:
                if x == 0:
                    union(parent, rank, idx, red_top)  # Connect to top virtual node
                if x == board_size - 1:
                    union(parent, rank, idx, red_bottom)  # Connect to bottom virtual node
            elif c == BLUE:
                if y == 0:
                    union(parent, rank, idx, blue_left)  # Connect to left virtual node
                if y == board_size - 1:
                    union(parent, rank, idx, blue_right)  # Connect to right virtual node
            # Union with same-color neighbors
            for d in range(6):
                nx = x + NEIGHBOUR_DISPLACEMENTS[d, 0]  # Neighbor's row
                ny = y + NEIGHBOUR_DISPLACEMENTS[d, 1]  # Neighbor's column
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    n_idx = nx * board_size + ny  # Neighbor's index
                    if simulation_board[n_idx] == c:
                        union(parent, rank, idx, n_idx)  # Connect same-colored neighbors

    # Check if any player has already won
    if connected(parent, red_top, red_bottom):
        return RED  # Red player has connected top to bottom
    if connected(parent, blue_left, blue_right):
        return BLUE  # Blue player has connected left to right

    current_player = player  # Start with the current player

    while True:
        # Find available moves
        available_moves = get_available_moves_1d_numba(simulation_board, board_size)
        if available_moves.size == 0:
            return 1 - current_player  # Return the opposite player as fallback if no moves left

        # Randomly select a move
        move_idx = available_moves[np.random.randint(available_moves.size)]
        simulation_board[move_idx] = current_player  # Make the move on the simulation board
        x = move_idx // board_size  # Row of the move
        y = move_idx % board_size   # Column of the move

        # Union with virtual edges if needed
        if current_player == RED:
            if x == 0:
                union(parent, rank, move_idx, red_top)  # Connect to top virtual node
            if x == board_size - 1:
                union(parent, rank, move_idx, red_bottom)  # Connect to bottom virtual node
        elif current_player == BLUE:
            if y == 0:
                union(parent, rank, move_idx, blue_left)  # Connect to left virtual node
            if y == board_size - 1:
                union(parent, rank, move_idx, blue_right)  # Connect to right virtual node

        # Union with same-color neighbors
        for d in range(6):
            nx = x + NEIGHBOUR_DISPLACEMENTS[d, 0]  # Neighbor's row
            ny = y + NEIGHBOUR_DISPLACEMENTS[d, 1]  # Neighbor's column
            if 0 <= nx < board_size and 0 <= ny < board_size:
                n_idx = nx * board_size + ny  # Neighbor's index
                if simulation_board[n_idx] == current_player:
                    union(parent, rank, move_idx, n_idx)  # Connect same-colored neighbors

        # Check for a win after the move
        if current_player == RED and connected(parent, red_top, red_bottom):
            return RED  # Red player wins
        if current_player == BLUE and connected(parent, blue_left, blue_right):
            return BLUE  # Blue player wins

        # Switch player for next turn
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
    max_ucb1 = -1e10  # Initialize with a very low value
    best_index = -1
    for i in range(child_visits.shape[0]):
        if child_visits[i] == 0:
            ucb1 = 1e10  # Assign a very high value to prioritize unexplored nodes
        else:
            exploitation = child_wins[i] / child_visits[i]  # Calculate exploitation term
            exploration = c_param * sqrt((2 * log(parent_visits)) / child_visits[i])  # Calculate exploration term
            ucb1 = exploitation + exploration  # UCB1 formula
        if ucb1 > max_ucb1:
            max_ucb1 = ucb1  # Update max UCB1 value
            best_index = i  # Update best child index
    return best_index  # Return the index of the best child

class TreeNode:
    def __init__(self, board_1d: np.ndarray, board_size: int, parent=None, move: Move = None, player: Colour = Colour.RED):
        self.board_1d = board_1d.copy()  # Ensure separate copy to avoid mutations
        self.board_size = board_size  # Size of the board
        self.parent = parent  # Reference to the parent node
        self.move = move  # The move which led to this node
        self.player = colour_to_int(player)  # Convert Colour enum to int for internal use
        self.children = []  # List to hold child nodes
        self.visits = 0  # Number of times node was visited
        self.wins = 0  # Number of wins from this node

    def is_fully_expanded(self):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(get_available_moves_1d(self.board_1d, self.board_size))

    def best_child(self, c_param):
        """
        Selects the best child node based on the UCB1 formula using a Numba-optimized function.
        """
        if self.visits == 0:
            # Avoid division by zero and log(0) issues
            return None

        num_children = len(self.children)
        child_visits = np.empty(num_children, dtype=np.int32)  # Array to hold visits for each child
        child_wins = np.empty(num_children, dtype=np.float32)  # Array to hold wins for each child
        for i, child in enumerate(self.children):
            child_visits[i] = child.visits
            child_wins[i] = child.wins

        best_index = select_best_child_numba(self.visits, child_visits, child_wins, c_param)  # Get best child index
        if best_index == -1 or best_index >= num_children:
            return None  # No valid children found
        return self.children[best_index]  # Return the best child node

    def expand(self):
        """Expands the node by adding a new child for an untried move."""
        tried_moves = set((child.move.x, child.move.y) for child in self.children)  # Set of moves already tried
        available_indices = get_available_moves_1d(self.board_1d, self.board_size)  # Get available moves
        for idx in available_indices:
            x, y = divmod(idx, self.board_size)  # Get 2D coordinates from index
            if (x, y) not in tried_moves:
                new_board_1d = self.board_1d.copy()  # Clone the current board state
                new_board_1d[idx] = self.player  # Apply the move to the cloned board

                next_player_int = 1 - self.player  # Switch player
                next_player = int_to_colour(next_player_int)  # Get the next player's Colour enum

                child_node = TreeNode(
                    board_1d=new_board_1d,
                    board_size=self.board_size,
                    parent=self,
                    move=Move(x, y),
                    player=next_player
                )  # Create a new child node

                self.children.append(child_node)  # Add the child to the list
                return child_node  # Return the newly created child
        return self  # If all moves are tried, return self

    def simulate_random_playoff(self):
        """Runs a random simulation from the current node."""
        result_int = simulate_random_playoff_numba(self.board_1d.copy(), self.board_size, self.player)  # Run simulation
        return result_int  # Return the result of the simulation

    def backpropagate(self, result: int):
        """Updates the visit and win counts up the tree based on simulation result."""
        self.visits += 1  # Increment visit count
        if (1 - self.player) == result:
            self.wins += 1  # Increment win count if the opponent won
        if self.parent:
            self.parent.backpropagate(result)  # Recursively backpropagate the result to the parent

class MCTS:
    def __init__(self, c_param: float = 0.5, time_limit: float = 5.0, confidence_margin: float = 0.05):
        self.c_param = c_param  # Exploration parameter for UCB1
        self.time_limit = time_limit  # Time limit for the search in seconds
        self.iterations_run = 0  # Counter for the number of iterations run
        self.confidence_margin = confidence_margin  # Margin to decide when to stop early

        # Warm-up calls to compile Numba functions
        dummy_board = np.full(1, EMPTY, dtype=np.int32)  # Create a dummy board
        simulate_random_playoff_numba(dummy_board.copy(), 1, RED)  # Compile the simulation function
        select_best_child_numba(0, np.array([0], dtype=np.int32), np.array([0.0], dtype=np.float32), 0.5)  # Compile the selection function

    def search(self, initial_board: Board, player: Colour):
        """Performs the MCTS search to find the best move."""
        board_size = initial_board.size  # Get the size of the board
        initial_board_1d = board_to_1d(initial_board)  # Convert the board to a 1D array
        root = TreeNode(board_1d=initial_board_1d, board_size=board_size, player=player)  # Create the root node
        start_time = time.time()  # Record the start time

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.time_limit:
                print(f"Time limit {self.time_limit} seconds reached, completed {self.iterations_run} iterations")
                break  # Exit if time limit is reached

            node = self._select(root)  # Select a node to explore
            if not self.check_terminal(node):
                node = node.expand()  # Expand the node if it's not terminal

            result = node.simulate_random_playoff()  # Run a simulation from the node
            if result is not None:
                node.backpropagate(result)  # Backpropagate the simulation result

            self.iterations_run += 1  # Increment the iteration counter

            # Every 10,000 iterations, check if a clear best move has been found
            if len(root.children) >= 2 and self.iterations_run % 10000 == 0:
                # Get win rates of the top two moves
                sorted_children = sorted(root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else 0, reverse=True)
                best_win_rate = sorted_children[0].wins / sorted_children[0].visits if sorted_children[0].visits > 0 else 0
                second_best_win_rate = sorted_children[1].wins / sorted_children[1].visits if sorted_children[1].visits > 0 else 0
                print(f"confidence margin {best_win_rate - second_best_win_rate}")
                if (best_win_rate - second_best_win_rate) >= self.confidence_margin:
                    print(f"Confidence threshold met after {self.iterations_run} iterations")
                    break  # Stop search if confidence margin is met

        best_child = root.best_child(c_param=0)  # Choose the best child without exploration
        # Convert integer result back to Colour enum if needed
        if best_child is not None and best_child.visits > 0:
            selected_move = best_child.move  # Get the move from the best child
            win_ratio = best_child.wins / best_child.visits  # Calculate win ratio
            print(f"Selected Move ({selected_move.x}, {selected_move.y}) - Wins: {best_child.wins}, Visits: {best_child.visits}, Win Ratio: {win_ratio:.2f}")
        else:
            # Fallback if no best child found
            available_moves = get_available_moves_1d(initial_board_1d, board_size)  # Get available moves
            if available_moves.size > 0:
                move_idx = available_moves[0]  # Select the first available move
                x, y = divmod(move_idx, board_size)  # Convert index to 2D coordinates
                selected_move = Move(x, y)
                print(f"No visits yet. Selecting first available move ({x}, {y})")
            else:
                selected_move = Move(-1, -1)  # No moves available
                print("No available moves.")
        return selected_move  # Return the selected move

    def _select(self, node: TreeNode):
        """Traverses the tree to select a node for expansion."""
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c_param)  # Choose the best child based on UCB1
            if node is None:
                break
        return node  # Return the selected node

    def check_terminal(self, node: TreeNode) -> bool:
        # Terminal check is handled in simulation; can be optimized further if needed
        return False  # Currently always returns False

def cloneBoard(board: Board) -> Board:
    """Creates a deep copy of the board to avoid side effects."""
    new_board = Board(board.size)  # Initialize a new board with the same size
    new_board.winner = None  # Reset the winner
    new_board._tiles = deepcopy(board.tiles)  # Deep copy the tiles
    return new_board  # Return the cloned board

def get_available_moves(board: Board) -> list[Move]:
    """Lists all possible moves that can be made from the current board state."""
    available_moves = []
    for i in range(board.size):
        for j in range(board.size):
            if board.tiles[i][j].colour is None:
                available_moves.append(Move(i, j))  # Add the move if the tile is empty
    return available_moves  # Return the list of available moves

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour, c_param: float = 0.5, time_limit: float = 5.0, max_moves: int = 30, confidence_margin: float = 0.05):
        super().__init__(colour)  # Initialize the base Agent

        self.c_param = c_param  # Exploration constant for UCB1
        self.time_limit = time_limit  # Time limit per move
        self.confidence_margin = confidence_margin  # Margin to decide when to stop search early

        self.max_moves = max_moves  # Maximum number of moves to consider
        self.time_used = 0  # Tracks time used so far
        self.time_limit = 300  # Total time limit (appears to override the earlier time_limit)

        self.fair_opens = [(0, 10), (1,2), (1, 8), (2,0)]  # Predefined fair opening moves
        self.swaps = [(0,10), (1,9), (1,10), (9,0), (9,1), (10,0)]  # Predefined swap conditions

    def make_move(self, turn: int, board: Board, opp_move: Move | None):
        print(self.time_used, self.time_limit)  # Print the time used and time limit
        if turn // 2 > self.max_moves - 5:
            self.max_moves += 10  # Increase max_moves as the game progresses

        # Clone the board to avoid side effects
        if turn == 1:
            # Use a list of known fair first moves
            x, y = random.choice(self.fair_opens)  # Select a random fair opening move
            return Move(x, y)  # Return the selected move

        elif turn == 2:
            # Decide to swap if opponent's move is part of self.swaps or within a certain range
            if opp_move and (2 <= opp_move.x <= 8 or (opp_move.x, opp_move.y) in self.swaps):
                return Move(-1, -1)  # Indicate a swap

        # Calculate the remaining time budget for the search
        time_budget = (self.time_limit - self.time_used) / (self.max_moves - (turn // 2))
        
        start = time.time()  # Record the start time of the move decision
        board_clone = cloneBoard(board)  # Clone the board for simulation

        # Handle opponent's move (if any)
        if opp_move and opp_move.x != -1 and opp_move.y != -1:
            opp_colour = Colour.opposite(self.colour)  # Get the opponent's colour
            board_clone.set_tile_colour(opp_move.x, opp_move.y, opp_colour)  # Apply the opponent's move to the cloned board

        # Initialize MCTS with the cloned board and current player
        mcts = MCTS(c_param=self.c_param, time_limit=time_budget, confidence_margin=self.confidence_margin)
        best_move = mcts.search(initial_board=board_clone, player=self.colour)  # Perform the search to find the best move
        print(f"MCTS selected move at ({best_move.x}, {best_move.y})")  # Print the selected move
        end = time.time()  # Record the end time of the move decision
        self.time_used += end - start  # Update the time used
        return Move(best_move.x, best_move.y)  # Return the selected move
