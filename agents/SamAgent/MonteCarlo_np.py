from copy import deepcopy
import random
from math import sqrt, log
import time

import numpy as np

from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Board import Board
from src.Move import Move

# Board helper functions for a 1d board representation using NumPy arrays
def board_to_1d(board: Board) -> np.ndarray:
    return np.array([tile.colour for row in board.tiles for tile in row], dtype=object)

def board_from_1d(board_1d: np.ndarray, board_size) -> Board:
    """Converts 1d numpy array back into a Board object."""
    new_board = Board(board_size)
    for idx, colour in enumerate(board_1d):
        x, y = divmod(idx, board_size)
        new_board.set_tile_colour(x, y, colour)
    return new_board

def get_available_moves_1d(board_1d: np.ndarray, board_size):
    """Generate available moves from the 1d board using NumPy."""
    empty_indices = np.where(board_1d == None)[0]
    moves = [Move(idx // board_size, idx % board_size) for idx in empty_indices]
    return moves

NEIGHBOUR_DISPLACEMENTS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

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
            board_clone.set_tile_colour(opp_move.x, opp_move.y, Colour.opposite(self.colour))
                
        # Initialize MCTS
        mcts = MCTS(iterations=self.iterations, c_param=self.c_param, time_limit=self.time_limit)
        best_move = mcts.search(initial_board=board_clone, player=self.colour)
        print(f"MCTS selected move at ({best_move.x}, {best_move.y})")
        return Move(best_move.x, best_move.y)

class TreeNode:
    def __init__(self, board_1d: np.ndarray, board_size: int, parent=None, move: Move = None, player: Colour = Colour.RED):
        self.board_1d = board_1d
        self.board_size = board_size
        self.parent = parent
        self.move = move  # The move which led to this node
        self.player = player  # The current player whose turn it is
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(get_available_moves_1d(self.board_1d, self.board_size))

    def best_child(self, c_param):
        # Selects the child with the highest UCB1 value
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
        tried_moves = [child.move for child in self.children]
        possible_moves = get_available_moves_1d(self.board_1d, self.board_size)
        for move in possible_moves:
            if move not in tried_moves:
                new_board_1d = self.board_1d.copy()
                idx = move.x * self.board_size + move.y
                new_board_1d[idx] = self.player

                # Determine next player
                next_player = Colour.opposite(self.player)

                # Create the child node with next player's turn
                child_node = TreeNode(
                    board_1d=new_board_1d, 
                    board_size=self.board_size,
                    parent=self,
                    move=move,
                    player=next_player)
                
                self.children.append(child_node)
                return child_node
        return self

    def simulate_random_playoff(self):
        simulation_board = self.board_1d.copy()
        current_player = self.player  # Start with the player whose turn it is at this node

        # Check for immediate win
        if self.check_win(simulation_board, Colour.opposite(current_player)):
            return Colour.opposite(current_player)

        while True:
            available_moves = np.where(simulation_board == None)[0]
            if not len(available_moves):
                raise Exception("Unexpected game state, no possible moves.")
            
            # Randomly select a move
            move_idx = np.random.choice(available_moves)
            simulation_board[move_idx] = current_player

            # Check for win
            if self.check_win(simulation_board, current_player):
                return current_player

            # Switch player
            current_player = Colour.opposite(current_player)

    def check_win(self, board_1d, colour):
        board_size = self.board_size
        visited = np.zeros(board_size * board_size, dtype=bool)

        def dfs(idx):
            if visited[idx]:
                return False
            visited[idx] = True
            x, y = divmod(idx, board_size)

            # Check win condition
            if colour == Colour.RED and x == board_size - 1:
                return True
            if colour == Colour.BLUE and y == board_size - 1:
                return True

            # Explore neighbors
            for displacement in NEIGHBOUR_DISPLACEMENTS:
                nx, ny = x + displacement[0], y + displacement[1]
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    neighbor_idx = nx * board_size + ny
                    if board_1d[neighbor_idx] == colour and not visited[neighbor_idx]:
                        if dfs(neighbor_idx):
                            return True
            return False

        # Start DFS from relevant edges
        if colour == Colour.RED:
            # Indices where x == 0
            start_indices = [idx for idx in range(len(board_1d)) if (idx // board_size == 0 and board_1d[idx] == Colour.RED)]
        elif colour == Colour.BLUE:
            # Indices where y == 0
            start_indices = [idx for idx in range(len(board_1d)) if (idx % board_size == 0 and board_1d[idx] == Colour.BLUE)]
        else:
            return False

        for idx in start_indices:
            if dfs(idx):
                return True
        return False

    def backpropagate(self, result: Colour):
        """
        Updates the node's statistics based on the simulation result.

        Args:
            result (Colour): The winner of the simulation
        """
        self.visits += 1
        if Colour.opposite(self.player) == result:
            self.wins += 1

        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, iterations: int = 3, c_param: float = 1.4, time_limit: float = 10.0):
        self.iterations = iterations  # Can limit by time in the future instead
        self.c_param = c_param
        self.time_limit = time_limit
        self.iterations_run = 0

    def search(self, initial_board: Board, player: Colour):
        board_size = initial_board.size
        initial_board_1d = board_to_1d(initial_board)
        root = TreeNode(board_1d=initial_board_1d, board_size=board_size, player=player)
        start_time = time.time()

        for _ in range(self.iterations):
            # Track time
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.time_limit:
                print(f"Time limit {self.time_limit} reached, completed {self.iterations_run} iterations")
                break
            
            node = self._select(root)
            if not self.check_terminal(node):
                node = node.expand()
            
            result = node.simulate_random_playoff()
            if result is not None:
                node.backpropagate(result)

            self.iterations_run += 1

        best_child = root.best_child(c_param=0)
        return best_child.move
    
    def _select(self, node: TreeNode):
        """
        Uses Upper Confidence Bound to select node for expansion
        """
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c_param)
        return node
    
    def check_terminal(self, node: TreeNode) -> bool:
        return node.check_win(node.board_1d, Colour.RED) or node.check_win(node.board_1d, Colour.BLUE)

def cloneBoard(board: Board):
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
