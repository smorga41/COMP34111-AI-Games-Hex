from copy import deepcopy
import random
from math import sqrt, log
import time

from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Board import Board
from src.Move import Move

# Board helper funtions for a 1d board representation
def board_to_1d(board: Board) -> list:
    return [tile.colour for row in board.tiles for tile in row]

def board_from_1d(board_1d: Board, board_size) -> list:
    """Converts 1d list back into board"""
    new_board = Board(board_size)
    for idx, colour in enumerate(board_1d):
        x, y = divmod(idx, board_size)
        new_board.set_tile_colour(x, y, colour)
    return new_board

def get_available_moves_1d(board_1d, board_size):
    """Generate available moves from the 1d board"""
    return [Move(idx // board_size, idx % board_size) for idx, colour in enumerate(board_1d) if colour is None]

NEIGHBOUR_DISPLACEMENTS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)


class HexWinChecker:
    def __init__(self, board_size):
        self.board_size = board_size
        self.total_tiles = board_size * board_size

        # Virtual nodes
        self.red_top = self.total_tiles
        self.red_bottom = self.total_tiles+1
        self.blue_left = self.total_tiles+2
        self.blue_right = self.total_tiles+3

        # Initialize UnionFind with extra 4 nodes for virtual edges
        self.uf = UnionFind(self.total_tiles + 4)

    def xy_to_idx(self, x, y):
        return x*self.board_size + y

    def add_move(self, x, y, colour, board_1d):
        """Call this method after placing a tile of the given colour at (x,y)."""
        idx = self.xy_to_idx(x, y)

        # Union with virtual edges if needed
        if colour == Colour.RED:
            if x == 0:
                self.uf.union(idx, self.red_top)
            if x == self.board_size - 1:
                self.uf.union(idx, self.red_bottom)
        elif colour == Colour.BLUE:
            if y == 0:
                self.uf.union(idx, self.blue_left)
            if y == self.board_size - 1:
                self.uf.union(idx, self.blue_right)

        # Union with same-colour neighbours
        for dx, dy in NEIGHBOUR_DISPLACEMENTS:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                n_idx = self.xy_to_idx(nx, ny)
                if board_1d[n_idx] == colour:
                    self.uf.union(idx, n_idx)

    def check_win(self, colour):
        """Check if the given colour has achieved a win."""
        if colour == Colour.RED:
            return self.uf.connected(self.red_top, self.red_bottom)
        elif colour == Colour.BLUE:
            return self.uf.connected(self.blue_left, self.blue_right)
        return False


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
    def __init__(self, board_1d: list, board_size: int, parent=None, move: Move = None, player: Colour = Colour.RED):
        self.board_1d = board_1d
        self.board_size = board_size
        self.parent = parent
        self.move = move # The move which lead to this node
        self.player = player # The current player whose turn it is
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
        tried_moves = [child.move for child in self.children]
        possible_moves = get_available_moves_1d(self.board_1d, self.board_size)
        for move in possible_moves:
            if move not in tried_moves:
                new_board_1d = self.board_1d.copy()
                idx = move.x * self.board_size + move.y
                new_board_1d[idx] = self.player

                next_player = Colour.opposite(self.player)

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
        # We create a copy of the board state and set up union-find based win checking
        simulation_board = self.board_1d.copy()
        current_player = self.player

        # Initialize HexWinChecker
        win_checker = HexWinChecker(self.board_size)
        # Add all existing stones to the union-find structure
        for idx, c in enumerate(simulation_board):
            if c is not None:
                x, y = divmod(idx, self.board_size)
                win_checker.add_move(x, y, c, simulation_board)

        # Check if the previous player just made a winning move (if any)
        # Actually, at the start, we are about to make a move for current_player,
        # so we should check if opposite(current_player) had already won.
        # Usually not needed if the game is consistent, but safe to check:
        if win_checker.check_win(Colour.opposite(current_player)):
            return Colour.opposite(current_player)

        while True:
            available_moves = [idx for idx, colour in enumerate(simulation_board) if colour is None] 
            if not available_moves:
                # No moves left should not happen in a full hex game until a winner is found,
                # but let's just break and return opposite player as a fallback
                return Colour.opposite(current_player)
            
            # Randomly select a move
            move_idx = random.choice(available_moves)
            simulation_board[move_idx] = current_player
            x, y = divmod(move_idx, self.board_size)
            # Update union-find structure with the new move
            win_checker.add_move(x, y, current_player, simulation_board)

            # Check for win
            if win_checker.check_win(current_player):
                return current_player

            # Switch player
            current_player = Colour.opposite(current_player)

    def backpropagate(self, result: Colour):
        self.visits += 1
        if Colour.opposite(self.player) == result:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    def __init__(self, iterations: int = 3, c_param: float = 1.4, time_limit: float = 10.0):
        self.iterations = iterations
        self.c_param = c_param
        self.time_limit = time_limit
        self.iterations_run = 0
    
    def search(self, initial_board: Board, player: Colour):
        board_size = initial_board.size
        initial_board_1d = board_to_1d(initial_board)
        root = TreeNode(board_1d=initial_board_1d, board_size=board_size, player=player)
        start_time = time.time()

        for _ in range(self.iterations):
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
         # Print stats for the best move
        print(f"Selected Move ({best_child.move.x}, {best_child.move.y}) - Wins: {best_child.wins}, Visits: {best_child.visits}, Win Ratio: {best_child.wins/ best_child.visits}")
        return best_child.move
    
    def _select(self, node: TreeNode):
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c_param)
        return node
    
    def check_terminal(self, node: TreeNode) -> bool:
        # We rely on simulate_random_playoff checks for a winner efficiently;
        # If you want a quick terminal check: 
        # It's cheap to do with union-find if we integrated it fully at each node,
        # but we haven't. We'll trust simulate_random_playoff for now.
        return False


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
