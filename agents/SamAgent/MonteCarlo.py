from copy import deepcopy
import random
from math import sqrt, log

from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Board import Board
from src.Move import Move

# Tree NOde (represents a board state)

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour, iterations: int = 10000, c_param: float = 0.5):
        super().__init__(colour)
        self.iterations = iterations
        self.c_param = c_param

    

    def make_move(self, turn: int, board: Board, opp_move: Move | None):
        # Clone the board to avoid side effects
        board_clone = cloneBoard(board)

        # Handle opponent move (if any)
        if opp_move and opp_move.x != -1 and opp_move.y != -1:
            board_clone.set_tile_colour(opp_move.x, opp_move.y, Colour.opposite(self.colour))
            
        # Initialize MCTS
        mcts = MCTS(iterations=self.iterations, c_param=self.c_param)
        best_move = mcts.search(initial_board=board_clone, player=self.colour)
        print(f"MCTS best move {best_move.move.x}, {best_move.move.y}")
        return Move(best_move.move.x, best_move.move.y)

class TreeNode:
    def __init__(self, board, parent=None, move: Move = None, player: Colour = Colour.RED):
        self.board = board
        self.parent = parent
        self.move = move # The move which lead to this node
        self.player = player # The current player whos turn it is
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(get_available_moves(self.board))

    def best_child(self, c_param):
        # Selects the child with the highest value to explore next
        # c_param: exploration modiifer, higher -> more exploration
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                #prioritise unvisited nodes
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
        possible_moves = get_available_moves(self.board)
        for move in possible_moves:
            if move not in tried_moves:
                new_board = cloneBoard(self.board)
                new_board.set_tile_colour(move.x, move.y, self.player)

                # Determine next player
                next_player = Colour.opposite(self.player)

                # Create the child node with next player's turn
                child_node = TreeNode(board=new_board, parent=self, move=move, player=next_player)
                self.children.append(child_node)
                return child_node
        return self

    
    def simulate_random_playoff(self):
        simulation_board = cloneBoard(self.board)
        # Check for immediate win
        if simulation_board.has_ended(Colour.opposite(self.player)):
                return Colour.opposite(self.player)

        simulation_board._winner = None  # Reset the winner
        current_player = self.player  # Start with the player whose turn it is at this node

        while True:
            available_moves = get_available_moves(simulation_board)
            if not available_moves:
                if simulation_board.has_ended(current_player):
                    return current_player

                raise Exception("Unexpected game state, no possible moves ")
            
            # Randomly select a move
            move = random.choice(available_moves)
            simulation_board.set_tile_colour(move.x, move.y, current_player)

            # Check for win
            if simulation_board.has_ended(current_player):
                return current_player

            # Switch player
            current_player = Colour.opposite(current_player)

    
    def backpropagate(self, result: Colour):
        """
        Updates the node's statistics based on the simuation result.

        Args:
            result (Color): The winner of the simulation
        """
        self.visits += 1
        if Colour.opposite(self.player) == result:
            self.wins += 1

        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, iterations: int = 3, c_param: float = 1.4):
        self.iterations = iterations #TODO cound limit by time in future instead maybe
        self.c_param = c_param
    

    def search(self, initial_board: Board, player: Colour):
        root = TreeNode(board=cloneBoard(initial_board), player=player)
        for _ in range(self.iterations):

            node = self._select(root)
            if not node.board.has_ended(node.player):
                node = node.expand()
            
            result = node.simulate_random_playoff()
            node.backpropagate(result)

        best_child = root.best_child(c_param = 0)
        return best_child
    
    def _select(self, node: TreeNode):
        """
        Uses Upper Confidence bound to select node for expansion
        """
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c_param)
        return node

#NOTE These should really be a part of the board class but im unsure if were allowed to change that
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
