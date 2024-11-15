import random
from math import sqrt, log

from src.Colour import Colour
from src.Board import Board
from src.Move import Move

# Tree NOde (represents a board state)
class TreeNode:
    def __init__(self, board, parent=None, move: Move = None, player: Colour = Colour.RED):
        self.board = board
        self.parent = parent
        self.move = move # The move which lead to this node
        self.player = player # The player who made the move
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.board.get_available_moves)

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

    def expand(self): # returns the treenode to be explored
        tried_moves = [child.move for child in self.children]
        possible_moves = self.board.get_available_moves()
        for move in possible_moves:
            if move not in tried_moves:
                # CLone board and try move
                #TODO Make the board representation a 1d list, cloning board is ineffivcient
                new_board = self.board.clone()
                new_board.set_tile_color(move.x, move.y, self.player)

                # Determine next player
                next_player = Colour.opposite(self.player)

                # Create the child
                child_node = TreeNode(board=new_board, parent=self, move=move, player=next_player)
                self.children.append(child_node)
                return child_node
        #TODO remove this before final submission as raising an error would result in an auto loss
        raise Exception("No moves left to expand")
    
    def simulate_random_playoff(self):
        #TODO This may be improved in future with some kind of 'heuristic' random approach which better represents human play than random
        """
        Performs random playoff from current node down to endstate

        Returns:
            Color: the color of the winning player
        """
        simulation_board = self.board.clone()
        current_player = self.player

        while True:
            available_moves = simulation_board.get_available_moves()
            if not available_moves:
                # Shoud be impossible as hex has no draws
                raise("Unexpected game state, no possible moves ")
            
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
        if self.player == result:
            self.wins += 1

        if self.parent:
            self.parent.backpropagated(result)
