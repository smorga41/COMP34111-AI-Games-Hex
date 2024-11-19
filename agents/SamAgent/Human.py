import logging
from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class Human(AgentBase):
    """This class describes a Human Hex agent that interacts via console input.
    It prompts the user to input their move coordinates in the form x, y.

    The class inherits from AgentBase, which is an abstract class.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    _board_size: int

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._board_size = 7  # Default board size; will be updated based on the actual board
        self._available_moves = set()  # Use a set for O(1) lookups

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """

        # Update board size based on the current board
        self._board_size = board.size

        # Prompt user for input until a valid move is entered
        while True:
            try:
                user_input = input(f"Enter your move as 'x,y' (0-{self._board_size -1}): ")
                x_str, y_str = user_input.strip().split(',')

                # Convert input strings to integers
                x = int(x_str)
                y = int(y_str)

                # Validate coordinates are within bounds
                if not (0 <= x < self._board_size) or not (0 <= y < self._board_size):
                    print(f"Coordinates out of bounds. Please enter values between 0 and {self._board_size -1}.")
                    continue


                # Move is valid; remove from available moves and return
                return Move(x, y)

            except ValueError:
                print("Invalid input format. Please enter your move as two integers separated by a comma, e.g., '3,4'.")
            except KeyboardInterrupt:
                print("\nMove input cancelled by user. Exiting game.")
                exit()
