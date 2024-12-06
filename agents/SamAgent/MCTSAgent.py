import subprocess

class MCTSAgent:
    def __init__(self, colour):
        self.colour = colour

    def makeMove(self, turn, board, oppMove):
        # Call hex_game with arguments and parse output
        command = ["./hex_game", "--turn", str(turn), "--colour", str(self.colour)]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout.strip()  # Return the move
