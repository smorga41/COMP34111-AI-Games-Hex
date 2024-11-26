#include "Board.h"
#include "../agents/SamAgent/SamNaive.h"
#include <iostream>

int main() {
    try {
        // Create a board of size 11
        Board board(11);

        // Create a SamNaive agent with the RED color
        SamNaive agent(Colour::RED);

        // Get the agent's first move
        Move move = agent.makeMove(1, board, nullptr);

        // Output the move in the expected format "x,y"
        std::cout << move.getX() << "," << move.getY() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1; // Non-zero exit code for errors
    }

    return 0; // Success
}
