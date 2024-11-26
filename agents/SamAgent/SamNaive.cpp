#include "SamNaive.h"
#include <iostream>
#include <random>
#include <algorithm>

// Constructor
SamNaive::SamNaive(Colour colour, int boardSize)
    : AgentBase(colour), _boardSize(boardSize) {
    // Initialize the choices list with all possible moves on the board
    for (int i = 0; i < boardSize; ++i) {
        for (int j = 0; j < boardSize; ++j) {
            _choices.emplace_back(i, j);
        }
    }
}

// Removes a move from _choices
void SamNaive::removeChoice(int x, int y) {
    _choices.erase(
        std::remove(_choices.begin(), _choices.end(), std::make_pair(x, y)),
        _choices.end()
    );
}

// Implements the makeMove method
Move SamNaive::makeMove(int turn, const Board& board, const Move* oppMove) {
    // If the opponent has made a move, remove it from the available choices
    if (oppMove != nullptr) {
        if (oppMove->getX() != -1 && oppMove->getY() != -1) {
            removeChoice(oppMove->getX(), oppMove->getY());
        }
    }

    // If it's the second turn, return a swap move with 50% chance
    if (turn == 2) {
        return Move(-1, -1); // Swap move
    }

    // Otherwise, choose a random valid move
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, _choices.size() - 1);

    int randomIndex = dis(gen);
    auto [x, y] = _choices[randomIndex];

    // Remove the chosen move from the available choices
    removeChoice(x, y);

    return Move(x, y);
}
