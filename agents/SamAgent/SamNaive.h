#ifndef SAMNAIVE_H
#define SAMNAIVE_H

#include "../../src/AgentBase.h"  
#include "../../src/Move.h"      
#include "../../src/Board.h"    
#include "../../src/Colour.h"
#include <vector>
#include <utility>
#include <random>
#include <algorithm>

class SamNaive : public AgentBase {
private:
    std::vector<std::pair<int, int>> _choices; // List of available moves
    int _boardSize; // Board size

    // Helper function to remove a move from _choices
    void removeChoice(int x, int y);

public:
    // Constructor
    explicit SamNaive(Colour colour, int boardSize = 11);

    // Implementation of the makeMove method
    Move makeMove(int turn, const Board& board, const Move* oppMove) override;
};

#endif // SAMNAIVE_H
