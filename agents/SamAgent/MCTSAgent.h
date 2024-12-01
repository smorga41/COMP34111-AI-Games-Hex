#ifndef MCTS_AGENT_H
#define MCTS_AGENT_H

#include "AgentBase.h"
#include "Move.h"
#include "Board.h"
#include "Colour.h"
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <chrono>

// Function to get available moves in 1D board representation
std::vector<Move> getAvailableMoves1D(const std::vector<Colour>& board1D, int boardSize);

class TreeNode {
public:
    TreeNode(const std::vector<Colour>& board1D, int boardSize, TreeNode* parent = nullptr, Move move = Move(), Colour player = Colour::RED);

    bool isFullyExpanded() const;
    TreeNode* bestChild(double cParam) const;
    TreeNode* expand();
    Colour simulateRandomPlayoff();
    void backpropagate(Colour result);
    const std::vector<Colour>& getBoard1D() const;
    bool checkWin(const std::vector<Colour>& board, Colour colour) const;

    const Move& getMove() const { return move; }
    int getVisits() const { return visits; }
    double getWins() const { return wins; }
    const std::vector<std::unique_ptr<TreeNode>>& getChildren() const { return children; } // Getter for children

private:
    std::vector<Colour> board1D;
    int boardSize;
    TreeNode* parent;
    Move move;
    Colour player;
    std::vector<std::unique_ptr<TreeNode>> children;
    int visits;
    double wins;

    static const std::vector<std::pair<int, int>> NEIGHBOUR_DISPLACEMENTS;
};

class MCTS {
public:
    MCTS(int iterations = 20000, double cParam = 0.5, double timeLimit = 10.0);
    bool checkTerminal(TreeNode* node) const; // Correctly declared here
    Move search(const Board& initialBoard, Colour player);

private:
    int iterations;
    double cParam;
    double timeLimit;

    TreeNode* select(TreeNode* node);
    std::vector<Colour> boardTo1D(const Board& board) const;
    Board boardFrom1D(const std::vector<Colour>& board1D, int boardSize) const;
    std::vector<Move> getAvailableMoves1D(const std::vector<Colour>& board1D, int boardSize) const;
};

#endif // MCTS_AGENT_H
