#include "MCTSAgent.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <ctime>

const std::vector<std::pair<int, int>> TreeNode::NEIGHBOUR_DISPLACEMENTS = {
    {-1, 0}, {-1, 1}, {0, 1}, {1, 0}, {1, -1}, {0, -1}
};

TreeNode::TreeNode(const std::vector<Colour>& board1D, int boardSize, TreeNode* parent, Move move, Colour player)
    : board1D(board1D), boardSize(boardSize), parent(parent), move(move), player(player), visits(0), wins(0) {}

bool TreeNode::isFullyExpanded() const {
    auto availableMoves = getAvailableMoves1D(board1D, boardSize);
    return children.size() == availableMoves.size();
}

TreeNode* TreeNode::bestChild(double cParam) const {
    double bestValue = -std::numeric_limits<double>::infinity(); 
    TreeNode* bestNode = nullptr;

    for (const auto& child : children) {
        if (child->visits == 0) {
            return child.get();
        }

        double exploitation = static_cast<double>(child->wins) / child->visits;
        double exploration = cParam * sqrt(2 * log(visits) / child->visits);
        double value = exploitation + exploration;

        if (value > bestValue) {
            bestValue = value;
            bestNode = child.get();
        }
    }

    if (!bestNode) {
        throw std::runtime_error("No valid child found in bestChild()");
    }
    return bestNode;
}

TreeNode* TreeNode::expand() {
    auto availableMoves = getAvailableMoves1D(board1D, boardSize);

    for (const auto& move : availableMoves) {
        // Check if the move is already represented by a child
        if (std::none_of(children.begin(), children.end(), [&](const std::unique_ptr<TreeNode>& child) {
                return child->getMove() == move;
            })) {
            // Create a new board state with the current player's move applied
            auto newBoard1D = board1D;
            newBoard1D[move.getX() * boardSize + move.getY()] = player;

            // Determine the next player
            Colour nextPlayer = ColourUtil::opposite(player);

            // Create a new child node and add it to the children vector
            auto newNode = std::make_unique<TreeNode>(newBoard1D, boardSize, this, move, nextPlayer);
            children.push_back(std::move(newNode));

            // Return the newly created child
            return children.back().get();
        }
    }

    // If no moves are available to expand, throw an error
    throw std::runtime_error("No moves available to expand");
}


std::vector<Move> getAvailableMoves1D(const std::vector<Colour>& board1D, int boardSize) {
    std::vector<Move> availableMoves;

    for (size_t idx = 0; idx < board1D.size(); ++idx) {
        if (board1D[idx] == Colour::NONE) {
            int x = idx / boardSize;
            int y = idx % boardSize;
            availableMoves.emplace_back(x, y);
        }
    }

    return availableMoves;
}



// Simulate a random game from the current state
Colour TreeNode::simulateRandomPlayoff() {
    auto simulationBoard = board1D;
    Colour currentPlayer = player;

    std::random_device rd;
    std::mt19937 gen(rd());

    while (true) {
        auto availableMoves = getAvailableMoves1D(simulationBoard, boardSize);
        if (availableMoves.empty()) {
            return Colour::NONE;
        }

        std::uniform_int_distribution<> dis(0, availableMoves.size() - 1);
        auto move = availableMoves[dis(gen)];
        simulationBoard[move.getX() * boardSize + move.getY()] = currentPlayer;

        if (checkWin(simulationBoard, currentPlayer)) {
            return currentPlayer;
        }

        currentPlayer = ColourUtil::opposite(currentPlayer);
    }
}




// Check if a player has won
bool TreeNode::checkWin(const std::vector<Colour>& board1D, Colour colour) const {
    std::vector<bool> visited(board1D.size(), false);
    std::function<bool(int)> dfs = [&](int idx) {
        if (visited[idx]) return false;
        visited[idx] = true;

        int x = idx / boardSize;
        int y = idx % boardSize;

        if ((colour == Colour::RED && x == boardSize - 1) || (colour == Colour::BLUE && y == boardSize - 1)) {
            return true;
        }

        for (const auto& [dx, dy] : NEIGHBOUR_DISPLACEMENTS) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < boardSize && ny >= 0 && ny < boardSize) {
                int nIdx = nx * boardSize + ny;
                if (board1D[nIdx] == colour && dfs(nIdx)) {
                    return true;
                }
            }
        }

        return false;
    };

    for (int i = 0; i < boardSize; ++i) {
        if ((colour == Colour::RED && board1D[i] == Colour::RED && dfs(i)) ||
            (colour == Colour::BLUE && board1D[i * boardSize] == Colour::BLUE && dfs(i * boardSize))) {
            return true;
        }
    }

    return false;
}

bool MCTS::checkTerminal(TreeNode* node) const {
    return node->checkWin(node->getBoard1D(), Colour::RED) || 
           node->checkWin(node->getBoard1D(), Colour::BLUE);
}
const std::vector<Colour>& TreeNode::getBoard1D() const {
    return board1D;
}
void TreeNode::backpropagate(Colour result) {
    ++visits;

    // Use ColourUtil::opposite to get the opposite Colour
    if (result == ColourUtil::opposite(player)) {
        ++wins;
    }

    if (parent) {
        parent->backpropagate(result);
    }
}


// MCTS constructor
MCTS::MCTS(int iterations, double cParam, double timeLimit)
    : iterations(iterations), cParam(cParam), timeLimit(timeLimit) {}

// MCTS search
Move MCTS::search(const Board& initialBoard, Colour player) {
    int boardSize = initialBoard.getSize();
    auto board1D = boardTo1D(initialBoard);
    auto root = std::make_unique<TreeNode>(board1D, boardSize, nullptr, Move(), player);

    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = currentTime - startTime;
        if (elapsedTime.count() >= timeLimit) break;

        TreeNode* node = select(root.get());
        if (!checkTerminal(node)) {
            node = node->expand();
        }

        Colour result = node->simulateRandomPlayoff();
        node->backpropagate(result);
    }

    return root->bestChild(0.0)->getMove();
}

// Select the best node for expansion
TreeNode* MCTS::select(TreeNode* node) {
    while (node->isFullyExpanded() && !node->getChildren().empty()) {
        node = node->bestChild(cParam);
    }
    return node;
}


// Convert board to 1D
std::vector<Colour> MCTS::boardTo1D(const Board& board) const {
    std::vector<Colour> board1D;
    const auto& tiles = board.getTiles(); // Assuming getTiles returns the 2D vector of tiles
    for (const auto& row : tiles) {
        for (const auto& tile : row) {
            board1D.push_back(tile.getColour()); // Assuming Tile has getColour
        }
    }
    return board1D;
}

// Convert 1D to board
Board MCTS::boardFrom1D(const std::vector<Colour>& board1D, int boardSize) const {
    Board newBoard(boardSize);
    for (size_t idx = 0; idx < board1D.size(); ++idx) {
        int x = idx / boardSize;
        int y = idx % boardSize;
        newBoard.setTileColour(x, y, board1D[idx]);
    }
    return newBoard;
}

// Get available moves in 1D
std::vector<Move> MCTS::getAvailableMoves1D(const std::vector<Colour>& board1D, int boardSize) const {
    std::vector<Move> moves;
    for (size_t idx = 0; idx < board1D.size(); ++idx) {
        if (board1D[idx] == Colour::NONE) {
            int x = idx / boardSize;
            int y = idx % boardSize;
            moves.emplace_back(x, y);
        }
    }
    return moves;
}

