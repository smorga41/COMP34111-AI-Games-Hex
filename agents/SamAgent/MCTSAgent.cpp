#include "MCTSAgent.h"
#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>

// Main function
int main() {
    Board board(11);
    Colour playerColour = Colour::RED;
    MCTSAgent agent(playerColour, 1000, 1.4, 10.0);
    Move oppMove(3, 4);
    Move bestMove = agent.makeMove(1, board, oppMove);
    std::cout << "Agent's move: (" << bestMove.getX() << ", " << bestMove.getY() << ")" << std::endl;
    return 0;
}

// TreeNode Class Implementation
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
        if (child->visits == 0) return child.get();
        double exploitation = static_cast<double>(child->wins) / child->visits;
        double exploration = cParam * sqrt(2 * log(visits) / child->visits);
        double value = exploitation + exploration;
        if (value > bestValue) {
            bestValue = value;
            bestNode = child.get();
        }
    }
    if (!bestNode) throw std::runtime_error("No valid child found in bestChild()");
    return bestNode;
}

TreeNode* TreeNode::expand() {
    auto availableMoves = getAvailableMoves1D(board1D, boardSize);
    for (const auto& move : availableMoves) {
        if (std::none_of(children.begin(), children.end(), [&](const std::unique_ptr<TreeNode>& child) {
                return child->getMove() == move;
            })) {
            auto newBoard1D = board1D;
            newBoard1D[move.getX() * boardSize + move.getY()] = player;
            auto nextPlayer = ColourUtil::opposite(player);
            auto newNode = std::make_unique<TreeNode>(newBoard1D, boardSize, this, move, nextPlayer);
            children.push_back(std::move(newNode));
            return children.back().get();
        }
    }
    throw std::runtime_error("No moves available to expand");
}

Colour TreeNode::simulateRandomPlayoff() {
    auto simulationBoard = board1D;
    Colour currentPlayer = player;
    std::random_device rd;
    std::mt19937 gen(rd());
    while (true) {
        auto availableMoves = getAvailableMoves1D(simulationBoard, boardSize);
        if (availableMoves.empty()) return Colour::NONE;
        std::uniform_int_distribution<> dis(0, availableMoves.size() - 1);
        auto move = availableMoves[dis(gen)];
        simulationBoard[move.getX() * boardSize + move.getY()] = currentPlayer;
        if (checkWin(simulationBoard, currentPlayer)) return currentPlayer;
        currentPlayer = ColourUtil::opposite(currentPlayer);
    }
}

bool TreeNode::checkWin(const std::vector<Colour>& board1D, Colour colour) const {
    std::vector<bool> visited(board1D.size(), false);
    std::function<bool(int)> dfs = [&](int idx) {
        if (visited[idx]) return false;
        visited[idx] = true;
        int x = idx / boardSize;
        int y = idx % boardSize;
        if ((colour == Colour::RED && x == boardSize - 1) || (colour == Colour::BLUE && y == boardSize - 1)) return true;
        for (const auto& [dx, dy] : NEIGHBOUR_DISPLACEMENTS) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < boardSize && ny >= 0 && ny < boardSize) {
                int nIdx = nx * boardSize + ny;
                if (board1D[nIdx] == colour && dfs(nIdx)) return true;
            }
        }
        return false;
    };
    for (int i = 0; i < boardSize; ++i) {
        if ((colour == Colour::RED && board1D[i] == Colour::RED && dfs(i)) || 
            (colour == Colour::BLUE && board1D[i * boardSize] == Colour::BLUE && dfs(i * boardSize))) return true;
    }
    return false;
}

void TreeNode::backpropagate(Colour result) {
    ++visits;
    if (result == ColourUtil::opposite(player)) ++wins;
    if (parent) parent->backpropagate(result);
}

// MCTS Class Implementation
MCTS::MCTS(int iterations, double cParam, double timeLimit)
    : iterations(iterations), cParam(cParam), timeLimit(timeLimit) {}

Move MCTS::search(const Board& initialBoard, Colour player) {
    int boardSize = initialBoard.getSize();
    auto board1D = boardTo1D(initialBoard);
    auto root = std::make_unique<TreeNode>(board1D, boardSize, nullptr, Move(), player);
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(currentTime - startTime).count() >= timeLimit) break;
        TreeNode* node = select(root.get());
        if (!checkTerminal(node)) node = node->expand();
        Colour result = node->simulateRandomPlayoff();
        node->backpropagate(result);
    }
    return root->bestChild(0.0)->getMove();
}

TreeNode* MCTS::select(TreeNode* node) {
    while (node->isFullyExpanded() && !node->getChildren().empty()) node = node->bestChild(cParam);
    return node;
}

bool MCTS::checkTerminal(TreeNode* node) const {
    return node->checkWin(node->getBoard1D(), Colour::RED) || node->checkWin(node->getBoard1D(), Colour::BLUE);
}

std::vector<Colour> MCTS::boardTo1D(const Board& board) const {
    std::vector<Colour> board1D;
    const auto& tiles = board.getTiles();
    for (const auto& row : tiles) {
        for (const auto& tile : row) board1D.push_back(tile.getColour());
    }
    return board1D;
}

std::vector<Move> MCTS::getAvailableMoves1D(const std::vector<Colour>& board1D, int boardSize) const {
    std::vector<Move> moves;
    for (size_t idx = 0; idx < board1D.size(); ++idx) {
        if (board1D[idx] == Colour::NONE) moves.emplace_back(idx / boardSize, idx % boardSize);
    }
    return moves;
}

// MCTSAgent Class Implementation
Move MCTSAgent::makeMove(int turn, const Board& board, const Move& oppMove) {
    Board boardClone = cloneBoard(board);
    if (oppMove.getX() != -1 && oppMove.getY() != -1) boardClone.setTileColour(oppMove.getX(), oppMove.getY(), ColourUtil::opposite(colour));
    MCTS mcts(iterations, cParam, timeLimit);
    Move bestMove = mcts.search(boardClone, colour);
    std::cout << "MCTS selected move at (" << bestMove.getX() << ", " << bestMove.getY() << ")" << std::endl;
    return bestMove;
}

const std::vector<Colour>& TreeNode::getBoard1D() const {
        return board1D;
    }

std::vector<Move> getAvailableMoves1D(const std::vector<Colour>& board1D, int boardSize) {
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



