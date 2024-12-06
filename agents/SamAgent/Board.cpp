#include "Board.h"
#include <iostream>
#include <sstream>

// Constructor
Board::Board(int boardSize) : _size(boardSize), _winner(static_cast<Colour>(-1)) {
    for (int i = 0; i < boardSize; ++i) {
        std::vector<Tile> row;
        for (int j = 0; j < boardSize; ++j) {
            row.emplace_back(Tile(i, j));
        }
        _tiles.emplace_back(row);
    }
}

Board Board::clone() const {
    Board newBoard(_size); 
    newBoard._winner = _winner; 

    // Deep copy of tiles
    for (int i = 0; i < _size; ++i) {
        for (int j = 0; j < _size; ++j) {
            newBoard._tiles[i][j] = _tiles[i][j]; 
        }
    }

    return newBoard;
}

// Create a board from a string representation
Board Board::fromString(const std::string& stringInput, int boardSize) {
    Board board(boardSize);
    std::istringstream stream(stringInput);
    std::string line;
    int row = 0;

    while (std::getline(stream, line)) {
        int col = 0;
        for (char c : line) {
            if (c != ' ' && col < boardSize) {
                board._tiles[row][col].setColour(ColourUtil::fromChar(c));
                ++col;
            }
        }
        ++row;
    }
    return board;
}

// Perform depth-first search for connectivity
void Board::DFSColour(int x, int y, Colour colour) {
    _tiles[x][y].visit();

    // Win condition for RED (top-to-bottom) or BLUE (left-to-right)
    if ((colour == Colour::RED && x == _size - 1) ||
        (colour == Colour::BLUE && y == _size - 1)) {
        _winner = colour;
        return;
    }

    // Visit neighbours
    for (int i = 0; i < Tile::NEIGHBOUR_COUNT; ++i) {
        int nx = x + Tile::I_DISPLACEMENTS[i];
        int ny = y + Tile::J_DISPLACEMENTS[i];

        if (nx >= 0 && nx < _size && ny >= 0 && ny < _size) {
            Tile& neighbour = _tiles[nx][ny];
            if (!neighbour.isVisited() && neighbour.getColour() == colour) {
                DFSColour(nx, ny, colour);
                if (_winner != static_cast<Colour>(-1)) return;  // Early exit if a winner is found
            }
        }
    }
}

// Clears the visited status from all tiles
void Board::clearTiles() {
    for (auto& row : _tiles) {
        for (auto& tile : row) {
            tile.clearVisit();
        }
    }
}

// Check if the game has ended
bool Board::hasEnded(Colour colour) {
    if (colour == Colour::RED) {
        // Check top-to-bottom connection for RED
        for (int col = 0; col < _size; ++col) {
            if (!_tiles[0][col].isVisited() && _tiles[0][col].getColour() == Colour::RED && _winner == static_cast<Colour>(-1)) {
                DFSColour(0, col, Colour::RED);
            }
        }
    } else if (colour == Colour::BLUE) {
        // Check left-to-right connection for BLUE
        for (int row = 0; row < _size; ++row) {
            if (!_tiles[row][0].isVisited() && _tiles[row][0].getColour() == Colour::BLUE && _winner == static_cast<Colour>(-1)) {
                DFSColour(row, 0, Colour::BLUE);
            }
        }
    } else {
        throw std::invalid_argument("Invalid colour");
    }

    // Clear visited tiles
    clearTiles();

    return _winner != static_cast<Colour>(-1);
}

// Prints the board
std::string Board::printBoard() const {
    std::ostringstream output;
    std::string leadingSpaces = "";

    for (const auto& row : _tiles) {
        output << leadingSpaces;
        leadingSpaces += " ";
        for (const auto& tile : row) {
            char c = ColourUtil::getChar(tile.getColour());
            if (c == 'R') {
                output << "\033[91mR\033[0m ";  // Red
            } else if (c == 'B') {
                output << "\033[94mB\033[0m ";  // Blue
            } else {
                output << "X ";  // Default
            }
        }
        output << "\n";
    }

    return output.str();
}

// Getters
int Board::getSize() const {
    return _size;
}

const std::vector<std::vector<Tile>>& Board::getTiles() const {
    return _tiles;
}

Colour Board::getWinner() const {
    return _winner;
}

// Sets the colour of a tile
void Board::setTileColour(int x, int y, Colour colour) {
    _tiles[x][y].setColour(colour);
}
