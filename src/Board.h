#ifndef BOARD_H
#define BOARD_H

#include <vector>
#include <string>
#include "Tile.h"
#include "Colour.h"

class Board {
private:
    int _size;                                 // Size of the board
    std::vector<std::vector<Tile>> _tiles;     // 2D vector of tiles
    Colour _winner;                            // Winner of the game

    // Depth-first search helper function for colour-based connectivity
    void DFSColour(int x, int y, Colour colour);

    // Clears the visited state of all tiles
    void clearTiles();

public:
    // Constructor
    explicit Board(int boardSize = 11);

    // Creates a board from a string representation
    static Board fromString(const std::string& stringInput, int boardSize = 11);

    // Checks if the game has ended for a given colour
    bool hasEnded(Colour colour);

    // Prints the board to the console
    std::string printBoard() const;

    // Getters
    int getSize() const;
    const std::vector<std::vector<Tile>>& getTiles() const;
    Colour getWinner() const;

    // Sets the colour of a tile at a specific position
    void setTileColour(int x, int y, Colour colour);
};

#endif  // BOARD_H
