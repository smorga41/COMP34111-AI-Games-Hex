#ifndef BOARD_H
#define BOARD_H

#include <vector>
#include <string>
#include "Tile.h"
#include "Colour.h"

class Board {
private:
    int _size;                                 
    std::vector<std::vector<Tile>> _tiles;    
    Colour _winner;                           

    void DFSColour(int x, int y, Colour colour);
    void clearTiles();

public:
    explicit Board(int boardSize = 11);
    Board clone() const;    
    static Board fromString(const std::string& stringInput, int boardSize = 11);

    bool hasEnded(Colour colour);

    std::string printBoard() const;

    int getSize() const;
    const std::vector<std::vector<Tile>>& getTiles() const;
    Colour getWinner() const;

    void setTileColour(int x, int y, Colour colour);
};

#endif  // BOARD_H
