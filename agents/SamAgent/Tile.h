#ifndef TILE_H
#define TILE_H

#include "Colour.h"

class Tile {
public:
    // Constants for neighbours and displacements
    static const int NEIGHBOUR_COUNT = 6;
    static const int I_DISPLACEMENTS[NEIGHBOUR_COUNT];
    static const int J_DISPLACEMENTS[NEIGHBOUR_COUNT];

private:
    int _x;                // x-coordinate of the tile
    int _y;                // y-coordinate of the tile
    Colour _colour;        // Colour of the tile (default: None/empty)
    bool _visited;         // Whether the tile is visited (default: false)

public:
    // Constructor
    Tile(int x, int y, Colour colour = static_cast<Colour>(-1), bool visited = false);

    // Getter for x
    int getX() const;

    // Getter for y
    int getY() const;

    // Getter for colour
    Colour getColour() const;

    // Setter for colour
    void setColour(Colour colour);

    // Mark the tile as visited
    void visit();

    // Check if the tile is visited
    bool isVisited() const;

    // Clear the visited state
    void clearVisit();
};

#endif  // TILE_H
