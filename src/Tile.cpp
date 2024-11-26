#include "Tile.h"

// Define the neighbour displacement arrays
const int Tile::I_DISPLACEMENTS[Tile::NEIGHBOUR_COUNT] = {-1, -1, 0, 1, 1, 0};
const int Tile::J_DISPLACEMENTS[Tile::NEIGHBOUR_COUNT] = {0, 1, 1, 0, -1, -1};

// Constructor
Tile::Tile(int x, int y, Colour colour, bool visited)
    : _x(x), _y(y), _colour(colour), _visited(visited) {}

// Getter for x
int Tile::getX() const {
    return _x;
}

// Getter for y
int Tile::getY() const {
    return _y;
}

// Getter for colour
Colour Tile::getColour() const {
    return _colour;
}

// Setter for colour
void Tile::setColour(Colour colour) {
    _colour = colour;
}

// Mark the tile as visited
void Tile::visit() {
    _visited = true;
}

// Check if the tile is visited
bool Tile::isVisited() const {
    return _visited;
}

// Clear the visited state
void Tile::clearVisit() {
    _visited = false;
}
