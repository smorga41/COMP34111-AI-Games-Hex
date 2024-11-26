#include "Move.h"
#include <sstream>

// Constructor
Move::Move(int x, int y) : _x(x), _y(y) {}

// Getter for x-coordinate
int Move::getX() const {
    return _x;
}

// Getter for y-coordinate
int Move::getY() const {
    return _y;
}

// Returns a string representation of the move
std::string Move::toString() const {
    if (_x == -1 && _y == -1) {
        return "SWAP()";
    } else {
        std::ostringstream output;
        output << "(x=" << _x << ", y=" << _y << ")";
        return output.str();
    }
}
