#include "Move.h"
#include <sstream>

// Constructor
Move::Move(int x, int y) : _x(x), _y(y) {}


// Define equality operator
bool Move::operator==(const Move& other) const {
    return _x == other._x && _y == other._y;
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
