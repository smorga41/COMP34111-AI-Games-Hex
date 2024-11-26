#include "Colour.h"
#include <stdexcept>  // for std::invalid_argument

// Returns the name of the colour as an uppercase character
char ColourUtil::getChar(Colour colour) {
    switch (colour) {
        case Colour::RED:
            return 'R';
        case Colour::BLUE:
            return 'B';
        default:
            return '0';  // Default case, though unreachable with valid input
    }
}

// Returns a Colour from its char representation ('R' or 'B')
Colour ColourUtil::fromChar(char c) {
    switch (c) {
        case 'R':
            return Colour::RED;
        case 'B':
            return Colour::BLUE;
        default:
            throw std::invalid_argument("Invalid character for Colour.");
    }
}

// Returns the opposite Colour
Colour ColourUtil::opposite(Colour colour) {
    switch (colour) {
        case Colour::RED:
            return Colour::BLUE;
        case Colour::BLUE:
            return Colour::RED;
        default:
            throw std::invalid_argument("Invalid Colour.");
    }
}
