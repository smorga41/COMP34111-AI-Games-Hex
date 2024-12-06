#include "Colour.h"

Colour ColourUtil::opposite(Colour colour) {
    return (colour == Colour::RED) ? Colour::BLUE : (colour == Colour::BLUE ? Colour::RED : Colour::NONE);
}

char ColourUtil::getChar(Colour colour) {
    switch (colour) {
        case Colour::RED: return 'R';
        case Colour::BLUE: return 'B';
        default: return 'X';
    }
}

Colour ColourUtil::fromChar(char c) {
    if (c == 'R') return Colour::RED;
    if (c == 'B') return Colour::BLUE;
    return Colour::NONE;
}
