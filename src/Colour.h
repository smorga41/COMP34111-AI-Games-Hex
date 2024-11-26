#ifndef COLOUR_H
#define COLOUR_H

#include <string>

enum class Colour {
    NONE,   // Add a neutral value
    RED,
    BLUE
};


class ColourUtil {
public:
    // Returns the name of the colour as an uppercase character
    static char getChar(Colour colour);

    // Returns a Colour from its char representation ('R' or 'B')
    static Colour fromChar(char c);

    // Returns the opposite Colour
    static Colour opposite(Colour colour);
};

#endif  // COLOUR_H
