#ifndef COLOUR_H
#define COLOUR_H

enum class Colour { NONE, RED, BLUE };

class ColourUtil {
public:
    static Colour opposite(Colour colour);
    static char getChar(Colour colour);
    static Colour fromChar(char c);
};

#endif // COLOUR_H
