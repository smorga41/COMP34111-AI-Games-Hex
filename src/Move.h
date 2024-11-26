#ifndef MOVE_H
#define MOVE_H

#include <string>

class Move {
private:
    int _x; // x-coordinate of the move
    int _y; // y-coordinate of the move

public:
    // Default constructor for a SWAP move
    explicit Move(int x = -1, int y = -1);

    // Getters
    int getX() const;
    int getY() const;

    // Returns a string representation of the move
    std::string toString() const;

    // Destructor
    ~Move() = default;
};

#endif // MOVE_H
