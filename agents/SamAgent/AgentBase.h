#ifndef AGENTBASE_H
#define AGENTBASE_H

#include "Colour.h"
#include "Board.h"
#include "Move.h"
#include <stdexcept>
#include <string>

class AgentBase {
protected:
    Colour _colour; 

public:
    // Constructor
    explicit AgentBase(Colour colour);

    // Virtual destructor
    virtual ~AgentBase() = default;

    // Pure virtual function to make a move
    virtual Move makeMove(int turn, const Board& board, const Move* oppMove) = 0;

    // Getter for agent's colour
    Colour getColour() const;

    // Setter for agent's colour
    void setColour(Colour colour);

    // Returns the opposite colour of the agent
    Colour oppColour() const;

    // Hash function (not directly applicable in C++, just included for illustration)
    virtual std::size_t hash() const;
};

#endif // AGENTBASE_H
