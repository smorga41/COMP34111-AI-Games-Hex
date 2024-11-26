#include "AgentBase.h"
#include <stdexcept>

// Constructor
AgentBase::AgentBase(Colour colour) : _colour(colour) {}

// Getter for colour
Colour AgentBase::getColour() const {
    return _colour;
}

// Setter for colour
void AgentBase::setColour(Colour colour) {
    _colour = colour;
}

// Returns the opposite colour
Colour AgentBase::oppColour() const {
    if (_colour == Colour::RED) {
        return Colour::BLUE;
    } else if (_colour == Colour::BLUE) {
        return Colour::RED;
    } else {
        throw std::invalid_argument("Invalid colour.");
    }
}

// Hash function (not directly used, placeholder implementation)
std::size_t AgentBase::hash() const {
    return std::hash<std::string>{}("AgentBase");
}
