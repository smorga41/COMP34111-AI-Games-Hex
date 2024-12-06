#include "AgentBase.h"
#include <stdexcept>

// Constructor
AgentBase::AgentBase(Colour colour) : _colour(colour) {}

// Getter for colour
Colour AgentBase::getColour() const {
    return _colour;
}

void AgentBase::setColour(Colour colour) {
    _colour = colour;
}

Colour AgentBase::oppColour() const {
    if (_colour == Colour::RED) {
        return Colour::BLUE;
    } else if (_colour == Colour::BLUE) {
        return Colour::RED;
    } else {
        throw std::invalid_argument("Invalid colour.");
    }
}

std::size_t AgentBase::hash() const {
    return std::hash<std::string>{}("AgentBase");
}
