#include "Player.h"

// Constructor
Player::Player(const std::string& name, std::shared_ptr<AgentBase> agent, int moveTime, int turn)
    : name(name), agent(agent), moveTime(moveTime), turn(turn) {}

// Getters
std::string Player::getName() const {
    return name;
}

std::shared_ptr<AgentBase> Player::getAgent() const {
    return agent;
}

int Player::getMoveTime() const {
    return moveTime;
}

int Player::getTurn() const {
    return turn;
}

// Setters
void Player::setMoveTime(int time) {
    moveTime = time;
}

void Player::incrementTurn() {
    ++turn;
}

// Comparison operator
bool Player::operator==(const Player& other) const {
    // Use the hash of the agents and compare other attributes
    return (name == other.name &&
            moveTime == other.moveTime &&
            (agent ? agent.get() == other.agent.get() : !other.agent));
}
