#ifndef PLAYER_H
#define PLAYER_H

#include <string>
#include <memory>
#include "AgentBase.h" // Base class for Agent

class Player {
private:
    std::string name;
    std::shared_ptr<AgentBase> agent;
    int moveTime = 0;
    int turn = 0;

public:
    // Constructor
    Player(const std::string& name, std::shared_ptr<AgentBase> agent, int moveTime = 0, int turn = 0);

    // Getters
    std::string getName() const;
    std::shared_ptr<AgentBase> getAgent() const;
    int getMoveTime() const;
    int getTurn() const;

    // Setters
    void setMoveTime(int time);
    void incrementTurn();

    // Comparison operator
    bool operator==(const Player& other) const;

    // Destructor
    ~Player() = default;
};

#endif // PLAYER_H
