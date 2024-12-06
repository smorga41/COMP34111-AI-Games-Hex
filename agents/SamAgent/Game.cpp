#include "Game.h"
#include <stdexcept>
#include <cassert>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iostream>
#include "AgentBase.h"

// Constructor
Game::Game(std::shared_ptr<Player> player1, std::shared_ptr<Player> player2, int boardSize, bool verbose, bool silent)
    : player1(player1), player2(player2), _turn(0), hasSwapped(false), currentPlayer(Colour::RED) {
    board = std::make_shared<Board>(boardSize);
    startTime = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Initialize the players map
    players[Colour::RED] = player1;
    players[Colour::BLUE] = player2;

    if (silent) {
        logDest = &std::cerr; // Set to silent output
    } else {
        logDest = &std::cout;
    }
}


// Destructor
Game::~Game() {
    // No resources to clean up in this case
}

// Public Methods
void Game::run() {
    play();
}

int Game::getTurn() const {
    return _turn;
}

std::shared_ptr<Board> Game::getBoard() const {
    return board;
}

// Private Methods
void Game::play() {
    EndState endState = EndState::WIN;
    const Move* opponentMove = nullptr; // Start with no previous move

    while (true) {
        _turn++;
        auto currentPlayerObj = players[currentPlayer];
        auto playerAgent = currentPlayerObj->getAgent();

        Move move = playerAgent->makeMove(_turn, *board, opponentMove);

        if (isValidMove(move)) {
            makeMove(move);
            opponentMove = &move; // Update opponentMove to point to the current move
        } else {
            endState = EndState::BAD_MOVE;
            break;
        }

        if (board->hasEnded(currentPlayer)) {
            break;
        }

        currentPlayer = ColourUtil::opposite(currentPlayer);
    }

    endGame(endState);
}


void Game::makeMove(const Move& move) {
    if (move.getX() == -1 && move.getY() == -1 && !hasSwapped) {
        std::swap(players[Colour::RED], players[Colour::BLUE]);
        players[Colour::RED]->getAgent()->setColour(Colour::RED);
        players[Colour::BLUE]->getAgent()->setColour(Colour::BLUE);
        currentPlayer = ColourUtil::opposite(currentPlayer);
        hasSwapped = true;
    } else {
        board->setTileColour(move.getX(), move.getY(), currentPlayer);
    }
}

bool Game::isValidMove(const Move& move) const {
    if (move.getX() == -1 && move.getY() == -1 && _turn == 2 && !hasSwapped) {
        return true; // Swap move allowed only on second turn
    }
    if (move.getX() >= 0 && move.getX() < board->getSize() &&
        move.getY() >= 0 && move.getY() < board->getSize()) {
        return board->getTiles()[move.getX()][move.getY()].getColour() == Colour::NONE;
    }
    return false;
}

void Game::endGame(EndState status) {
    long long totalTime = std::chrono::high_resolution_clock::now().time_since_epoch().count() - startTime;
    std::string winner = (status == EndState::WIN) ? players[currentPlayer]->getName() : "None";
    formatResult(winner, endStateToString(status), totalTime);
}

std::map<std::string, std::string> Game::formatResult(
    const std::string& winner,
    const std::string& winMethod,
    long long totalTime
) const {
    return {
        {"Winner", winner},
        {"Win Method", winMethod},
        {"Total Time", std::to_string(nsToS(totalTime)) + "s"}
    };
}

double Game::nsToS(long long ns) {
    return ns / 1e9;
}
