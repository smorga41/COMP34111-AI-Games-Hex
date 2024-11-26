#ifndef GAME_H
#define GAME_H

#include <string>
#include <map>
#include <memory>
#include "Board.h"
#include "Move.h"
#include "Player.h"
#include "EndState.h"

class Game {
public:
    static const long long MAXIMUM_TIME = 5LL * 60 * 1000000000; // 5 minutes in nanoseconds


    Game(std::shared_ptr<Player> player1, std::shared_ptr<Player> player2, int boardSize = 11, bool verbose = false, bool silent = false);
    ~Game();

    void run();
    int getTurn() const;
    std::shared_ptr<Board> getBoard() const;

private:
    int _turn;
    bool hasSwapped;
    Colour currentPlayer;
    std::shared_ptr<Board> board;
    long long startTime;
    std::shared_ptr<Player> player1;
    std::shared_ptr<Player> player2;

    std::map<Colour, std::shared_ptr<Player>> players; // Declare the players map

    std::ostream* logDest;

    void play();
    void makeMove(const Move& move);
    bool isValidMove(const Move& move) const;
    void endGame(EndState status);
    std::map<std::string, std::string> formatResult(
        const std::string& winner,
        const std::string& winMethod,
        long long totalTime
    ) const;
    static double nsToS(long long ns);
};

#endif // GAME_H
