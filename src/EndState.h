#ifndef ENDSTATE_H
#define ENDSTATE_H

#include <string>

// Enum representing the possible end conditions of a match
enum class EndState {
    WIN = 0,          // Indicates a win
    TIMEOUT = 1,      // Indicates timeout
    BAD_MOVE = 2,     // Indicates a bad move
    FAILED_LOAD = 3   // Indicates a failed load
};

// Helper functions for EndState
std::string endStateToString(EndState state);

#endif // ENDSTATE_H
