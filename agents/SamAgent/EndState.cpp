#include "EndState.h"

// Convert EndState enum to a string representation
std::string endStateToString(EndState state) {
    switch (state) {
        case EndState::WIN:
            return "WIN";
        case EndState::TIMEOUT:
            return "TIMEOUT";
        case EndState::BAD_MOVE:
            return "BAD_MOVE";
        case EndState::FAILED_LOAD:
            return "FAILED_LOAD";
        default:
            return "UNKNOWN";
    }
}
