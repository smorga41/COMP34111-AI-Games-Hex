import argparse
import csv
import logging
import traceback
from datetime import datetime
from itertools import combinations, product
from multiprocessing import Pool, TimeoutError

from src.Colour import Colour
from src.Game import Game, format_result
from src.Player import Player
from src.EndState import EndState

from agents.SamAgent.MonteCarlo import MCTSAgent  # Import your MCTSAgent class

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the timeout limit in seconds
TIME_OUT_LIMIT = 300

fieldnames = [
    "player1",
    "player2",
    "winner",
    "win_method",
    "player1_move_time",
    "player2_move_time",
    "player1_turns",
    "player2_turns",
    "total_turns",
    "total_game_time",
]
time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def run(games: list[tuple[str, str]], agents: dict):
    """Run the tournament."""
    result_file_path = f"game_results_{time_stamp}.csv"
    error_game_list_path = f"error_game_list_{time_stamp}.log"

    with open(result_file_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    # Run the tournament
    game_results = []
    with Pool() as pool:
        result = [
            pool.apply_async(
                run_match,
                (agent_pair, agents),
            )
            for agent_pair in games
        ]

        for i, game_result in enumerate(result):
            try:
                r = game_result.get(timeout=TIME_OUT_LIMIT)
                with open(result_file_path, "a", newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writerow(r)
                game_results.append(r)

            except TimeoutError as error:
                logger.warning(f"Timed out between {games[i]}")
                with open(error_game_list_path, "a") as err_file:
                    err_file.write(f"{games[i]}, {repr(error)}\n")

            except Exception as error:
                logger.error(f"Exception occurred between {games[i]}: {repr(error)}")
                logger.error(traceback.format_exc())
                with open(error_game_list_path, "a") as err_file:
                    err_file.write(f"{games[i]}, {repr(error), {traceback.format_exc()}}\n")

    export_stats(game_results)

def run_match(agent_pair: tuple[str, str], agents: dict) -> dict:
    """Run a single game between two agents."""
    player1_name, player2_name = agent_pair
    logger.info(f"Starting game between {player1_name} and {player2_name}")

    params1 = agents[player1_name]
    params2 = agents[player2_name]

    # Instantiate agents with specified hyperparameters
    player1_agent = MCTSAgent(colour=Colour.RED, **params1)
    player2_agent = MCTSAgent(colour=Colour.BLUE, **params2)

    player1 = Player(name=player1_name, agent=player1_agent)
    player2 = Player(name=player2_name, agent=player2_agent)

    # Run the game
    game = Game(
        player1=player1,
        player2=player2,
        board_size=11,
        silent=True,
    )
    result = game.run()
    logger.info(f"Game complete between {player1_name} and {player2_name}")

    return result

def export_stats(gameResults: list[dict]):
    playerStats = {}

    statEntry = {
        "matches": 0,
        "wins": 0,
        "win_rate": 0,
        "total_move_time": 0,
        "total_moves": 0,
        "average_move_time": 0,
        "illegal_moves_loss": 0,
        "time_out_loss": 0,
        "regular_loss": 0,
    }

    # populate the player stats dictionary
    for result in gameResults:
        for player in [result["player1"], result["player2"]]:
            if player not in playerStats:
                playerStats[player] = statEntry.copy()

    # fill in the data
    for result in gameResults:
        player1 = result["player1"]
        player2 = result["player2"]
        winner = result["winner"]

        if result["win_method"] == EndState.FAILED_LOAD:
            if winner == "":
                continue
            else:
                playerStats[winner]["matches"] += 1
                playerStats[winner]["wins"] += 1
        else:
            if winner == player1:
                loser = player2
            else:
                loser = player1

            playerStats[player1]["matches"] += 1
            playerStats[player2]["matches"] += 1
            playerStats[player1]["total_move_time"] += result["player1_move_time"]
            playerStats[player2]["total_move_time"] += result["player2_move_time"]
            playerStats[player1]["total_moves"] += result["player1_turns"]
            playerStats[player2]["total_moves"] += result["player2_turns"]

            playerStats[winner]["wins"] += 1
            playerStats[loser]["illegal_moves_loss"] += 1 if result["win_method"] == "BAD_MOVE" else 0
            playerStats[loser]["time_out_loss"] += 1 if result["win_method"] == "TIMEOUT" else 0
            playerStats[loser]["regular_loss"] += 1 if result["win_method"] == "WIN" else 0

    for player, stats in playerStats.items():
        playerStats[player]["win_rate"] = stats["wins"] / stats["matches"] if stats["matches"] > 0 else 0
        playerStats[player]["average_move_time"] = (
            stats["total_move_time"] / stats["total_moves"] if stats["total_moves"] > 0 else 0
        )

    with open(f"game_stat_{time_stamp}.csv", "w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["player"] + list(statEntry.keys()))
        for player, stats in playerStats.items():
            writer.writerow([player] + list(stats.values()))


if __name__ == "__main__":
    # Define the hyperparameter grid
    hyperparameter_grid = {
        'iterations': [20000],
        'c_param': [0 + 0.1 * x for x in range(0,5,1)],
        'time_limit': [1]
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameter_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Create agents with unique names
    agents = {}
    for idx, params in enumerate(hyperparameter_combinations):
        agent_name = f"MCTSAgent_"
        for param, value in params.items():
            agent_name += f"{param}={value}, "
        
        agents[agent_name] = params

    # Generate all possible matchups between agents
    agent_names = list(agents.keys())
    games = list(combinations(agent_names, 2))

    run(games, agents)
