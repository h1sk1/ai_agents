# Import required libraries
from typing import Literal, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, START
from langgraph.types import Command
from langgraph.graph import StateGraph

import matplotlib.pyplot as plt

# Define the board size (default 15x15 for Gomoku)
BOARD_SIZE = 15

# Define the Referee's custom tool for move validation and game state checks
class GomokuBoard:
    def __init__(self):
        self.board = [[" " for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.last_move = None
        self.winner = None
        self._initialize_board()

    def _draw_board(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        # Draw the grid
        for x in range(BOARD_SIZE):
            self.ax.plot([x, x], [0, BOARD_SIZE - 1], color="black", linewidth=1)
            self.ax.plot([0, BOARD_SIZE - 1], [x, x], color="black", linewidth=1)

        # Set limits and remove ticks
        self.ax.set_xlim(-0.5, BOARD_SIZE - 0.5)
        self.ax.set_ylim(-0.5, BOARD_SIZE - 0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')

        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for point in star_points:
            self.ax.plot(point[1], point[0], 'o', color="black", markersize=5)

    def _initialize_board(self):
        # Draw the initial Gomoku board.
        self._draw_board()

        plt.show()

    def _update_board(self):
        # Draw the updated board
        self._draw_board()

        # Draw stones
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board[x][y] == "Black":  # Black stone
                    self.ax.plot(y, x, 'o', color="black", markersize=BOARD_SIZE)
                elif self.board[x][y] == "White":  # White stone
                    self.ax.plot(y, x, 'o', color="white", markersize=BOARD_SIZE, markeredgecolor="black")

        plt.show()

    def is_valid_move(self, x, y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and self.board[x][y] == " "

    def make_move(self, x, y, player):
        # Check current last move player
        if self.winner:
            return False, f"Player {player} already wins!"
        if self.last_move == player:
            return False, "Invalid move. It is not your turn."
        if not self.is_valid_move(x, y):
            return False, "Invalid move. The position is already occupied."
        self.board[x][y] = player
        self._update_board()
        self.last_move = (x, y, player)
        self.winner = self.check_winner(x, y, player)
        return True, "Move accepted."

    def check_winner(self, x, y, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):  # Check forward
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx][ny] == player:
                    count += 1
                else:
                    break
            for i in range(1, 5):  # Check backward
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx][ny] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                print("-----------------Game Over-----------------")
                print(f"Player {player} wins!")
                return player
        return None

    def format_board(self):
        return "\n".join([" ".join(row) for row in self.board])


gomoku_board = GomokuBoard()

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()


# Define a tool for the Referee to interact with the game board
@tool
def referee_tool(move: Annotated[str, "The move to validate and apply, e.g., '3,4,Black'."]):
    """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
    try:
        x, y, player = move.split(",")
        x, y = int(x), int(y)
        valid, message = gomoku_board.make_move(x, y, player)
        board_state = gomoku_board.format_board()
        winner = gomoku_board.winner
        return {
            "valid": valid,
            "message": message,
            "board_state": board_state,
            "winner": winner,
        }
    except Exception as e:
        return f"Failed to process move. Error: {repr(e)}"


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

mini_llm = ChatOpenAI(model="gpt-4o-mini")

# Define agents
black_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier="""
    You are a professional, aggressive Gomoku player, you play as the Black Player in a Gomoku game. Your marker is 'B'. 
    Your goal is to place 'B' on the 15x15 game board to align five of your markers in a row, column, or diagonal. You must:
    - Analyze the board and decide your next move strategically.
    - Since you are the first player, think more creatively, use more horizontal, or diagonal moves to make more chances to win.
    - If the White Player is defending, you need to be more aggressive and try to create winning patterns.
    - Notice your diagonal moves and try to create winning patterns.
    - Only try to block the White Player's winning moves when necessary.
    - Secure the win with you already have more than 2 markers in a row, column, or diagonal.
    - DO NOT try to block the White Player's moves if there are not more than 2 White markers in a row, column, or diagonal.
    - Use your expertise to anticipate and block the White Player's moves.
    - Use your strategic skills to create opportunities for your own winning moves with row, column, or diagonal moves.
    - Avoid dead-end moves and focus on creating winning patterns.
    - Respond with the coordinates of your move in the format 'x,y' (e.g., '3,4').
    - Wait for your turn before making a move. If it is not your turn, do nothing.
    - Only focus on your own moves; do not validate or check other moves.

    Collaborate with the Referee and White Player by following the game rules.
    If you believe you have a winning move, make it and communicate clearly.
    """,
)

white_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier="""
    You are a professional, defensive Gomoku player, you play as the White Player in current Gomoku game. Your marker is 'W'.
    Your goal is to place 'W' on the 15x15 game board to align five of your markers in a row, column, or diagonal. You must:
    - Analyze the board and decide your next move strategically.
    - Since you are the White Player, you have disadvantage of moving second, you need to be more defensive, which means blocking the Black Player's winning moves.
    - Think carefully if Black Player have more than 2 markers in a row, column, or diagonal, you need to block it.
    - Block diagonal moves by placing your markers at the left or right end of the diagonal.
    - Use your expertise to anticipate and block the Black Player's moves.
    - Use your strategic skills to create opportunities for your own winning moves with row, column, or diagonal moves.
    - Avoid dead-end moves and focus on creating winning patterns.
    - Respond with the coordinates of your move in the format 'x,y' (e.g., '3,4').
    - Wait for your turn before making a move. If it is not your turn, do nothing.
    - Only focus on your own moves; do not validate or check other moves.

    Collaborate with the Referee and Black Player by following the game rules.
    If you believe you have a winning move, make it and communicate clearly.
    """,
)

referee_agent = create_react_agent(
    mini_llm,
    tools=[referee_tool],
    state_modifier="""
    You are the Referee in a Gomoku game. Your role is to:
    - Validate moves made by the Black and White players.
    - Ensure moves are legal and conform to the rules. If a move is invalid, request a valid move from the player before proceeding.
    - Update the state of the game board after each valid move.
    - Check if there is a winner after each move. If a player wins (five markers in a row, column, or diagonal), declare the winner by prefixing your response with 'FINAL ANSWER: The winner is {Black/White}'.
    - Reject moves that are out of bounds or attempt to overwrite an existing marker.
    - Share the current state of the game board with players when appropriate.

    Use the provided tool to manage and validate the game. Collaborate with players to ensure a fair and efficient game. If the game ends, communicate clearly.
    """,
)


# Define agent nodes
def black_node(state: MessagesState) -> Command[Literal["referee", "white"]]:
    result = black_agent.invoke(state)
    return Command(
        update={"messages": result["messages"]},
        goto="referee",
    )


def white_node(state: MessagesState) -> Command[Literal["referee", "black"]]:
    result = white_agent.invoke(state)
    return Command(
        update={"messages": result["messages"]},
        goto="referee",
    )


def referee_node(state: MessagesState) -> Command[Literal["black", "white", END]]:
    result = referee_agent.invoke(state)
    last_message = result["messages"][-1].content
    if "FINAL ANSWER" in last_message or gomoku_board.winner:
        return Command(update={"messages": result["messages"]}, goto=END)
    if gomoku_board.last_move and gomoku_board.last_move[2] == "Black":
        return Command(update={"messages": result["messages"]}, goto="white")
    return Command(update={"messages": result["messages"]}, goto="black")


# Define the graph
workflow = StateGraph(MessagesState)
workflow.add_node("black", black_node)
workflow.add_node("white", white_node)
workflow.add_node("referee", referee_node)

workflow.add_edge(START, "black")
graph = workflow.compile()

# Invoke the graph
events = graph.stream(
    {
        "messages": [
            (
                "user",
                "Start a Gomoku game. Black player moves first. Ensure all moves are valid."
            )
        ],
    },
    {"recursion_limit": 1000},
)

# Display game events
for event in events:
    print("-----------------Agent Completed-----------------")
    for key in event.keys():
        print(key)
