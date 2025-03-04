from minesweeper_solver import MinesweeperSolver
#from utils import compare_methods




def compare_methods(solved_board):
    from minesweeper_solver import MinesweeperSolver
    """
        Compare the performance of different Minesweeper solving algorithms.
        Executes Backtracking, Fail First Principle, Forward Checking, Backjumping,
        and Generalized Arc Consistency (GAC4) on the provided solved board.

        Parameters:
            solved_board (list of lists): The fully solved Minesweeper board
                                          (used as ground truth).

        Returns:
            None: Prints the results of each algorithm to the console.
        """

    solver = MinesweeperSolver(solved_board)
    print("The solution of the Game:")
    display_board(solved_board)

    print("\nSolving with Backtracking:")
    bt_result = solver.solve_with_backtracking()
    display_board(bt_result["player_board"])
    print(f"Time Taken: {bt_result['time_taken']} seconds")
    print(f"Assignments Made: {bt_result['assignments']}")
    print(f"Flags Correctly Placed: {len(bt_result['flags'])} flags")
    print(f"Status: {bt_result['status']}")

    print("\nSolving with Fail First Principle (FFP):")
    ffp_result = solver.solve_with_ffp()
    display_board(ffp_result["player_board"])
    print(f"Time Taken: {ffp_result['time_taken']} seconds")
    print(f"Assignments Made: {ffp_result['assignments']}")
    print(f"Flags Correctly Placed: {len(ffp_result['flags'])} flags")
    print(f"Status: {ffp_result['status']}")

    print("\nSolving with Forward Checking (FC1):")
    fc1_result = solver.solve_with_fc1()
    display_board(fc1_result["player_board"])
    print(f"Time Taken: {fc1_result['time_taken']} seconds")
    print(f"Assignments Made: {fc1_result['assignments']}")
    print(f"Flags Correctly Placed: {len(fc1_result['flags'])} flags")
    print(f"Status: {fc1_result['status']}")

    print("\nSolving with Backjumping:")
    bj_result = solver.solve_with_backjumping()
    display_board(bj_result["player_board"])
    print(f"Time Taken: {bj_result['time_taken']} seconds")
    print(f"Assignments Made: {bj_result['assignments']}")
    print(f"Flags Correctly Placed: {len(bj_result['flags'])} flags")
    print(f"Status: {bj_result['status']}")

    print("\nSolving with GAC4:")
    gac4_result = solver.solve_with_gac4()
    display_board(gac4_result["player_board"])
    print(f"Time Taken: {gac4_result['time_taken']} seconds")
    print(f"Assignments Made: {gac4_result['assignments']}")
    print(f"Flags Correctly Placed: {len(gac4_result['flags'])} flags")
    print(f"Status: {gac4_result['status']}")


def display_board(board):
    """
    Display the Minesweeper board in a readable format.

    Parameters:
        board (list of lists): The Minesweeper board to display, where:
            - '?' represents unrevealed cells.
            - 'F' represents flagged mines.
            - Integers represent the number of adjacent mines.

    Returns:
        None: Prints the board row by row.
    """
    for row in board:  # Iterate through each row of the board
        # Convert each cell to a string and join them with spaces for neat formatting
        print(' '.join(str(cell) for cell in row))



if __name__ == "__main__":
    solved_board1 = [
        [0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, -1, 1, 0],
        [0, 0, 1, -1, 1, 1, 2, 3, 2],
        [1, 1, 1, 1, 1, 0, 2, -1, -1],
        [-1, 1, 0, 0, 0, 1, 3, -1, 3],
        [2, 2, 1, 0, 0, 1, -1, 2, 1],
        [1, -1, 2, 1, 1, 1, 1, 1, 0],
        [1, 1, 2, -1, 2, 1, 0, 0, 0],
        [0, 0, 1, 2, -1, 1, 0, 0, 0]
    ]

    solved_board2 = [
        [0, 1, 1, 1, 1, -1, 1, 0, 0],
        [0, 1, -1, 1, 1, 1, 1, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 0, 0],
        [0, 1, -1, 1, 1, 2, 2, 1, 0],
        [0, 1, 2, 3, 3, -1, -1, 1, 0],
        [0, 0, 1, -1, -1, 3, 2, 2, 1],
        [0, 0, 2, 3, 3, 1, 0, 2, -1],
        [0, 0, 1, -1, 1, 0, 0, 2, -1],
        [0, 0, 1, 1, 1, 0, 0, 1, 1]
    ]

    solved_board3 = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 0, 1, -1, 1],
        [0, 2, -1, 2, 0, 0, 1, 1, 1],
        [0, 2, -1, 2, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 2, 2],
        [0, 0, 0, 0, 0, 1, 2, -1, -1],
        [0, 0, 1, 1, 1, 1, -1, 3, 2],
        [1, 1, 2, -1, 2, 2, 1, 2, 1],
        [1, -1, 2, 2, -1, 1, 0, 1, -1]
    ]

    solved_board4 = [
        [-1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, -1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, -1, 1, 0],
        [0, 0, 0, 0, 0, 1, 2, 3, 2],
        [0, 0, 0, 0, 0, 0, 1, -1, -1],
        [0, 0, 1, 1, 1, 0, 1, 3, -1],
        [1, 1, 1, -1, 3, 2, 1, 1, 1],
        [-1, 1, 1, 2, -1, -1, 1, 0, 0, ]
    ]

    compare_methods(solved_board4)
