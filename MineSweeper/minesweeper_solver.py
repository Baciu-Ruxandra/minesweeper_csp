import itertools
import random
import time


class CSPHandler:
    def __init__(self):
        self.progress = False  # Tracks if progress is made during constraint simplification

    @staticmethod
    def calculate_probabilities(constraints):
        """
        Calculate probabilities of each cell being a mine based on constraints.
        """
        probabilities = {}
        for vars_set, count in constraints:
            for var in vars_set:
                probabilities[var] = probabilities.get(var, 0) + count / len(vars_set)
        return probabilities


    def process_constraints(self, solver):
        """
        Simplifies and resolves constraints in the CSP.
        Flags cells as mines or reveals safe cells when constraints are resolved.
        """
        self.progress = False  # Reset progress tracking
        new_constraints = []  # Store unresolved constraints after simplification
        for vars_set, count in solver.constraints:
            # Count how many variables are already flagged as mines
            flagged = sum(1 for var in vars_set if var in solver.flags)
            # Exclude already revealed or flagged variables from the constraint
            vars_set = [var for var in vars_set if var not in solver.revealed and var not in solver.flags]
            # Adjust the constraint count based on flagged variables
            count -= flagged

            if count == 0:
                # If no more mines remain, mark all variables as safe
                for var in vars_set:
                    solver.reveal(var[0], var[1])  # Reveal the safe cell
                    self.progress = True  # Progress was made
            elif count == len(vars_set):
                # If all remaining variables are mines, flag them
                for var in vars_set:
                    solver.flags.add(var)
                    self.progress = True
            else:
                # Add unresolved constraints back to the list
                new_constraints.append((vars_set, count))
        # Update the solver's constraints with unresolved ones
        solver.constraints = new_constraints
    def guess(self, solver):
        """
        Makes a probabilistic guess for the safest cell based on constraints.
        Falls back to a random guess if no probabilities are available.
        """
        probabilities = self.calculate_probabilities(solver.constraints)  # Compute probabilities for each cell
        if probabilities:
            # Pick the cell with the lowest probability of being a mine
            best_guess = min(probabilities, key=probabilities.get)
            solver.reveal(*best_guess)  # Reveal the safest cell
        else:
            # If no probabilities, randomly choose a cell
            for x in range(solver.rows):
                for y in range(solver.cols):
                    if solver.player_board[x][y] == '?':  # Unrevealed cell
                        solver.reveal(x, y)
                        return


    def forward_checking(self, solver, is_timed_out):
        """
        Solves Minesweeper using the Forward Checking algorithm.
        Propagates constraints and ensures all cells are resolved or flagged.
        """
        # Initialize domains: Each cell can be 'safe' or 'mine' if not revealed
        domains = {
            (x, y): ['safe', 'mine'] for x in range(solver.rows) for y in range(solver.cols)
            if solver.player_board[x][y] == '?'
        }

        def forward_check(var, value):
            """
            Updates the domains of neighboring variables after assigning a value to a cell.
            Returns False if an inconsistency is detected.
            """
            if is_timed_out():
                print("Timeout occurred.")
                return False  # Stop if timeout occurs

            x, y = var  # Cell coordinates
            for nx, ny in MinesweeperSolver.get_neighbors(x, y, solver.rows, solver.cols):
                if (nx, ny) in domains and value in domains[(nx, ny)]:
                    domains[(nx, ny)].remove(value)  # Remove inconsistent value
                    if not domains[(nx, ny)]:  # If domain is empty, inconsistency detected
                        return False
            return True

        def assign_var(var, value):
            """
            Assigns a value to a variable and propagates constraints to neighbors.
            """
            if value == 'mine':
                solver.flags.add(var)  # Flag the cell as a mine
            elif value == 'safe':
                solver.reveal(var[0], var[1])  # Reveal the cell
            return forward_check(var, value)

        def propagate_constraints():
            """
            Simplifies constraints and propagates safe assignments or flags.
            Repeats until no progress is made.
            """
            while True:
                previous_revealed = len(solver.revealed)  # Track revealed cells before iteration
                previous_flags = len(solver.flags)  # Track flagged cells
                solver.flag_mines_and_reveal()  # Use solver logic to process constraints
                # If no new cells revealed or flagged, stop propagation
                if len(solver.revealed) == previous_revealed and len(solver.flags) == previous_flags:
                    break

        def choose_next_variable():
            """
            Chooses the next variable with the smallest domain size (most constrained).
            """
            return min(domains.keys(), key=lambda var: len(domains[var]), default=None)

        def backtrack():
            """
            Recursive backtracking algorithm to solve the CSP.
            """
            if solver.is_solved():  # If the game is solved, stop recursion
                return True
            if not domains:  # If no domains are left, backtracking fails
                return False

            var = choose_next_variable()  # Select the most constrained variable
            if not var:
                return solver.is_solved()  # If no variable left, check if solved

            domain_copy = domains.copy()  # Backup current domains
            for value in domains[var]:
                if assign_var(var, value):  # Assign a value to the variable
                    propagate_constraints()  # Propagate constraints
                    if backtrack():  # Recursive call
                        return True
                # Undo assignment if it led to a dead end
                if value == 'mine':
                    solver.flags.discard(var)
                elif value == 'safe':
                    solver.revealed.discard(var)
                    solver.player_board[var[0]][var[1]] = '?'
                domains.update(domain_copy)  # Restore domains

            return False  # No solution found

        solver.initialize_game()  # Initialize the game with the first safe move
        propagate_constraints()  # Start propagating constraints
        backtrack()  # Begin backtracking
        propagate_constraints()  # Final constraint propagation after solving

        # Finalize the board: Replace '?' with 'F' for mines and reveal numbers
        for x in range(solver.rows):
            for y in range(solver.cols):
                if solver.player_board[x][y] == '?' or solver.player_board[x][y] == -1:
                    if (x, y) in solver.flags:
                        solver.player_board[x][y] = 'F'  # Flag mine

        return True  # Game solved successfully

    def backtracking(self, solver, is_timed_out):
        """
        Solve Minesweeper using Backtracking.
        """
        # Keep track of visited states to avoid redundant exploration
        visited_states = set()

        def state_key():
            """
            Generate a unique state key for the solver's current state.
            Combines revealed cells and flagged cells as a hashable state.
            """
            return (frozenset(solver.revealed), frozenset(solver.flags))

        def is_safe_to_reveal(x, y):
            """
            Check if revealing (x, y) violates any constraints.
            Ensures that the number of mines in the neighborhood matches constraints.
            """
            for nx, ny in MinesweeperSolver.get_neighbors(x, y, solver.rows, solver.cols):
                if isinstance(solver.player_board[nx][ny], int):  # If this is a numbered cell
                    flagged_neighbors = sum(  # Count how many neighbors are flagged
                        1 for fx, fy in MinesweeperSolver.get_neighbors(nx, ny, solver.rows, solver.cols) if (fx, fy) in solver.flags
                    )
                    unrevealed_neighbors = sum(  # Count how many neighbors are still unrevealed
                        1 for rx, ry in MinesweeperSolver.get_neighbors(nx, ny, solver.rows, solver.cols) if
                        solver.player_board[rx][ry] == '?'
                    )
                    remaining_mines = solver.player_board[nx][ny] - flagged_neighbors  # Mines left to account for
                    # If remaining mines exceed unrevealed cells or are negative, the move is unsafe
                    if remaining_mines > unrevealed_neighbors or remaining_mines < 0:
                        return False
            return True  # Safe to reveal

        def backtrack():
            """
            Recursive backtracking to solve the game.
            Tries revealing cells and flagging mines based on constraints.
            """
            if is_timed_out():  # Check if timeout has occurred
                return False

            if solver.is_solved():  # If the game is solved, end recursion
                return True

            current_state = state_key()  # Get the current state key
            if current_state in visited_states:  # Skip already visited states
                return False
            visited_states.add(current_state)  # Mark the state as visited

            solver.flag_mines_and_reveal()  # Use solver logic to flag mines and reveal cells

            # Explore all unrevealed cells
            for x in range(solver.rows):
                for y in range(solver.cols):
                    if solver.player_board[x][y] == '?':  # Unrevealed cell
                        # Try revealing the cell
                        if is_safe_to_reveal(x, y):  # Check if it's safe
                            solver.reveal(x, y)
                            if backtrack():  # Recurse with the new state
                                return True
                            # Undo the reveal if it leads to a dead end
                            solver.revealed.discard((x, y))
                            solver.player_board[x][y] = '?'

                        # Try flagging the cell as a mine
                        solver.flags.add((x, y))
                        if backtrack():  # Recurse with the new state
                            return True
                        # Undo the flag if it leads to a dead end
                        solver.flags.discard((x, y))

            return False  # No solution found in this branch

        solver.initialize_game()  # Start the game with an initial reveal
        backtrack()  # Start the recursive backtracking process

        # Finalize the board by marking flagged mines as 'F'
        for x in range(solver.rows):
            for y in range(solver.cols):
                if solver.player_board[x][y] == '?' or solver.player_board[x][y] == -1:
                    if (x, y) in solver.flags:
                        solver.player_board[x][y] = 'F'
        return True  # Return success

    def gac4(self, solver, is_timed_out):
        """
        Solve Minesweeper using Generalized Arc Consistency (GAC4).
        """

        def initialize_arcs():
            """
            Initialize arcs for all numbered cells in the grid.
            Each arc represents a constraint between a numbered cell and its unrevealed neighbors.
            """
            arcs = []
            for x in range(solver.rows):
                for y in range(solver.cols):
                    if isinstance(solver.player_board[x][y], int):  # If this is a numbered cell
                        neighbors = MinesweeperSolver.get_neighbors(x, y, solver.rows, solver.cols)
                        unknowns = [(nx, ny) for nx, ny in neighbors if solver.player_board[nx][ny] == '?']
                        if unknowns:  # If there are unrevealed neighbors
                            arcs.append(((x, y), unknowns, solver.player_board[x][y]))
            return arcs

        def enforce_arc_consistency(arc):
            """
            Enforces arc consistency for a given arc by adjusting the domains of neighbors.
            """
            if is_timed_out():  # Stop if timeout occurs
                print("Timeout occurred.")
                return False

            constraint_cell, neighbors, constraint_value = arc
            flagged = sum(1 for cell in neighbors if cell in solver.flags)  # Count flagged neighbors
            remaining_neighbors = [cell for cell in neighbors if
                                   cell not in solver.revealed and cell not in solver.flags]
            remaining_constraint = constraint_value - flagged  # Remaining mines to be found

            if remaining_constraint < 0:  # Inconsistent constraint
                return False

            if remaining_constraint == 0:  # If no more mines, all neighbors are safe
                for cell in remaining_neighbors:
                    solver.reveal(cell[0], cell[1])
                return True

            if remaining_constraint == len(remaining_neighbors):  # If all neighbors are mines, flag them
                for cell in remaining_neighbors:
                    solver.flags.add(cell)
                return True

            return False  # No updates made

        def propagate_arcs():
            """
            Propagates arc consistency until no more changes can be made.
            """
            arcs = initialize_arcs()  # Create initial arcs
            progress = True
            while progress:
                progress = False
                for arc in list(arcs):  # Iterate over arcs
                    if enforce_arc_consistency(arc):  # Apply arc consistency
                        progress = True  # Track progress
                        arcs.remove(arc)  # Remove resolved arc
            return True

        solver.initialize_game()  # Start the game with an initial reveal
        propagate_arcs()  # Apply arc consistency initially

        # Continue solving until all cells are resolved
        while any(solver.player_board[x][y] == '?' for x in range(solver.rows) for y in range(solver.cols)):
            self.guess(solver)  # Guess if necessary
            propagate_arcs()  # Apply arc consistency after each guess

        # Finalize the board by marking flagged mines as 'F'
        for x in range(solver.rows):
            for y in range(solver.cols):
                if solver.player_board[x][y] == '?' or solver.player_board[x][y] == -1:
                    if (x, y) in solver.flags:
                        solver.player_board[x][y] = 'F'
        return True  # Return success

    def backjumping(self, solver, is_timed_out):
        """
        Solve Minesweeper using Dependency-Directed Backtracking (Backjumping).
        """
        # Track visited states to avoid redundancy
        visited_states = set()
        # Store dependencies causing backtracking for each variable
        dependencies = {}

        def state_key():
            """
            Generate a unique, hashable key for the current state.
            """
            return (frozenset(solver.revealed), frozenset(solver.flags))

        def backjump():
            """
            Perform the backjumping process.
            Returns the variable to backtrack to based on dependency analysis.
            """
            while assignments:
                last_var = assignments.pop()  # Get the most recent assignment
                if last_var not in dependencies or not dependencies[last_var]:  # No dependencies
                    continue
                return last_var  # Jump to the most recent dependency
            return None  # No valid variable to backjump to

        def propagate_constraints():
            """
            Simplify constraints and propagate changes using the solver's logic.
            """
            solver.flag_mines_and_reveal()

        def choose_next_variable():
            """
            Select the next variable (unrevealed cell) to explore.
            """
            unrevealed = [
                (x, y) for x in range(solver.rows) for y in range(solver.cols) if solver.player_board[x][y] == '?'
            ]
            if not unrevealed:
                return None
            # Choose variable with the smallest number of constraints (most likely to fail)
            return min(unrevealed, key=lambda var: len(MinesweeperSolver.get_neighbors(var[0], var[1], solver.rows, solver.cols)))

        assignments = []  # Track current variable assignments

        def backjump_recursive():
            """
            Recursive function implementing backjumping logic.
            """
            if is_timed_out():  # Stop recursion on timeout
                print("Timeout occurred.")
                return False

            if solver.is_solved():  # Stop recursion if solved
                return True

            current_state = state_key()  # Get the current state
            if current_state in visited_states:  # Skip already visited states
                return False
            visited_states.add(current_state)  # Mark state as visited

            propagate_constraints()  # Simplify constraints

            next_var = choose_next_variable()  # Choose the next variable to explore
            if not next_var:
                return solver.is_solved()

            # Try assigning values ('safe' or 'mine') to the variable
            for value in ['safe', 'mine']:
                if value == 'mine':
                    solver.flags.add(next_var)  # Flag the cell as a mine
                elif value == 'safe':
                    solver.reveal(next_var[0], next_var[1])  # Reveal the cell

                propagate_constraints()  # Propagate constraints after assignment

                if backjump_recursive():  # Recurse with the new state
                    return True

                # Undo the assignment if it leads to a dead end
                if value == 'mine':
                    solver.flags.discard(next_var)
                elif value == 'safe':
                    solver.revealed.discard(next_var)
                    solver.player_board[next_var[0]][next_var[1]] = '?'

            # Perform backjumping if a failure occurs
            backjump_var = backjump()
            if backjump_var:
                solver.revealed.discard(backjump_var)
                solver.player_board[backjump_var[0]][backjump_var[1]] = '?'
            return False

        solver.initialize_game()  # Start with an initial reveal
        backjump_recursive()  # Begin the recursive backjumping process

        # Finalize the board by marking flagged mines as 'F'
        for x in range(solver.rows):
            for y in range(solver.cols):
                if solver.player_board[x][y] == '?' or solver.player_board[x][y] == -1:
                    if (x, y) in solver.flags:
                        solver.player_board[x][y] = 'F'

        return True  # Return success

    def fail_first_principle(self, solver, is_timed_out):
        """
        Solve Minesweeper using the Fail First Principle (FFP).
        This method prioritizes resolving cells with the highest constraints and the lowest probability of being a mine.
        """

        def calculate_probabilities():
            """
            Calculate probabilities for all unrevealed cells based on the current constraints.
            Each cell is assigned a probability of being a mine.
            """
            probabilities = {}
            for vars_set, count in solver.constraints:  # Iterate through each constraint
                for var in vars_set:  # For each cell in the constraint
                    # Accumulate the probability of being a mine
                    probabilities[var] = probabilities.get(var, 0) + count / len(vars_set)
            return probabilities  # Return the calculated probabilities for all cells

        def choose_next_cell(probabilities):
            """
            Choose the next cell to process:
            - Prioritize cells with the lowest probability of being a mine.
            - Among those, prioritize cells with the highest number of constraints (most neighbors).
            """
            if probabilities:  # If probabilities are calculated
                return min(
                    probabilities.keys(),  # Iterate over all cells
                    key=lambda cell: (  # Sort by:
                        probabilities[cell],  # 1. Lowest mine probability
                        len(MinesweeperSolver.get_neighbors(cell[0], cell[1], solver.rows, solver.cols))  # 2. Highest constraints
                    )
                )
            return None  # If no probabilities are available, return None

        # Start the game by initializing it (reveal the first safe cell)
        solver.initialize_game()

        # Keep solving the game until all cells are resolved or timeout occurs
        while not solver.is_solved():
            if is_timed_out():  # Check if the timeout limit has been reached
                return False  # Stop solving if timed out

            # Process constraints to propagate information (flag mines and reveal safe cells)
            solver.flag_mines_and_reveal()

            # Calculate probabilities for unresolved cells
            probabilities = calculate_probabilities()

            # Use the FFP heuristic to choose the next cell to reveal
            next_cell = choose_next_cell(probabilities)

            if next_cell:  # If a cell was chosen
                solver.reveal(next_cell[0], next_cell[1])  # Reveal the chosen cell
            else:
                # If no probabilities are available, fall back to a random guess
                unrevealed = [
                    (x, y) for x in range(solver.rows) for y in range(solver.cols)
                    if solver.player_board[x][y] == '?'  # Find all unrevealed cells
                ]
                if unrevealed:  # If there are still unrevealed cells
                    random_cell = random.choice(unrevealed)  # Randomly pick one
                    solver.reveal(random_cell[0], random_cell[1])  # Reveal the chosen cell

        # Finalize the board by marking flagged cells as 'F' and replacing unresolved cells
        for x in range(solver.rows):
            for y in range(solver.cols):
                if solver.player_board[x][y] == '?' or solver.player_board[x][
                    y] == -1:  # Unresolved or placeholder cells
                    if (x, y) in solver.flags:  # If the cell is flagged as a mine
                        solver.player_board[x][y] = 'F'  # Mark it as a flagged mine

        return True  # Return success



    def calculate_probabilities(self,constraints):
        """
        Calculate the probabilities of each unrevealed cell being a mine
        based on the current set of constraints.

        Parameters:
            constraints (list): A list of tuples, where each tuple consists of:
                - vars_set (list): A list of cells involved in the constraint.
                - count (int): The number of mines in those cells.

        Returns:
            dict: A dictionary where keys are cell coordinates (x, y) and
                  values are the calculated probabilities of those cells being mines.
        """
        probabilities = {}  # Initialize an empty dictionary to store probabilities
        for vars_set, count in constraints:  # Iterate through each constraint
            for var in vars_set:  # For each cell in the constraint
                # Add the probability contribution from this constraint to the cell
                # Probability contribution = (mines count) / (number of cells in vars_set)
                probabilities[var] = probabilities.get(var, 0) + count / len(vars_set)
        return probabilities  # Return the dictionary of probabilities






class MinesweeperSolver:
    def __init__(self, solved_board):
        """
        Initializes the MinesweeperSolver with the solved board and sets up the player's board.
        """
        self.solved_board = solved_board  # The actual solution of the game
        self.rows = len(solved_board)  # Number of rows in the board
        self.cols = len(solved_board[0])  # Number of columns in the board
        self.player_board = [['?' for _ in range(self.cols)] for _ in range(self.rows)]  # Player's view of the board
        self.flags = set()  # Set to track flagged cells (mines)
        self.revealed = set()  # Set to track revealed cells
        self.constraints = []  # List of constraints for CSP solving
        self.assignments = 0  # Number of cells revealed or flagged
        self.csp_handler = CSPHandler()  # Initialize the CSPHandler

    @staticmethod
    def get_neighbors(x, y, rows, cols):
        """
        Find all valid neighbors for a cell (x, y) in a grid of given dimensions.
        """
        return [
            (nx, ny)
            for dx, dy in itertools.product([-1, 0, 1], repeat=2)
            if (dx != 0 or dy != 0)
               and 0 <= (nx := x + dx) < rows
               and 0 <= (ny := y + dy) < cols
        ]


    def initialize_game(self):
        """
        Reveals the first safe cell to start the game.
        """
        for x in range(self.rows):
            for y in range(self.cols):
                if self.solved_board[x][y] != -1:  # If cell is not a mine
                    self.reveal(x, y)
                    return


    def reveal(self, x, y):
        """
        Reveals a cell and recursively reveals neighbors if the cell is empty (0).
        """
        if (x, y) in self.revealed or self.player_board[x][y] != '?':  # Ignore already revealed cells
            return
        self.player_board[x][y] = self.solved_board[x][y]  # Update player's board with the actual value
        self.revealed.add((x, y))  # Mark cell as revealed
        if self.solved_board[x][y] == 0:  # If the cell is empty, reveal its neighbors
            for nx, ny in MinesweeperSolver.get_neighbors(x, y, self.rows, self.cols):
                self.reveal(nx, ny)


    def add_constraint(self, x, y):
        """
        Adds a constraint for a numbered cell based on its unrevealed neighbors.
        """
        neighbors = MinesweeperSolver.get_neighbors(x, y, self.rows, self.cols)  # Get all neighboring cells
        unknowns = [(nx, ny) for nx, ny in neighbors if self.player_board[nx][ny] == '?']  # Unrevealed neighbors
        if unknowns:
            self.constraints.append((unknowns, self.player_board[x][y]))  # Add constraint: neighbors and mine count

    def flag_mines_and_reveal(self):
        """
        Iteratively flags mines and reveals safe cells using constraints.
        """
        for x in range(self.rows):
            for y in range(self.cols):
                if isinstance(self.player_board[x][y], int) and self.player_board[x][y] > 0:
                    self.add_constraint(x, y)  # Add constraints for numbered cells
        self.csp_handler.process_constraints(self)  # Use CSP handler to process constraints

    def finalize_board(self):
        """
        Finalize the player's board: convert all unresolved cells and mines.
        """
        for x in range(self.rows):
            for y in range(self.cols):
                if self.player_board[x][y] == '?' or self.player_board[x][y] == -1:
                    if (x, y) in self.flags: # If the cell is flagged as a mine
                        self.player_board[x][y] = 'F'  # Mark as flagged
                    else:
                        self.player_board[x][y] = self.solved_board[x][y]  # Revealed number

    def reset_game(self):
        """
        Reset the game state for re-solving.
        """
        self.player_board = [['?' for _ in range(self.cols)] for _ in range(self.rows)]
        self.flags = set()
        self.revealed = set()
        self.constraints = []
        self.assignments = 0

    def is_solved(self):
        """
        Checks if the game is fully solved:
        - All mines must be flagged.
        - All safe cells must be revealed.
        """
        return all(
            (self.solved_board[x][y] == -1 and (x, y) in self.flags) or  # Mine correctly flagged
            (self.solved_board[x][y] != -1 and (x, y) in self.revealed)  # Safe cell revealed
            for x in range(self.rows) for y in range(self.cols)
        )

    def solve_with_backtracking(self, timeout=10):
        """
        Solve Minesweeper using the backtracking algorithm.
        """
        self.reset_game()  # Reset the game state
        start_time = time.time() # Track the start time

        def is_timed_out():
            return time.time() - start_time > timeout # Check if the timeout is exceeded

        # Use the CSPHandler's backtracking method to solve the game
        if self.csp_handler.backtracking(self, is_timed_out):
            self.finalize_board() # Finalize the board after solving
            time_taken = time.time() - start_time # Calculate time taken
            if len(self.flags) == 0:
                # Handle no solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "No solution found",
                }
            if len(self.flags) < 10:
                # Handle inconplete solution case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Incomplete solution",
                }
            if len(self.flags) == 10:
                # Handle solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Solution found",
                }
        else:
            # Timeout occurred
            return {
                "player_board": self.player_board,
                "flags": self.flags,
                "time_taken": timeout,
                "assignments": len(self.revealed) + len(self.flags),
                "status": "Timeout",
            }


    def solve_with_fc1(self, timeout=10):
        """
        Solve Minesweeper using the Forward Checking (FC1) algorithm.
        Stops if the solution is not found within the specified timeout (in seconds).
        """
        self.reset_game()  # Reset the game state
        start_time = time.time()  # Start timing

        def is_timed_out():
            return time.time() - start_time > timeout # Check if the timeout is exceeded

        # Use the CSPHandler's forward checking method
        if self.csp_handler.forward_checking(self, is_timed_out):
            self.finalize_board() # Finalize the board after solving
            time_taken = time.time() - start_time # Calculate time taken
            if len(self.flags) == 0:
                # Handle no solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "No solution found",
                }
            if len(self.flags) < 10:
                # Handle incomplete solution case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Incomplete solution",
                }
            if len(self.flags) == 10:
                # Handle solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Solution found",
                }
        else:
            # Timeout occurred
            return {
                "player_board": self.player_board,
                "flags": self.flags,
                "time_taken": timeout,
                "assignments": len(self.revealed) + len(self.flags),
                "status": "Timeout",
            }

    def solve_with_gac4(self, timeout=10):
        """
        Solve Minesweeper using Generalized Arc Consistency (GAC4).
        Stops if the solution is not found within the specified timeout (in seconds).
        """
        self.reset_game()  # Reset the game state
        start_time = time.time()  # Start timing

        # Use the CSPHandler's gac4 method
        def is_timed_out():
            return time.time() - start_time > timeout  # Check for timeout

        if self.csp_handler.gac4(self, is_timed_out):
            self.finalize_board() # Finalize the board after solving
            time_taken = time.time() - start_time # Calculate time taken
            if len(self.flags) == 0:
                # Handle no solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "No solution found",
                }
            if len(self.flags) < 10:
                # Handle incomplete solution case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Incomplete solution",
                }
            if len(self.flags) == 10:
                # Handle solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Solution found",
                }
        else:
            # Timeout occurred
            return {
                "player_board": self.player_board,
                "flags": self.flags,
                "time_taken": timeout,
                "assignments": len(self.revealed) + len(self.flags),
                "status": "Timeout",
            }

    def solve_with_backjumping(self, timeout=10):
        """
        Solve Minesweeper using Backjumping.
        Stops if the solution is not found within the specified timeout (in seconds).
        """
        self.reset_game()  # Reset the game state
        start_time = time.time()  # Start timing

        def is_timed_out():
            return time.time() - start_time > timeout # Check for timeout

        # Use the CSPHandler's backjumping method
        if self.csp_handler.backjumping(self, is_timed_out):
            self.finalize_board() # Finalize the board after solving
            time_taken = time.time() - start_time # Calculate time taken
            if len(self.flags) == 0:
                # Handle no solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "No solution found",
                }
            if len(self.flags) < 10:
                # Handle incomplete solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Incomplete solution",
                }
            if len(self.flags) == 10:
                # Handle solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Solution found",
                }
        else:
            # Timeout occurred
            return {
                "player_board": self.player_board,
                "flags": self.flags,
                "time_taken": timeout,
                "assignments": len(self.revealed) + len(self.flags),
                "status": "Timeout",
            }

    def solve_with_ffp(self, timeout=10):
        """
        Solve Minesweeper using the Fail First Principle (FFP).
        Stops if the solution is not found within the specified timeout (in seconds).
        """
        self.reset_game()  # Reset the game state
        start_time = time.time()  # Start timing

        def is_timed_out():
            return time.time() - start_time > timeout # Check for timeout

        # Use the CSPHandler's FFP method
        if self.csp_handler.fail_first_principle(self, is_timed_out):
            self.finalize_board()  # Finalize the board after solving
            time_taken = time.time() - start_time # Calculate time taken
            if len(self.flags) == 0:
                # Handle no solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "No solution found",
                }
            if len(self.flags) < 10:
                # Handle incomplete solution case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Incomplete solution",
                }
            if len(self.flags) == 10:
                # Handle solution found case
                return {
                    "player_board": self.player_board,
                    "flags": self.flags,
                    "time_taken": time_taken,
                    "assignments": len(self.revealed) + len(self.flags),
                    "status": "Solution found",
                }
        else:
            # Timeout occurred
            return {
                "player_board": self.player_board,
                "flags": self.flags,
                "time_taken": timeout,
                "assignments": len(self.revealed) + len(self.flags),
                "status": "Timeout",
            }

