# Minesweeper Solver   

## Project Description  
The objective is to uncover all safe cells while avoiding mines. The solver uses **constraint satisfaction techniques** to logically deduce mine placements based on revealed numbers.  

**Game Constraints:**  
1️⃣ A revealed cell with a number `n` indicates `n` adjacent mines.  
2️⃣ These constraints can be represented as sets of variables (unrevealed neighboring cells) and their mine count.  


Constraints dynamically update as the board evolves:  
   - Flagging a mine reduces the mine count in neighboring constraints.  
   - Revealing a safe cell removes it from active constraints.  


---

## 🛠️ Approach & Methods  
The problem is formulated as a **Constraint Satisfaction Problem (CSP)**, where:  
- **Variables** → Unrevealed cells on the grid.  
- **Domains** → Binary values `{Mine, Safe}`.  
- **Constraints** → Minesweeper rules based on revealed numbers.  

### 🔹 **Algorithms Used:**  
✔ **Backtracking Search** 
✔ **Forward Checking**  
✔ **Fail-First Principle (FFP)** 
✔ **Generalized Arc Consistency (GAC)**   
✔ **Backjumping** 

I test these techniques separately or combine them to determine the best approach for solving this game.

### 🔹 **Guessing Strategy:**  
- Initially selects the **top-left corner (0,0)**, as it has the lowest probability of being a mine.  
- If the first choice is incorrect, it selects the first non-mine cell.  
- If no safe moves remain, the solver makes a **probability-based guess** or a **random move**.  

### 🔹 **Timeout Handling:**  
To prevent excessive computation, the solver includes a **timeout mechanism**, returning the best solution found within the allotted time.  

