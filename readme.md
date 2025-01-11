# **Maze Solver with Q-Learning**

## **Overview**
This project implements a maze solving agent and uses Q-Learning to solve the maze, it works most of the time, and has more performance spikes the more complicated the maze gets, the agent learns to navigate the maze to get to the end goal, this project uses Python and Pygame to visualize and the logical implementation.

### **Key Features**
- **Maze Generation**: Randomly generates mazes using a depth-first search algorithm.
- **Q-Learning Agent**: Implements a Q-learning algorithm to train an agent to solve the maze.
- **Visualization**: Uses Pygame to visualize the maze, the agent's path, and the learning process.
- **Metrics Tracking**: Tracks and visualizes key metrics such as rewards, success rates, and exploration rates over episodes.

---

## **Installation**

### **Prerequisites**
- Python 3.x
- Pygame
- NumPy
- Matplotlib

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/KunjVPatel/maze-solver-qlearning.git
   cd maze-solver-qlearning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   ```bash
   python maze_solver.py
   ```

---

## **How It Works**

### **Maze Generation**
The maze is generated using a **depth-first search (DFS)** algorithm:
- The maze is represented as a grid of cells.
- Each cell has walls on all four sides initially.
- The algorithm randomly "breaks" walls between cells to create a path, ensuring there is always a solution.

### **Q-Learning Agent**
The agent learns to solve the maze using **Q-learning**:
- The agent explores the maze by taking actions (up, down, left, right).
- It receives rewards for reaching the goal and penalties for hitting walls or revisiting cells.
- The Q-table is updated iteratively to improve the agent's policy.

### **Visualization**
The Pygame interface provides real-time visualization of:
- The maze structure.
- The agent's path.
- The goal (marked in red).
- The agent's current position (marked in green).

### **Metrics Tracking**
The project tracks and visualizes the following metrics over episodes:
- **Total Rewards**: Cumulative rewards per episode.
- **Success Rate**: Whether the agent reached the goal (1) or not (0).
- **Steps per Episode**: Number of steps taken to reach the goal.
- **Exploration Rate (Epsilon)**: Decay of the exploration rate over time.

---

## **Code Structure**
- **`maze_solver.py`**: Main script to run the project.
- **`Cell` Class**: Represents a cell in the maze grid.
- **`Maze` Class**: Generates and manages the maze.
- **`Player` Class**: Represents the agent navigating the maze.
- **`QLearningAgent` Class**: Implements the Q-learning algorithm.
- **`Game` Class**: Manages the Pygame visualization and game loop.
- **`MetricsTracker` Class**: Tracks and visualizes performance metrics.

---

## **Usage**
1. Run the script:
   ```bash
   python maze_solver.py
   ```

2. Watch the agent learn to solve the maze in real-time.

3. After training, the program will display performance metrics:
   - Total rewards per episode.
   - Success rate.
   - Steps per episode.
   - Exploration rate decay.
