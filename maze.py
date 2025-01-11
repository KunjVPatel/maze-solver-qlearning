import pygame as pg
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

class Cell:
    """This class represents a cell in the maze grid
    Where X and Y represent the X,Y Coordinate, and walls
    Represent if there is a wall present at Top, Right, 
    Bottom, Left, visited is to check if the path has been
    visited or not """

    # Init function, to define schema of the maze
    def __init__(self, x, y):
        # defining x and y coords with walls, and to check the cell is visited
        self.x = x
        
        self.y = y
        
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
        
        self.visited = False

    # looks for unvisited neighbors of the cell
    def neighbors(self, grid, cols, rows):
        # initiate the for unvisited neighbor
        n = []

        # checks top neighbor
        t = grid.get((self.x, self.y - 1))
        
        # checks right neighbor
        r = grid.get((self.x + 1, self.y))
        
        # checks bottom neighbor
        b = grid.get((self.x, self.y + 1))
        
        # checks left neighbor
        l = grid.get((self.x - 1, self.y))

        # If visits, appends to the list
        if t and not t.visited:
            n.append(t)

        if r and not r.visited:
            n.append(r)

        if b and not b.visited:
            n.append(b)

        if l and not l.visited:
            n.append(l)

        # This randomly selects a neighbor
        if n:
            hold = random.choice(n)
        else:
            hold = None

        return hold
    
        # the logic above is whats always used in maze gen
        # resources:
        # https://medium.com/@uva/build-a-simple-timed-maze-game-with-python-and-pygame-a20c1cea5406
        # https://pythonspot.com/maze-in-pygame/
        # https://electronstudio.github.io/pygame-zero-book/chapters/maze.html
        # https://github.com/HUANGXUANKUN/maze-master

class Maze:
    """This class is used to build the darn maze
    Initiate the maze based on input size, break open
    a path between 2 adjacent walls, and generates maze
    using depth first search"""

    def __init__(self, cols, rows, size):
        # num of cols in the maze    
        self.cols = cols
        
        # num of rows in the maze
        self.rows = rows
        
        # size of cell in the maze
        self.size = size
        
        # this initiates a closed grid of rows by cols
        self.grid = {}
        for y in range(rows):
            for x in range(cols):
                self.grid[(x, y)] = Cell(x, y)

        # this initiates a stack that will be used for back tracking
        self.stack = []
        
        self.current = self.grid[(0, 0)]
        
        # generating the maze
        self.generate()

    def pathwall(self, cur, nxt):
        # this function "breaks" walls between two adjacent cells
        
        # where cur is current cell and nxt is next cell
        
        # we create dx and dy for movement
        dx = cur.x - nxt.x
        
        dy = cur.y - nxt.y
        
        # we look in the direction of the wall to break 
        # based on the relative positions of the cells
        
        # is movement to the right
        if dx == 1:
            # break left wall of current cell
            cur.walls['left'] = False
            # break right wall of next cell
            nxt.walls['right'] = False

        # is movement to the left
        elif dx == -1:
            # break right wall of current cell
            cur.walls['right'] = False
            # break left wall of next cell
            nxt.walls['left'] = False

        # is movement down
        if dy == 1:
            # break top wall of current cell
            cur.walls['top'] = False
            # break bottom wall of next cell
            nxt.walls['bottom'] = False

        # is movement up
        elif dy == -1:
            # break bottom wall of current cell
            cur.walls['bottom'] = False
            # break top wall of next cell
            nxt.walls['top'] = False

    def generate(self):
        # now lets generate based on depth first search
        self.current.visited = True
        
        while True:
            # get an unvisited neighbor for the current cell
            nxt = self.current.neighbors(self.grid, self.cols, self.rows)
        
            if nxt:
                # if there is a neighbor
                # add the current cell to the stack
                # and break the wall between current and next
                
                self.stack.append(self.current)
        
                self.pathwall(self.current, nxt)
        
                self.current = nxt
        
                self.current.visited = True

            elif self.stack:
                # if no unvisited neighbors backtrack by popping from the stack
                self.current = self.stack.pop()

            # end
            else: 
                break

class Player:
    """Player that moves through the maze, is a square"""
    def __init__(self, x, y, size):
        # init the player at a specific x and y
        self.x = x
    
        self.y = y
        # init player based on cell size
        self.size = size
    
        x_pos = x * size + size // 2
    
        y_pos = y * size + size // 2

        # i.e making the player fat
        # yes, I am fat shaming a square
        self.rect = pg.Rect(x_pos, y_pos, size // 2, size // 2)
    
    def move(self, dx, dy, maze):
        # new coords based on movement direction
        nx, ny = self.x + dx, self.y + dy

        # check the new position if its in bound
        if (0 <= nx < maze.cols and 0 <= ny < maze.rows):
            
            # current cell from the maze grid
            cell = maze.grid.get((self.x, self.y))

            # moving left and there is no wall
            if dx == -1:
                if not cell.walls['left']:
                    self.x -= 1

            # moving right and there is no wall
            elif dx == 1:
                if not cell.walls['right']:
                    self.x += 1

            # moving up and there is no wall
            elif dy == -1:
                if not cell.walls['top']:
                    self.y -= 1

            # moving down and there is no wall
            elif dy == 1:
                if not cell.walls['bottom']:
                    self.y += 1

            # update player pos on the screen
            x_pos = self.x * self.size + self.size // 3
            y_pos = self.y * self.size + self.size // 3
            self.rect.x, self.rect.y = x_pos, y_pos

class QLearningAgent:
    """QLearning Agent to solve the maze"""
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, epsilon=1.0):
        # initiating the qlearning agent

        # the maze i spent 4 billion years getting right
        self.maze = maze  

        # table for info
        self.q_table = {}

        # learning rate
        self.learning_rate = learning_rate

        # setting up the future rewards
        self.discount_factor = discount_factor 
        
        # exploration
        self.epsilon = epsilon

        # init the agent
        self.initq()

        # last actions dict
        self.last_actions = {}

    def initq(self):
        # input info base on the cell
        for (x, y), cell in self.maze.grid.items():
            # adding stuff to qtable
            self.q_table[(x, y)] = {action: 0 for action in ['left', 'right', 'up', 'down']}

    def action(self, state):
        # randomly picking
        if np.random.rand() < self.epsilon:
            paction = ['left', 'right', 'up', 'down']

            if state in self.last_actions:
                # to stop going through the same path
                opposite_action = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}.get(self.last_actions[state])
                if opposite_action in paction:
                    paction.remove(opposite_action)
            if not paction:

                # if stuck YOLO pick any direction
                return random.choice(['left', 'right', 'up', 'down'])
            
            return random.choice(paction)
        # choose best action
        return max(self.q_table[state], key=self.q_table[state].get)

    def uqtab(self, state, action, reward, next_state):
        
        # updating q table for better actions in the next iter
        
        best_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        
        td_target = reward + self.discount_factor * self.q_table[next_state][best_action]
        
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.learning_rate * td_error
        


class Game:

    def __init__(self, size):
        # init py game
        pg.init()
        
        # get screen info
        info = pg.display.Info()
        
        # use screen info for full screen
        self.screen = pg.display.set_mode((info.current_w, info.current_h), pg.FULLSCREEN)
        
        # set title of pygame window
        pg.display.set_caption("Maze Game")
        
        # init game clock
        self.clock = pg.time.Clock()
        
        # setting cell size
        self.cell_size = size
        
        # setting columns and rows based on current width and height
        self.cols, self.rows = info.current_w // size, info.current_h // size
        
        # init maze
        self.maze = Maze(self.cols, self.rows, size)
        
        # init player
        self.player = Player(0, 0, size)
        
        # set end marker
        self.end = (self.cols - 1, self.rows - 1)

    def viz(self, agent):
        # init the player at the starting position
        self.player.x, self.player.y = 0, 0
            
        # init the starting state
        state = (self.player.x, self.player.y)
            
        # keeping track of visted paths
        vpath = []
            
        while state != self.end:
            # next action for agent
            action = agent.action(state)
                
            # init the direction of movement
            dx, dy = 0, 0
                
            # updating directions
            if action == 'left':
                dx = -1
            
            elif action == 'right':
                dx = 1
            
            elif action == 'up':
                dy = -1
            
            elif action == 'down':
                dy = 1
                
            # moving the player
            self.player.move(dx, dy, self.maze)
                
            # updating current state
            state = (self.player.x, self.player.y)
                
            # appending state to path list, to color the path
            vpath.append(state)
                
            # filling screen with the darkness
            # "Forget the promise of progress and understanding, for in the grim darkness of the far future there is only war."
            self.screen.fill(pg.Color("black"))
                
            # maze go brrrr
            for (x, y), cell in self.maze.grid.items():
                # top corner of the cell
                cx, cy = x * self.cell_size, y * self.cell_size
                    
                # displaying path walked by the agent
                if (x, y) in vpath:
                    pg.draw.rect(self.screen, pg.Color("lightgreen"), (cx, cy, self.cell_size, self.cell_size))
                    
                # top wall of cell
                if cell.walls['top']:
                    pg.draw.line(self.screen, pg.Color("white"), (cx, cy), (cx + self.cell_size, cy), 2)
                    
                # right wall of  cell
                if cell.walls['right']:
                    pg.draw.line(self.screen, pg.Color("white"), (cx + self.cell_size, cy), (cx + self.cell_size, cy + self.cell_size), 2)
                    
                # bottom wall of cell
                if cell.walls['bottom']:
                    pg.draw.line(self.screen, pg.Color("white"), (cx + self.cell_size, cy + self.cell_size), (cx, cy + self.cell_size), 2)
                    
                # left wall of cell
                if cell.walls['left']:
                    pg.draw.line(self.screen, pg.Color("white"), (cx, cy + self.cell_size), (cx, cy), 2)
                
            # player as dark green square
            pg.draw.rect(self.screen, pg.Color("darkgreen"), self.player.rect)
                
            # end point as dark green square
            x = self.end[0] * self.cell_size + self.cell_size // 3
            
            y = self.end[1] * self.cell_size + self.cell_size // 3

            pg.draw.rect(self.screen, pg.Color("red"), (x, y, self.cell_size // 3, self.cell_size // 3))
            
            # update the display
            pg.display.flip()
                
            # frame rate
            self.clock.tick(50)


class MetricsTracker:
    def __init__(self):

        self.episode_rewards = []

        self.success_rates = []

        self.steps_per_episode = []

        self.eps_val = []
    
    def log_episode(self, rewards, steps, success, epsilon):
        self.episode_rewards.append(sum(rewards))

        self.steps_per_episode.append(steps)

        self.success_rates.append(success)

        self.eps_val.append(epsilon)
    
    def plot_metrics(self):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards, label="Total Rewards per Episode", color='blue')
        plt.axhline(0, color='red', linestyle='--', label='Neutral Reward Line')
        plt.title("Total Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.success_rates, label="Success Rate", color='green')
        plt.title("Success Rate (0 or 1)")
        plt.xlabel("Episode")
        plt.ylabel("Success (1 = Goal Reached)")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(self.steps_per_episode, label="Steps per Episode", color='orange')
        plt.title("Steps Taken per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(self.eps_val, label="Epsilon (Exploration Rate)", color='purple')
        plt.title("Exploration Rate Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    game = Game(70)
    agent = QLearningAgent(game.maze)
    tracker = MetricsTracker()

    # set episode length
    for episode in range(5000):        
        rewards = []
        
        # setting pos to start
        game.player.x, game.player.y = 0, 0
        
        # state is now set to start
        state = (game.player.x, game.player.y)
        6
        # creating a visiting set
        visited = set()
        
        # counter to check if we are stuck
        cons_stuck = 0
        
        # looking at out last state
        last_state = None
        
        # checking per episode to improve the model
        for t in range(1000):
            
            # action setup
            action = agent.action(state)
            
            # setting movement variables
            dx, dy = 0, 0
            
            # now moving
            if action == 'left': 
                dx = -1
            
            if action == 'right':
                dx = 1
            
            if action == 'up': 
                dy = -1
            
            if action == 'down': 
                dy = 1
            
            # older cords
            old_x, old_y = game.player.x, game.player.y
            
            # moving
            game.player.move(dx, dy, game.maze)
            
            # new state of current pos
            new_state = (game.player.x, game.player.y)
            
            # dont hit the wall
            
            if (old_x, old_y) == new_state:
            
                # penalizing face planting
                reward = -5
            
                # adding stuck counter
                cons_stuck += 1
            
            else:
                # small penalty
                reward = -0.1
                
                # resetting stuck counter
                cons_stuck = 0
            
            if new_state in visited:
            
                # if revising, slightly large penalty
            
                reward = -2
            
            visited.add(new_state)
            
            # ayyyy, get a lot of points for reaching the end
            
            if new_state == game.end:
            
                reward = 15
    
            # learning more
            agent.uqtab(state, action, reward, new_state)
    
            rewards.append(reward)
            # print(reward)
    
            state = new_state
    
            if new_state == game.end:
    
                break
    
            if cons_stuck > 5:
    
                break
    
        # reducing randomness
    
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        # recording metrics for performance
        total_reward = sum(rewards)
        success = 1 if new_state == game.end else 0
        steps = len(rewards)

        tracker.log_episode(rewards, steps, success, agent.epsilon)   
    
    # final, no randomness, all on the model
    
    agent.epsilon = 0
    
    # init the viz function to show path, walls and bg
    
    game.viz(agent)
    
    tracker.plot_metrics()
    
    # THE END, was it happy no clue. But we are done
    pg.quit()
    
    sys.exit()