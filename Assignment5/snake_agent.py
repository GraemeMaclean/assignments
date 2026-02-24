import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write. 
    #   Function Helper:IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        # YOUR CODE HERE
        head_x, head_y, body, food_x, food_y = state
        
        # 1 & 2: Adjoining Walls (Grid-based boundaries)
        wall_x = 0
        if head_x <= helper.BOARD_LIMIT_MIN: wall_x = 1
        elif head_x >= helper.BOARD_LIMIT_MAX: wall_x = 2
            
        wall_y = 0
        if head_y <= helper.BOARD_LIMIT_MIN: wall_y = 1
        elif head_y >= helper.BOARD_LIMIT_MAX: wall_y = 2

        # 3 & 4: Food Direction
        food_dir_x = 0
        if food_x < head_x: food_dir_x = 1
        elif food_x > head_x: food_dir_x = 2
        
        food_dir_y = 0
        if food_y < head_y: food_dir_y = 1
        elif food_y > head_y: food_dir_y = 2

        # 5-8: Body Proximity
        body_top    = 1 if (head_x, head_y - helper.GRID_SIZE) in body else 0
        body_bottom = 1 if (head_x, head_y + helper.GRID_SIZE) in body else 0
        body_left   = 1 if (head_x - helper.GRID_SIZE, head_y) in body else 0
        body_right  = 1 if (head_x + helper.GRID_SIZE, head_y) in body else 0

        return (wall_x, wall_y, food_dir_x, food_dir_y, body_top, body_bottom, body_left, body_right)


    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the code you need to write. 
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above. 
    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make. 
    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state, points, dead):
        # YOUR CODE HERE
        head_x, head_y, _, food_x, food_y = state
        curr_dist = abs(head_x - food_x) + abs(head_y - food_y)
        
        curr_s = self.helper_func(state)
        
        if self._train and self.s is not None:
            # Reward Logic
            base_reward = self.compute_reward(points, dead)
            
            # Reward Shaping: did we get closer to food?
            if not dead:
                if curr_dist < self.prev_dist:
                    shaping = 0.1  # Moving closer
                else:
                    shaping = -0.2 # Moving away or staying same distance
                reward = base_reward + shaping
            else:
                reward = -10 # Increased death penalty to discourage risky moves
            
            # Bellman Update
            prev_sa = self.s + (self.a,)
            lr = self.LPC / (self.LPC + self.N[prev_sa])
            max_q_next = 0 if dead else np.max(self.Q[curr_s])
            self.Q[prev_sa] += lr * (reward + self.gamma * max_q_next - self.Q[prev_sa])

        self.points = points
        self.prev_dist = curr_dist # Track distance for the next step

        if dead:
            self.reset()
            return None

        # Action Selection
        best_action = 3 # Default to Right
        
        if self._train:
            best_f = -float('inf')
            # Priority tie-breaking (Right, Left, Down, Up)
            for action in [3, 2, 1, 0]:
                q_val = self.Q[curr_s + (action,)]
                n_val = self.N[curr_s + (action,)]
                
                # Exploration function f(q, n)
                f_val = 1 if n_val < self.Ne else q_val
                
                if f_val >= best_f:
                    best_f = f_val
                    best_action = action
            
            # Store state/action for the update in the next time-step
            self.s = curr_s
            self.a = best_action
            self.N[curr_s + (best_action,)] += 1
        else:
            # Pure exploitation for Testing
            best_q = -float('inf')
            for action in [3, 2, 1, 0]:
                if self.Q[curr_s + (action,)] >= best_q:
                    best_q = self.Q[curr_s + (action,)]
                    best_action = action
                    
        return best_action