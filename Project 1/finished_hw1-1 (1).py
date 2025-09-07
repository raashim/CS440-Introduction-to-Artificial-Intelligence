import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.patches import Rectangle
DEBUG = False # this runs the debug statements; set it to true to run them (there's a lot though... )
PRINTGRID = False # this will print a grid for every single test for every single grid 0-49

#This is the BinaryHeap class we implemented by ourself
class BinaryHeap:
   def __init__ (self):
       self.heap = [] 

   def push(self, priority, item):
       self.heap.append((priority, item))
       self.swim(len(self.heap) - 1)

   def pop(self):
       if not self.heap:
           raise IndexError("You cannot pop from an empty heap")
       self._swap(0, len(self.heap) - 1)
       popped = self.heap.pop()
       self.sink(0)
       return popped

   def is_empty(self):
       return len(self.heap) == 0

   def swim(self, index):
       parent = (index - 1) // 2
       while index > 0 and self.heap[index][0] < self.heap[parent][0]:
           self._swap(index, parent)
           index = parent
           parent = (index - 1) // 2

   def sink(self, index):
       length = len(self.heap)
       while True:
           left = 2 * index + 1
           right = 2 * index + 2
           smallest = index
           if left < length and self.heap[left][0] < self.heap[smallest][0]:
               smallest = left
           if right < length and self.heap[right][0] < self.heap[smallest][0]:
               smallest = right
           if smallest == index:
               break
           self._swap(index, smallest)
           index = smallest

   def _swap(self, i, j):
       self.heap[i], self.heap[j] = self.heap[j], self.heap[i]


#this is code to generate and plot the grid
def generate_grid(width = 101, height = 101):

   grid = [[None for _ in range(width)] for _ in range(height)]
   visited = [[False for _ in range(width)] for _ in range(height)]

   def get_neighbors(x, y):
       neighbors = []
       if y > 0:
           neighbors.append((x, y - 1))
       if y < height - 1:
           neighbors.append((x, y + 1))
       if x > 0:
           neighbors.append((x - 1, y))
       if x < width - 1:
           neighbors.append((x + 1, y))
       return neighbors
  
   def dfs(x, y):
       stack = [(x, y)]
       visited[y][x] = True
       grid[y][x] = 1  # mark a starting square as unblocked.
      
       while stack:
           cx, cy = stack[-1]
           unvisited = [(nx, ny) for nx, ny in get_neighbors(cx, cy) if not visited[ny][nx]]
           if unvisited:
               nx, ny = random.choice(unvisited)
               visited[ny][nx] = True
               # 70% probability mark as unblocked (1), else blocked (0).
               if random.random() < 0.7:
                   grid[ny][nx] = 1
                   stack.append((nx, ny))
               else:
                   grid[ny][nx] = 0
           else:
               stack.pop()

   all_cells = [(x, y) for y in range(height) for x in range(width)]
   random.shuffle(all_cells)
   for x, y in all_cells:
       if not visited[y][x]:
           dfs(x, y)

   return grid  # 2D list of 0/1

def load_grid(file_path):
   # loads a NumPy array from .npy (which should contain 0/1)
   return np.load(file_path)

def plot_grid(grid, path = None):
   fig, ax = plt.subplots(figsize=(8, 8))
   grid_array = np.array(grid, dtype=int)
   height, width = grid_array.shape
  
   # (0,0) is at the bottom-left.
   im = ax.imshow(
       grid_array,
       cmap='gray',
       origin='lower',
       extent=[0, width, 0, height],
       interpolation='nearest'
   )
  
   ax.set_aspect('equal')
  
   # Draw red blocks without overlapping previously drawn ones.
   if path:
       drawn_cells = set()
       for cell in path:
           if cell not in drawn_cells:
               drawn_cells.add(cell)
               x, y = cell
               rect = Rectangle((x, y), 1, 1, facecolor='red', edgecolor='none', alpha=0.8)
               ax.add_patch(rect)
  
   ax.set_xticks(np.arange(0, width+1, 10))
   ax.set_yticks(np.arange(0, height+1, 10))
   ax.set_xticks(np.arange(0, width+1, 1), minor=True)
   ax.set_yticks(np.arange(0, height+1, 1), minor=True)
  
   ax.grid(which='major', color='black', linestyle='-', linewidth=1)
   ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
  
   ax.set_title("Visual Mapping")
   ax.set_xlabel("X")
   ax.set_ylabel("Y")
   plt.show()

def save_grids(num_grids, folder_name='grid_worlds'):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)
  
   # Save each grid
   for i in range(num_grids):
       grid = generate_grid()  # Generate a new grid
       file_path = os.path.join(folder_name, f'grid_{i}.npy')  # Construct file path
       np.save(file_path, grid)  # Save the grid to a file
       print(f"Saved grid {i} to {file_path}")


"this is where the code for the search algorithms start "

# A* search (with adaptive heuristic option)
" this is the main heuristic function for A* search "
def manhattan_distance(cell, goal):
   return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

" should return any valid/within bounds neighboring coords (left, right, up, down) of a given cell "
def get_neighbors(cell, grid):
   x, y = cell
   height = len(grid)
   width = len(grid[0])
   neighbors = []
   if y > 0:
       neighbors.append((x, y - 1))
   if y < height - 1:
       neighbors.append((x, y + 1))
   if x > 0:
       neighbors.append((x - 1, y))
   if x < width - 1:
       neighbors.append((x + 1, y))
   return neighbors

#A star search 
"""
   Performs A* search using a custom binary heap as the open list.
   If adaptive_h is provided, h(x) = adaptive_h.get(x, heuristic_func(x, goal)).
   Returns (path, expanded_nodes, dictionary_mapping) or (None, expanded_nodes, dictionary_mapping)
   if no path is found.
"""
def a_star_search(grid, start, goal, heuristic_func, tie_break = 'high', adaptive_h = None):
   height = len(grid)
   width = len(grid[0])
   tie_breaker = width * height + 1  # constant for tie-breaking

   open_heap = BinaryHeap()
   dictionary_mapping = {start: 0}  # g-values: cost from start to node
   parent = {}  # to reconstruct the path

   # initial heuristic value.
   if adaptive_h is not None:
       h_start = adaptive_h.get(start, heuristic_func(start, goal))
   else:
       h_start = heuristic_func(start, goal)
   f_start = dictionary_mapping[start] + h_start

   if tie_break == 'high':
       priority = tie_breaker * f_start - dictionary_mapping[start]
   else:
       priority = tie_breaker * f_start + dictionary_mapping[start]
   open_heap.push(priority, start)

   expanded_nodes = set()  # closed set

   while not open_heap.is_empty():
       _, current = open_heap.pop()
       if current == goal:
           # reconstruct the path
           path = []
           while current in parent:
               path.append(current)
               current = parent[current]
           path.append(start)
           path.reverse()
           return path, expanded_nodes, dictionary_mapping
      
       if current in expanded_nodes:
           continue

       expanded_nodes.add(current)
       for nbr in get_neighbors(current, grid):
           nx, ny = nbr

           if grid[ny][nx] == 0:
               continue  # skip blocked cells

           tentative_g = dictionary_mapping[current] + 1
           if nbr not in dictionary_mapping or tentative_g < dictionary_mapping[nbr]:
               dictionary_mapping[nbr] = tentative_g
               parent[nbr] = current
               if adaptive_h is not None:
                   h_val = adaptive_h.get(nbr, heuristic_func(nbr, goal))
               else:
                   h_val = heuristic_func(nbr, goal)
               f_val = dictionary_mapping[nbr] + h_val
               if tie_break == 'high':
                   priority = tie_breaker * f_val - dictionary_mapping[nbr]
               else:
                   priority = tie_breaker * f_val + dictionary_mapping[nbr]
               open_heap.push(priority, nbr)
   return None, expanded_nodes, dictionary_mapping

" updates the agent's knowledge of the grid by revealing the status (blocked/unblocked) of all neighbors of the current position. "
def update_agent_knowledge(agent_knowledge, actual_grid, position):

   for nbr in get_neighbors(position, actual_grid):
       nx, ny = nbr
       agent_knowledge[ny][nx] = actual_grid[ny][nx]
       status = "unblocked" if actual_grid[ny][nx] == 1 else "blocked"
       if DEBUG:
           print(f"The neighbor at {(nx, ny)} is {status}.")
   return agent_knowledge

#Repeated forward A*
"""
   repeated forward A*:
     - plans a path from the agent's current cell to the goal.
     - follows the planned path until a blocked cell is discovered.
     - updates its knowledge and replans if necessary.
   if adaptive is true, uses and updates adaptive heuristics.
"""
def repeated_forward_a_star(actual_grid, start, goal, tie_break='high', adaptive=False): # this is supposed to be non-adaptive; h(s) is not updated
   
   height = len(actual_grid)
   width = len(actual_grid[0])
   agent_knowledge = [[None for _ in range(width)] for _ in range(height)]
   sx, sy = start
   gx, gy = goal
   agent_knowledge[sy][sx] = actual_grid[sy][sx]
   agent_knowledge[gy][gx] = actual_grid[gy][gx]
  
   # initialize the adaptive heuristics settings if needed.
   adaptive_h = {}
   if adaptive:
       for y in range(height):
           for x in range(width):
               adaptive_h[(x, y)] = manhattan_distance((x, y), goal)
  
   current = start
   path_taken = [current]
   while current != goal:
       if adaptive:
           path, expanded_nodes, g_values = a_star_search(agent_knowledge, current, goal, manhattan_distance, tie_break, adaptive_h)
       else:
           path, expanded_nodes, g_values = a_star_search(agent_knowledge, current, goal, manhattan_distance, tie_break)
      
       if path is None:
           if DEBUG:
               print("[FA*] A forwards path could not be found. The goal is most likely unreachable.")
           return path_taken, False
      
       if DEBUG:
           print(f"\nlength[FA* Replanning...] A* found a forward path of length {len(path)}:")
       for index, cell in enumerate(path):
           if DEBUG:
               print(f"  Step {index}: {cell}")
      
       # update the adaptive heuristics based on the search (Adaptive A* update)
       if adaptive and goal in g_values:
           goal_g = g_values[goal]
           for state in expanded_nodes:
               adaptive_h[state] = goal_g - g_values[state]
      
       for next_cell in path[1:]:
           agent_knowledge = update_agent_knowledge(agent_knowledge, actual_grid, current)
           if actual_grid[next_cell[1]][next_cell[0]] == 0:
               if DEBUG:
                   print(f"A blockage has been discovered at {next_cell} during foward search, replanning...")
               break
           current = next_cell
           path_taken.append(current)
           agent_knowledge = update_agent_knowledge(agent_knowledge, actual_grid, current)
           if current == goal:
               print("Reached the goal!")
               return path_taken, True
   print("Reached the goal!")
   return path_taken, True

#repeated backward A*
"""
   repeated Backward A*:
     - plans a path from the goal to the agent's current cell using the current knowledge.
     - reverses the found path and follows it.
     - updates the agent's knowledge as it moves.
 """
def repeated_backward_a_star(actual_grid, start, goal, tie_break='high'):
   
   height = len(actual_grid)
   width = len(actual_grid[0])
   agent_knowledge = [[None for _ in range(width)] for _ in range(height)]
   sx, sy = start # starting x and y coords
   gx, gy = goal # goal x and y coords
   agent_knowledge[sy][sx] = actual_grid[sy][sx]
   agent_knowledge[gy][gx] = actual_grid[gy][gx]
  
   current = start
   path_taken = [current]
   while current != goal:
       # plan from goal to current.
       path, expanded_nodes, g_values = a_star_search(agent_knowledge, goal, current, manhattan_distance, tie_break)
       if path is None:
           if DEBUG:
               print("[BA*] A backwards path could not be found. The goal is most likely unreachable.")
           return path_taken, False
       # reverse the path so that it goes from current to goal.
       path = list(reversed(path))
       if DEBUG:
           print(f"\nlength[BA* Replanning...] A* found a backward path of length {len(path)}:")
       for index, cell in enumerate(path):
           if DEBUG:
               print(f"  Step {index}: {cell}")
      
       for next_cell in path[1:]:
           agent_knowledge = update_agent_knowledge(agent_knowledge, actual_grid, current)
           if actual_grid[next_cell[1]][next_cell[0]] == 0:
               if DEBUG:
                   print(f"A blockage has been found at {next_cell} during backward search, replanning...")
               break
           current = next_cell
           path_taken.append(current)
           agent_knowledge = update_agent_knowledge(agent_knowledge, actual_grid, current)
           if current == goal:
               print("Reached the goal!")
               return path_taken, True
   print("Reached the goal!")
   return path_taken, True

#Adaptive A*
"""
   performs a single Adaptive A* search from start to goal on the given grid.

   if an adaptive_h is provided (a dictionary mapping states to heuristic values), A*
   then for each state s the heuristic used is adaptive_h.get(s, manhattan_distance(s, goal)).
   if it is not provided, adaptive_h is initialized with the Manhattan distance for every cell.

   after a successful search, the heuristic for every state in the expanded_nodes set
   is updated as: h_new(s) = g(goal) - g(s).

   rets a tuple (path, adaptive_h, g_values, success):
     - path: list of cells from start to goal (or None if no path exists)
     - adaptive_h: the updated dictionary of heuristic values
     - g_values: dictionary of g-values from the search
     - success: True if a path was found, False otherwise
"""
def adaptive_a_star(grid, start, goal, tie_break='high', adaptive_h=None):
   
   height = len(grid)
   width = len(grid[0])

   # if no adaptive heuristic dictionary is provided, initialize with Manhattan distance
   if adaptive_h is None:
       adaptive_h = {}
       for y in range(height):
           for x in range(width):
               adaptive_h[(x, y)] = manhattan_distance((x, y), goal)

   path, expanded_nodes, g_values = a_star_search(
       grid = grid,
       start = start,
       goal = goal,
       heuristic_func = manhattan_distance,
       tie_break = tie_break,
       adaptive_h = adaptive_h
   )

   if path is None:
       print("No path could be found by Adaptive A*.")
       return None, adaptive_h, g_values, False

   if goal in g_values:
       goal_g = g_values[goal]
       for state in expanded_nodes:
           adaptive_h[state] = goal_g - g_values[state]

   return path, adaptive_h, g_values, True


# #this is the methods for running all 3 test cases for all 50 grid environments
# #If you want to print the grids for all 50 of them (pop up one at a time), set the PRINTGRID boolean flag to TRUE
# def load_and_test_grids(num_grids=50):
#     for i in range(num_grids):
#         grid_filename = f"grid_worlds/grid_{i}.npy"
#         grid = load_grid(grid_filename)
#         print(f"\nTESTING GRID {i}:")
#         test_grid(grid)

# def test_grid(grid):
#     start, goal = get_random_start_goal()
#     if is_blocked(grid, start) or is_blocked(grid, goal):
#         print("Start or goal position is blocked, skipping test.")
#         return
#     print("Start:", start, "Goal:", goal)
#     run_tests(grid, start, goal)

# def get_random_start_goal():
#     return (random.randint(0, 100), random.randint(0, 100)), (random.randint(0, 100), random.randint(0, 100))

# def is_blocked(grid, position):
#     return grid[position[1]][position[0]] == 0

# def run_tests(grid, start, goal):
#     print("----------------------------------------")
#     run_algorithm(grid, start, goal, repeated_forward_a_star, 'high', False)
#     run_algorithm(grid, start, goal, repeated_forward_a_star, 'low', False)
#     run_algorithm(grid, start, goal, repeated_backward_a_star, 'high')
#     run_algorithm(grid, start, goal, adaptive_a_star, 'high', True)

# def run_algorithm(grid, start, goal, algorithm, tie_break, adaptive=False):
#     print(f"Running {algorithm.__name__} with tie-break: {tie_break}...")
#     start_time = time.time()
    
#     if algorithm == adaptive_a_star:
#         path_taken, _, _, success = algorithm(grid, start, goal, tie_break)
#     elif algorithm == repeated_backward_a_star:
#         path_taken, success = algorithm(grid, start, goal, tie_break)  # Do not pass `adaptive`
#     else:
#         path_taken, success = algorithm(grid, start, goal, tie_break, adaptive)
    
#     end_time = time.time()
#     process_results(grid, path_taken, success, start, goal, end_time - start_time)

# def process_results(grid, path_taken, success, start, goal, elapsed_time):
#     print(f"Elapsed time: {round(elapsed_time, 5)} seconds")
#     if success:
#         print("The path to the goal has been completed!")
#     else:
#         print("The goal could not be reached.")
#     print(f"The starting coordinate was {start}, and the goal was {goal}.")
#     if PRINTGRID and path_taken:
#         plot_grid(grid, filter_unique_path(path_taken))
#     plt.clf()

# def filter_unique_path(path_taken):
#     unique_path = []
#     visited_set = set()
#     for cell in path_taken:
#         if cell not in visited_set:
#             visited_set.add(cell)
#             unique_path.append(cell)
#     return unique_path

# if __name__ == '__main__':
#     load_and_test_grids()

# #End of helper functions to run all 3 test cases on all 50 grid environments



#individual helper functions for the main function to test one by one
def load_and_test_single_grid(grid_filename="grid_worlds/grid_29.npy"):
    grid = load_grid(grid_filename)
    print(f"\nTESTING GRID: {grid_filename}")
    test_grid(grid)

def test_grid(grid):
    start, goal = get_random_start_goal()
    if is_blocked(grid, start) or is_blocked(grid, goal):
        print("Start or goal position is blocked, skipping test.")
        return
    print("Start:", start, "Goal:", goal)
    run_tests(grid, start, goal)

def get_random_start_goal():
    return (random.randint(0, 100), random.randint(0, 100)), (random.randint(0, 100), random.randint(0, 100))

def is_blocked(grid, position):
    return grid[position[1]][position[0]] == 0

#comment out the tests that you don't want to run
def run_tests(grid, start, goal):
    print("----------------------------------------")
    run_algorithm(grid, start, goal, repeated_forward_a_star, 'high', False)
    run_algorithm(grid, start, goal, repeated_forward_a_star, 'low', False)
    #run_algorithm(grid, start, goal, repeated_backward_a_star, 'high')
    run_algorithm(grid, start, goal, adaptive_a_star, 'high', True)

def run_algorithm(grid, start, goal, algorithm, tie_break, adaptive=False):
    print(f"Running {algorithm.__name__} with tie-break: {tie_break}...")
    start_time = time.time()
    
    if algorithm == adaptive_a_star:
        path_taken, _, _, success = algorithm(grid, start, goal, tie_break)
    
    if algorithm == repeated_backward_a_star:  
        path_taken, success = algorithm(grid, start, goal, tie_break) 
    
    if algorithm == repeated_forward_a_star:
        path_taken, success = algorithm(grid, start, goal, tie_break, adaptive)  

    end_time = time.time()
    process_results(grid, path_taken, success, start, goal, end_time - start_time)

def process_results(grid, path_taken, success, start, goal, elapsed_time):
    print(f"Elapsed time: {round(elapsed_time, 5)} seconds")
    if success:
        print("The path to the goal has been completed!")
    else:
        print("The goal could not be reached.")
    print(f"The starting coordinate was {start}, and the goal was {goal}.")
    if PRINTGRID and path_taken:
        plot_grid(grid, filter_unique_path(path_taken))
    plt.clf()

def filter_unique_path(path_taken):
    unique_path = []
    visited_set = set()
    for cell in path_taken:
        if cell not in visited_set:
            visited_set.add(cell)
            unique_path.append(cell)
    return unique_path

if __name__ == '__main__':

    #if the grid environments need to be generated, call the method: save_grids(50, folder_name = 'grid_worlds')
    load_and_test_single_grid()


#End of helper functions to run one test case on one grid environment 