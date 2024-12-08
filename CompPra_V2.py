#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import numba
import random
from numba import jit
import time
import dask 
from dask import delayed
from collections import deque
import pandas as pd
#from numba.experimental import jitclass


def circle_points(center, radius):
    """
    Calculate all points within a circle in a 2D grid using a rasterization approach,
    returning the points as an array of arrays.

    Parameters:
        center (tuple): The (x, y) coordinates of the circle's center.
        radius (int): Radius of the circle.
    
    Returns:
        list: A list of [x, y] points inside the circle.
    """
    cx, cy = center
    radius_squared = radius ** 2
    points = []

    # Use the midpoint circle algorithm to find points
    x, y = radius, 0
    decision = 1 - radius

    while x >= y:
        # Add the points for each octant of the circle as lists
        points.extend([
            [cx + x, cy + y], [cx - x, cy + y], [cx + x, cy - y], [cx - x, cy - y],
            [cx + y, cy + x], [cx - y, cy + x], [cx + y, cy - x], [cx - y, cy - x]
        ])

        y += 1
        if decision <= 0:
            decision += 2 * y + 1
        else:
            x -= 1
            decision += 2 * (y - x) + 1

    # Filter points to ensure they are within the bounds of the grid (optional)
    points = [list(item) for item in set(tuple(point) for point in points)]  # Remove duplicates
    return np.array(points)



@jit(nopython=True)
def _compute_gradient(pos_x, pos_y, u, dx, dy, step = 1):
    
    x, y = pos_x, pos_y 

    # Compute central difference gradient
    if 1 <= x < u.shape[0] - 1 and 1 <= y < u.shape[1] - 1: # boundary rejection

        grad_x = (u[x+step, y] - u[x-step, y]) / (2 * dx) # x grad
        grad_y = (u[x, y+step] - u[x, y-step]) / (2 * dy) # y grad
    
    else:
        grad_x, grad_y = 0, 0 
    
    return grad_x, grad_y
#@jitclass(spec)
""" class Cell():

    def __init__(self, pos_x, pos_y):
        self.pos_x = int(pos_x)
        self.pos_y = int(pos_y)
        self.pos_history = deque()
        self.pos_history.append([0,self.pos_x,self.pos_y])
    
    def update_pos(self, grid_size, step = 1):
        self.pos_x = np.clip(self.pos_x + np.random.randint(-step, step), 0, grid_size - 1)
        self.pos_y = np.clip(self.pos_y + np.random.randint(-step, step), 0, grid_size - 1)


   # @jit(nopython = True)
    def compute_gradient(self, u, dx, dy, step = 1):
        
        grad_x, grad_y = _compute_gradient(self.pos_x, self.pos_y, u, dx, dy, step)
        return grad_x, grad_y
    
    #@jit(nopython = True)
    def update_pos_grad(self, u, dx, dy, sensitivity, time_curr, step=1,):
        
        grad_x, grad_y = self.compute_gradient(u, dx, dy)

        rand_x = random.randint(-step, step)
        rand_y = random.randint(-step, step)
        # Scale gradient to influence movement
        move_x = rand_x + grad_x * sensitivity
        move_y = rand_y + grad_y * sensitivity

        # Update cell position with stochastic movement, ensuring it stays within bounds
        self.pos_x = np.clip(self.pos_x + int(move_x), 0, u.shape[0] - 1)
        self.pos_y = np.clip(self.pos_y + int(move_y), 0, u.shape[1] - 1)

        self.pos_history.append([time_curr, self.pos_x, self.pos_y])

    def get_position_history(self):
        return list(self.pos_history)
     """



class Cell_2():
    '''
    ----------------
    This class represents a cell that can move within a grid. 
    The cell's position is updated based on random movement and gradients computed from a given field. 
    The cell's position history is recorded and can be retrieved in either list, dataframe or dictionary format.
    
    Attributes:
    pos_x (int): x-coordinate of the cell 
    pos_y (int): y-coordinate of the cell   
    pos_history (deque): history of the cell's position
    shape (list): shape of the cell, options:
        [circle, radius],
        [rectangular / square , width, height],
    
    polarity:
    saturation coefficient:
    -----------

    ''' 
    def __init__(self, pos_x, pos_y, shape  = [ "circle", 1] ):

        self.default = True

        if shape[0] == "circle":

            self.default = False
            self.points = circle_points((pos_x, pos_y), shape[1])

        self.shape = shape[0]
        self.pos_x = int(pos_x)
        self.pos_y = int(pos_y)
        self.pos_history = deque()
        self.pos_history.append([0,self.pos_x,self.pos_y])
    
    def update_pos(self, grid_size, step = 1):

        if self.default:
            self.pos_x = np.clip(self.pos_x + np.random.randint(-step, step), 0, grid_size - 1)
            self.pos_y = np.clip(self.pos_y + np.random.randint(-step, step), 0, grid_size - 1)
        else:
            self.points = np.clip(self.points + [np.random.randint(-step, step), np.random.randint(-step, step)], 0, grid_size - 1)


   # @jit(nopython = True)
    def compute_gradient(self, u, dx, dy, step = 1):
        
        grad_x, grad_y = _compute_gradient(self.pos_x, self.pos_y, u, dx, dy, step)
        return grad_x, grad_y
    
    #@jit(nopython = True)
    def update_pos_grad(self, u, dx, dy, sensitivity, time_curr, dt, step=1):
        
        grad_x, grad_y = self.compute_gradient(u, dx, dy)

        rand_x = random.randint(-step, step)
        rand_y = random.randint(-step, step)

        # Scale gradient to influence movement
        move_x = int(rand_x + grad_x * sensitivity)
        move_y = int(rand_y + grad_y * sensitivity)

        # Update cell position with stochastic movement, ensuring it stays within bounds

        if self.default:
            self.pos_x = np.clip(self.pos_x + move_x, 0, u.shape[0] - 1)
            self.pos_y = np.clip(self.pos_y + move_y, 0, u.shape[1] - 1)

        else:
            mov_vector = np.array([move_y, move_x])
            self.pos_x = np.clip(self.pos_x + move_x, 0, u.shape[0] - 1)
            self.pos_y = np.clip(self.pos_y + move_y, 0, u.shape[1] - 1)
            self.points = np.clip(self.points + mov_vector, 0, u.shape[0] - 1)
 

        self.pos_history.append([time_curr+dt, self.pos_x, self.pos_y])

    def get_position_history(self,type="list"):
        if type=="list":
            return list(self.pos_history)
        elif type == "df":
            df = pd.DataFrame ( list(self.pos_history), columns = ["time_step", "pos_x", "pos_y"]) 
            return df
        elif type == "dict":
            return { "time_step": [x[0] for x in self.pos_history], "pos_x": [x[1] for x in self.pos_history], "pos_y": [x[2] for x in self.pos_history]}
        

    
alpha = 400
length = 400
sim_time = 5
nodes = 250
num_cells= 8

dx = length / nodes
dy= length / nodes


dt = min(dx**2 / (4*alpha), dy**2 / (4/alpha))

t_nodes = int(sim_time/dt)

u = np.zeros((nodes, nodes))


array_len = len(u)

max_temp = 100

u[:,-1:-10] = max_temp
#u[:,int(nodes*0.75):]= max_temp
u[0:50,:] = max_temp
#u[ int(len(u)/2): int(len(u)/2)] = 100

center_x, center_y = nodes // 2, nodes // 2
radius = 15  # Radius of high temperature region



#u[center_x - radius:center_x + radius, center_y - radius:center_y + radius] = max_temp
# centre circle
#u[:,int(nodes*0.75):]= max_temp

#u[center_x - radius:center_x + radius, center_y - radius:center_y + radius] = max_temp
# centre circle


fig, axis = plt.subplots()
pcm = axis.pcolormesh(u, cmap = plt.cm.jet, vmin=0, vmax=100)

plt.colorbar(pcm, ax=axis)

#cells = [Cell( int(nodes/ 2),int( nodes / 2)) for _ in range(num_cells)]

cells = [Cell_2( int(nodes/ 2), int( nodes / 2), shape=["circle", 3] ) for _ in range(num_cells)]

counter = 0 

cellMarker = []

@jit(nopython=True)
def calc_grad_np(u):
    w = u.copy()
    w[1:-1, 1:-1] = (
        u[1:-1, 1:-1]
        + alpha * dt * (
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
            + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
    )
    return w

@delayed
def update_cell(c, u, dx, dy, counter, dt):
    c.update_pos_grad(u, dx, dy, 0.6, counter, dt)
    u[c.pos_x, c.pos_y] =  u[c.pos_x, c.pos_y] / 2
    return c

start = time.time()

while counter < sim_time : # O(t)

    w = u.copy()
    if cellMarker:
        for mark in cellMarker: # O(n)
            mark.remove()

    u = calc_grad_np(u) 

    tasks = [ delayed (update_cell)(c, u, dx, dy, counter, dt) for c in cells ]
        
    
    results = dask.compute(*tasks)

    print("t: {:.3f} [s], Concentration {:.2f} %".format(counter, np.average(u)))

    pcm.set_array(u)
    axis.set_title("Distribution at t: {:.3f} [s].".format(counter))

    print("here ")
    print(cells[0].points)
    print("end here ")
    cellMarker = [axis.plot(cell.points[:,0] , cell.points[:,1], 'wo', markersize=1)[0] for cell in cells]  

 
    plt.pause(0.01)
    counter += dt
    
end = time.time()

FINAL = end - start

print(f'Total Execution Time: {FINAL}')

#plt.show()


print(cells[0].get_position_history())