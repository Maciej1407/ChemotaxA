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

def getEffectiveStimulus(v_max, S, K_d):
    '''
    Calculates the effective stimulus (or degree to which the cell is affected by a chemoattractant) 
    according to the relative concentration of the chemoattractant of the cell as well as 
    it's saturation coefficient and the Michaelis-Menten constant.

    Parameters:
        cell (Cell): The cell object to calculate the effective stimulus for where 
            cell.v_max is the maximum reaction of the cell which is determined by the number of 
            receptors the cell is instantiateed with as well as single receptor sensitivity, also
            defined on cell instantiation,
        S (float): The concentration of the chemoattractant at the cell's position.
        K_d (float): The Michaelis-Menten constant defined by k_off / k_on which are 
            defined as global variables 
    
    Returns:
        v (float): The effective stimulus
    '''
    
    v = (v_max * S) / (K_d + S)
    return v

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
  #  radius_squared = radius ** 2
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
    return np.array(points) # return outer points of the cell which are drawn



@jit(nopython=True)
def _compute_gradient(pos_x, pos_y, u, dx, dy, step = 1):
    '''
    Compute the gradient of a field u at a given position (pos_x, pos_y) using central differences.
    Intended to be used within a wrapped function defined as part of the cell class.
    This version is defined outside the class to allow for the use of numba's JIT compiler.

    Parameters:
        pos_x (int): The x-coordinate of the position.
        pos_y (int): The y-coordinate of the position.
        u (numpy.ndarray): The field to compute the gradient of.
        dx (float): The grid spacing in the x-direction.
        dy (float): The grid spacing in the y-direction.
        step (int): The step size for the gradient computation.

    Returns:
        numpy array: A tuple containing the x and y components of the gradient.
    '''
    x, y = pos_x, pos_y 

    # Compute central difference gradient
    if 1 <= x < u.shape[0] - 1 and 1 <= y < u.shape[1] - 1: # boundary rejection

        grad_x = (u[x+step, y] - u[x-step, y]) / (2 * dx) # x grad
        grad_y = (u[x, y+step] - u[x, y-step]) / (2 * dy) # y grad
    
    else:
        grad_x, grad_y = 0, 0 
    
    return grad_x, grad_y

def degrade_pos_gen(center_x, center_y, side_length, grid_shape): 
    '''
    Function intended to run a single time. Calculates the area of degradation for a given cell perimeter.
    The area is updated as the cell moves using the same movement vector as the cell.
    
    Parameters:
        center_x (int): The x-coordinate of the center of the cell.
        center_y (int): The y-coordinate of the center of the cell.
        side_length (int): The side length of the square area to be degraded.
        grid_shape (tuple): The shape of the grid as (height, width).
    
    Returns:
        numpy.ndarray: An array of shape (N, 2) where N is the number of positions in the degraded area.
        Each row contains the (x, y) coordinates of a position within the degraded area.
    '''

    grid_h, grid_w = grid_shape

    str_x = max(0, center_x - side_length // 2)
    end_x = min(grid_w, center_x + side_length // 2 + 1)

    str_y = max(0, center_y - side_length // 2) 
    end_y = min(grid_h, center_y + side_length // 2 + 1)   

    x = np.arange(str_x, end_x)
    y = np.arange(str_y, end_y)

    x_mesh, y_mesh = np.meshgrid(x, y)

        # Stack x and y coordinates into a single array
    positions = np.vstack((x_mesh.ravel(), y_mesh.ravel())).T
    
    return positions


class Cell_2():
    '''
    ----------------
    This class represents a cell that can move within a grid. 
    The cell's position is updated based on random movement and gradients computed from a given field. 
    The cell's position history is recorded and can be retrieved in either list, dataframe or dictionary format.
    To-Add, Chemotaxing param recording if the cell is influenced by a chemical gradient.
    
    Attributes:
    grid (numpy.ndarray): The grid the cell is moving in, used to determine the boundaries of the cell's movement.
    
    pos_x (int): x-coordinate of the cell 
    pos_y (int): y-coordinate of the cell   
    
    pos_history (deque): history of the cell's position
        a default attribute of the cell not used as an instantiation parameter
    
    shape (list): shape of the cell, options:
        [circle, radius], 
        [rectangular / square , width, height],
    
    degradation_area (int): area of the grid to be degraded by the cell's movement
    
    nR (int): number of receptors on the cell
    
    k_cat (float): catalytic rate constant of the cell
        - used to determine the maximum reaction rate of the cell
    
    v_max (float): maximum reaction rate of the cell
        - determined by the number of receptors and the catalytic rate constant

    default (bool): flag to determine if the cell is a default circle or a custom shape
        - Please note that while the code includes implementations for custom shapes, of various sizes
        - Most advanced features are only impmplemeted for a single point cell (or a circle with a radius of 1)
    points (numpy.ndarray): array of points that make up the cell's shape
    -----------

    ''' 
    def __init__(self, grid, pos_x, pos_y, shape  = [ "circle", 1], degradation_area = 1, nR = 100, k_cat = 0.4, secretion = False):

        self.v_max = nR * k_cat
        self.default = True
        self.degRadius = degradation_area
        self.degArea = degrade_pos_gen(pos_x, pos_y, degradation_area, grid.shape)
        self.secrete = secretion

        if shape[0] == "circle" and shape[1] > 1:

            self.default = False
            self.points = circle_points((pos_x, pos_y), shape[1])
        else:
            self.points = np.array([[pos_x, pos_y]])
            
        self.shape = shape[0]
        self.pos_x = int(pos_x)
        self.pos_y = int(pos_y)
        self.pos_history = deque()
        self.pos_history.append([0,self.pos_x,self.pos_y])

        self.RS_history = deque()
    
    def update_pos(self, grid_size, step = 1):

        if self.default:
            self.pos_x = np.clip(self.pos_x + np.random.randint(-step, step), 0, grid_size - 1)
            self.pos_y = np.clip(self.pos_y + np.random.randint(-step, step), 0, grid_size - 1)
        else:
            self.points = np.clip(self.points + [np.random.randint(-step, step), np.random.randint(-step, step)], 0, grid_size - 1)

    def attractant_secretion_rule(self, rule = "random"):
        '''
        This function is intended to be used to determine the cell's secretion of an attractant based on the cell's position
        and the state of the cell. This function is intended to be called within the update_pos_grad function.
        The function is currently implemented as a random chance of secretion. But it is intended to be expanded
        as a Learned Rule in a Reinfocement Learning Model.
        '''
        if (random.uniform(0,1) < 0.001):
            return True
        else:
            return False
        


   # @jit(nopython = True)
    def compute_gradient(self, u, dx, dy, step = 1):
        
        grad_x, grad_y = _compute_gradient(self.pos_x, self.pos_y, u, dx, dy, step)
        return grad_x, grad_y
    
    #@jit(nopython = True)
    def update_pos_grad(self, u, dx, dy, sensitivity, time_curr, dt, grid_shape , step=1):
        
        grad_x, grad_y = self.compute_gradient(u, dx, dy)

        # Random movement with gradient influence
        rand_x = random.randint(-step, step)
        rand_y = random.randint(-step, step)

        v_max = self.v_max

        S = u[self.pos_x, self.pos_y]
        v = getEffectiveStimulus(v_max, S, K_d)

        sensitivity = v / v_max       # Sensitivity is defibed as the ratio of the effective stimulus to the maximum stimulus


        self.RS_history.append([time_curr + dt, v, (grad_x+grad_y)/2])

        move_x = int(rand_x + grad_x * sensitivity)
        move_y = int(rand_y + grad_y * sensitivity)

        # Apply the movement to all points
        mov_vector = np.array([move_y, move_x])
        self.points += mov_vector
        self.pos_x += move_x
        self.pos_y += move_y

        self.degArea += mov_vector

        # Boundary values
        x_min, x_max = 0, grid_shape[0] - 1
        y_min, y_max = 0, grid_shape[1] - 1

        # Find points outside the grid
        outside_points = self.points[
            (self.points[:, 0] < x_min) | (self.points[:, 0] > x_max) |
            (self.points[:, 1] < y_min) | (self.points[:, 1] > y_max)
        ]

        if outside_points.size > 0:
            # Compute correction vector as the negative sum of out-of-bounds offsets
            correction_vector = np.array([
                -np.sum(outside_points[:, 0] - np.clip(outside_points[:, 0], x_min, x_max)),
                -np.sum(outside_points[:, 1] - np.clip(outside_points[:, 1], y_min, y_max))
            ])

            # Apply correction to all points and the center
            self.points += correction_vector
            self.pos_x += correction_vector[1]
            self.pos_y += correction_vector[0]

            self.degArea += correction_vector

        # Append the position to history
        self.pos_history.append([time_curr + dt, self.pos_x, self.pos_y])

    def get_position_history(self, type="list"):
        
        if type=="list":
            return list(self.pos_history)
        elif type == "df":
            df = pd.DataFrame ( list(self.pos_history), columns = ["time_step", "pos_x", "pos_y"]) 
            return df
        elif type == "dict":
            return { "time_step": [x[0] for x in self.pos_history], "pos_x": [x[1] for x in self.pos_history], "pos_y": [x[2] for x in self.pos_history]}
        
    def get_stimulation_stats(self, type="list"):
        
        if type=="list":
            return list(self.RS_history)
        elif type == "df":
            df = pd.DataFrame ( list(self.RS_history), columns = ["time_step", "effective_stimulus", "gradient_magnitude"]) 
            return df
        elif type == "dict":
            return { "time_step": [x[0] for x in self.RS_history], "effective_stimulus": [x[1] for x in self.RS_history], "gradient_magnitude": [x[2] for x in self.RS_history]}


k_on = 2e2  # M^-1 s^-1#
k_off = 10e4 # s^-1

K_d = k_off / k_on # M

alpha = 50
length = 400
sim_time = 50
nodes = 250
num_cells= 3

dx = length / nodes
dy= length / nodes


dt = min(dx**2 / (4*alpha), dy**2 / (4/alpha))

t_nodes = int(sim_time/dt)

u = np.zeros((nodes, nodes))


array_len = len(u)

max_temp = 100

u[:,-1:-10] = max_temp
#u[:,int(nodes*0.75):]= max_temp
u[0:75,:] = max_temp
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

#cells = [Cell_2( u, int(nodes/ 2), int( nodes / 2), shape= ["circle", 3], degradation_area = 10) for _ in range(num_cells)]
cells = [Cell_2( u, int(nodes/ 2), int( nodes / 2), secretion=False, degradation_area=2) for _ in range(1)]

cells.append( Cell_2(u, int(nodes/2), int(nodes/2),secretion=False, shape=["circle",3], degradation_area=4))
cells.append( Cell_2(u, int(nodes/2), int(nodes/2),secretion=False, shape=["circle",2], degradation_area=2))
cells.append( Cell_2(u, int(nodes/2), int(nodes/2),secretion=False, shape=["circle",5], degradation_area=5))

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
def update_cell(c, u, dx, dy, counter, dt, grid_size):
    
    c.update_pos_grad(u, dx, dy, 0.5, counter, dt, grid_size)
    
    if c.degRadius > 1:

        x_min, x_max = 0, u.shape[1] - 1
        y_min, y_max = 0, u.shape[0] - 1

        clipped_degArea = np.copy(c.degArea)
        clipped_degArea[:, 0] = np.clip(clipped_degArea[:, 0], y_min, y_max)  # Y-axis
        clipped_degArea[:, 1] = np.clip(clipped_degArea[:, 1], x_min, x_max)  # X-axis

        u[clipped_degArea[:,1], clipped_degArea[:,0]] = u[clipped_degArea[:,1], clipped_degArea[:,0]] / 10


        #u[c.degArea[:,1], c.degArea[:,0]] = u[c.degArea[:,1], c.degArea[:,0]] / 10
    else:
        u[c.pos_x, c.pos_y] =  u[c.pos_x, c.pos_y] / 10
        if c.secrete:
            if c.attractant_secretion_rule():
                u[c.pos_x, c.pos_y] =  u[c.pos_x, c.pos_y] + 40
        
    return c

start = time.time()

while counter < sim_time : # O(t)

    w = u.copy()
    if cellMarker:
        for mark in cellMarker: # O(n)
            mark.remove()

    u = calc_grad_np(u) 

    tasks = [ delayed (update_cell)(c, u, dx, dy, counter, dt, u.shape) for c in cells ]
        
    
    results = dask.compute(*tasks)

    print("t: {:.3f} [s], Concentration {:.2f} %".format(counter, np.average(u)))

    pcm.set_array(u)
    axis.set_title("Distribution at t: {:.3f} [s].".format(counter))


    try:
        cellMarker = [axis.plot(cell.points[:,0] , cell.points[:,1], 'wo', markersize=1)[0] for cell in cells]  
    except:
        print("Error in plotting")
        for cell in cells:
            print(cell.points)
        exit(1)
 
    plt.pause(0.01)
    counter += dt
    
end = time.time()

FINAL = end - start

print(f'Total Execution Time: {FINAL}')

#plt.show()


#print(cells[0].get_position_history())