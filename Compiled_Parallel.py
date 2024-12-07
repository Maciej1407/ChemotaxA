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
#from numba.experimental import jitclass
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
class Cell():

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
alpha = 400
length = 700
sim_time = 1
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

cells = [Cell( int(nodes/ 2),int( nodes / 2)) for _ in range(num_cells)]

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


def calc_grad(nodes, u, w):
    for i in range(1, nodes-1 ): # O(n)
        for j in range(1, nodes - 1): # O(n)

            dd_ux = ( w[i-1, j] - 2 * w[i, j] + w[i+1, j]) / dx**2
            dd_uy = ( w[i, j-1 ] - 2 * w[i, j] + w[i, j+1]) / dy**2 

            u[i,j] = dt * alpha * (dd_ux + dd_uy) + w[i,j]
    return u

@delayed
def update_cell(c, u, dx, dy, counter):
    c.update_pos_grad(u, dx, dy, 2, counter)
    u[c.pos_x, c.pos_y] =  u[c.pos_x, c.pos_y] / 10
    return c

start = time.time()

while counter < sim_time : # O(t)

    w = u.copy()
    if cellMarker:
        for mark in cellMarker: # O(n)
            mark.remove()
            
    tasks = [ delayed (update_cell)(c, u, dx, dy, counter) for c in cells ]
        
    
    results = dask.compute(*tasks)

    print("t: {:.3f} [s], Concentration {:.2f} %".format(counter, np.average(u)))

    pcm.set_array(u)
    axis.set_title("Distribution at t: {:.3f} [s].".format(counter))

    cellMarker = [axis.plot(cell.pos_y, cell.pos_x, 'wo', markersize=8)[0] for cell in cells]  

 
#    plt.pause(0.01)
    counter += dt
    
end = time.time()

FINAL = end - start

print(f'Total Execution Time: {FINAL}')

#plt.show()


print(cells[0].get_position_history())