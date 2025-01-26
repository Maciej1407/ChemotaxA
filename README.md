# PyChemoTax: Chemotaxis Simulation 

PyChemoTax is a Python-based framework designed to simulate chemotaxis-driven cellular dynamics in customizable environments. Leveraging both computational efficiency and biological realism, this project includes reinforcement learning capabilities for optimizing agent behavior, enabling the exploration of emergent phenomena in cell populations.

## Python Version,
For package compatibility, please use Python 3.12, alternatively 3.9

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

You are free to share and adapt this code for non-commercial purposes, provided that:
- Proper attribution is given to the original creator.
- Any derivatives are distributed under the same license.


## Features

- **Simulations of Chemotactic Behavior**: Models agents influenced by chemoattractant gradients using Michaelis-Menten kinetics.
- **Reinforcement Learning (RL) Integration**: A proof-of-concept RL module allows agents to optimize parameters such as degradation rate and secretion probability.
- **Highly Customizable Agents**: Define cell shapes, receptor density, sensitivity, degradation areas, and more.
- **Efficient Simulation**: Employs JIT compilation, parallelism with Dask, and vectorized operations for scalability.
- **Visualization**: Real-time visualization of agent dynamics and gradient fields using Matplotlib.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/<your-repo-name>/#ChemotaxA.git
cd ChemotaxA
```

Install dependencies:
(in CondaReq File Path)
```bash
conda env create -f environment.yml
```
( (OPTIONAL) Simulation.py is compatible without Conda / Jupyter, for this alone use (PipReq). You should be able to run all the code with the conda environment)
```bash
pip install -r requirements.txt
```
### Installation Considerations
Due to differences in OS / CPU Architecture (this codebase was made using Apple Silicon), there may be issues regarding the yml files and requirements. If this is the case please use the following instrucitons.

- Create a conda environment with python 3.12
- Install the following package:
- - Numba
- - Numpy
- - Pandas
- - Matplotlib
- - Plotly / Plotly.express
- - Dask
- - Seaborn 
- - Jupyter

## Branches

- **Main Branch**: Contains the stable and tested version of PyChemoTax. Use this branch for production-level simulations.
- **refactor\_RL Branch**: Experimental branch focusing on refactoring the reinforcement learning module. Contains preliminary implementations and may lack full stability.

To switch between branches:

```bash
git checkout <branch-name>
```

## Usage

### Running the Simulation

To run the main simulation:

```bash
python simulation.py
```

This will execute a reinforcement learning simulation where agents adapt their behavior over multiple epochs.

### Customizing Parameters

Key parameters are defined in the script and can be modified to suit specific needs:

- **Environment**: Grid size, initial chemoattractant distribution, and diffusion properties.
- **Agents**: Number of agents, shape, receptor density, degradation area, and secretion behavior.
- **Reinforcement Learning**: Number of epochs, time steps per epoch, and fitness function.

### Visualization

The simulation provides real-time visualization of the chemoattractant field and agent dynamics. At the end of the simulation, you can also plot:

- Fitness over time: Displays how the average fitness of agents evolves during training
- Cell Paths over time: Displays a static graph displaying cell movements
- Simulation Statistics: Look at cell properties over time of the simulation e.g. Gradient Field Concentration V.S. Effective Stimulus

```python
# Example of fitness plotting
plot_cell_history(cell.get_pos_history("df") )

#Example plotting Effective Stimulus compared to Gradient Concentration over time
plot_statistics(df, columns = ["time_step", "effective_stimulus", "gradient_magnitude"], polynomial_fit=True, degree=3)
```

## Known Issues
- The `refactor_RL` branch contains experimental features and may produce unstable results.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

### Branching Strategy

- **main**: Stable and production-ready.
- **refactor\_RL**: Development and experimental features.

Follow [GitHub Flow](https://guides.github.com/introduction/flow/) for submitting contributions.

## License

This project is licensed 

## Acknowledgments

- Thanks to researcher Dr Robert Insall for inspiring this project 
- Inspired by computational biology and the need for scalable, realistic chemotaxis simulations.

## Contact

For questions or suggestions, please create an issue or reach out at (dmcbmkm@ucl.ac.uk).


# PyChemoTax: Function Descriptions

This document provides a description of the key functions used in the PyChemoTax framework. Each function plays a vital role in enabling the simulation of chemotaxis-driven cellular dynamics and reinforcement learning.

## Core Functions

### 1. `getEffectiveStimulus(v_max, S, K_d)`
**Purpose**: Calculates the effective stimulus of a cell based on Michaelis-Menten kinetics.

**Parameters**:
- `v_max` (float): Maximum reaction rate determined by receptor number and sensitivity.
- `S` (float): Local chemoattractant concentration.
- `K_d` (float): Dissociation constant.

**Returns**:
- `v` (float): Effective stimulus.

---

### 2. `circle_points(center, radius)`
**Purpose**: Computes all points within a circle in a 2D grid.

**Parameters**:
- `center` (tuple): \((x, y)\) coordinates of the circle center.
- `radius` (int): Circle radius.

**Returns**:
- Numpy matrix of points within circle

---

### 3. `_compute_gradient(pos_x, pos_y, u, dx, dy, step=1)`
**Purpose**: Calculates the local cell gradient of a chemoattractant field at a given position using central differences.

**Parameters**:
- `pos_x` (int): x-coordinate.
- `pos_y` (int): y-coordinate.
- `u` (ndarray): Field matrix.
- `dx`, `dy` (float): Grid spacing.
- `step` (int): Step size for gradient computation.

**Returns**:
- Tuple of gradient components (\(grad_x, grad_y\)).

---

### 4. `degrade_pos_gen(center_x, center_y, side_length, grid_shape)`
**Purpose**: Generates positions within the degradation area of a cell.

**Parameters**:
- `center_x`, `center_y` (int): Center coordinates of the cell.
- `side_length` (int): Side length of the degradation area.
- `grid_shape` (tuple): Shape of the simulation grid.

**Returns**:
- Array of degraded positions.

---

## Class: `Cell_2`
### Description
A class representing a chemotactic cell in the simulation.

**Key Attributes**:
- `grid` (ndarray): Simulation grid.
- `pos_x`, `pos_y` (int): Cell's starting position.
- `shape` (list): Defines the cell's shape (circle or custom).
- `degradation_area` (int): Size of the degradation area.
- `secretion` (bool): Whether the cell secretes attractant.
- `nR` (int): Defines the number of receptors the cell contains
- `k_cat` (float): Defines the catalytic constant per receptor

---

**Key Methods**:

#### `compute_gradient(u, dx, dy, step=2)`
Calculates the local gradient of the chemoattractant field.

#### `update_pos_grad(u, dx, dy, sensitivity, time_curr, dt, grid_shape, step=2)`
Updates the cell's position based on the chemoattractant gradient and random movement.

#### `fitness_function()`
Calculates the fitness score of a cell based on gradient magnitude.

#### `get_position_history(type="list")`
Retrieves the history of the cell's positions in a specified format (list, dataframe, or dictionary).

#### `calc_grad_np(u)`
**Purpose**: Computes the diffusion of chemoattractant across the grid.

**Parameters**:
- `u` (ndarray): Field matrix.

**Returns**:
- Updated field matrix after diffusion.

### `update_cell(c, u, dx, dy, counter, dt, grid_size)`
**Purpose**: Updates the state of a single cell, including its position and degradation behavior. This function is decorated with the Dask's 'Delayed' decorator. The process runs in parallel with other instantiated agents

**Parameters**:
- `c` (Cell_2): Cell object.
- `u` (ndarray): Field matrix.
- `dx`, `dy`, `dt` (float): Simulation parameters.
- `counter` (int): Current timestep.
- `grid_size` (tuple): Dimensions of the grid.

**Returns**:
- Updated cell object.

---
## Helper Functions

#### `merge_cell_pos_df`
**Purpose** Merges dataframes of cell's movement data into a single dataframe

**Parameters**:
- `cells` (Array [Cell]): Array containing the Cell's whose positional dataframes are to be merged
- `on= "time_step"` (String): Which column to merge dataframes on, default = "time_step"

#### `compute_distances_cells`
**Purpose**: Computes pairwise distances and differences in \(x\)- and \(y\)-coordinates between all cells in a dataframe.

**Parameters**:
- `merged_df` (pd.DataFrame): A dataframe containing the positions of all cells, with columns for \(x\)- and \(y\)-coordinates.

**Returns**:
- `pd.DataFrame`: An updated dataframe with additional columns for pairwise distances and differences in \(x\)- and \(y\)-coordinates for all cell pairs.

#### `plot_statistics`
**Purpose**: Plots statistics from the simulation, such as effective stimulus and gradient magnitude, with optional polynomial trendlines.

**Parameters**:
- `df` (pd.DataFrame): Dataframe containing the data to plot.
- `columns` (list): A list of column names, where:
  - The first element is the \(x\)-axis column.
  - The second element is the primary \(y\)-axis column.
  - The third element is the secondary \(y\)-axis column.
- `polynomial_fit` (bool): Whether to fit and plot polynomial trendlines. Default is `True`.
- `degree` (int): Degree of the polynomial for trendlines. Default is `3`.

**Returns**:
- None (Generates an interactive plot using Plotly).


## `plot_cell_history`
**Purpose**: Plots the trajectory of a single cell's movement over time with a gradient line colored by time steps. The function is specifically designed for plotting trajectory of a single cell

**Parameters**:
- `cell_history` (numpy.ndarray): A 2D array where each row represents a time step and contains the following columns:
  - `[time_step, y_position, x_position]`.

**Returns**:
- None (Generates a plot displaying the cell's movement trajectory, with the line color representing time steps. The start and end points are annotated with green and red markers, respectively.)

#### `plot_history_2`
**Purpose**: Plots the trajectory of cell movement with a gradient line colored by time steps. It is an update version of `plot_cell_history` which is capable of generating cell path visualisations for multiple cells

**Parameters**:
- `position_history` (numpy.ndarray): A 2D array where each row represents a time step and contains `[time, y_position, x_position]`.
- `ax` (matplotlib.axes.Axes, optional): An existing matplotlib Axes object to plot on. If None, a new figure and axes are created.
- `color` (str or None, optional): Color for the base path with markers. If None, the default color is used.
- `cmap` (str, optional): Colormap to use for the gradient line. Default is `'viridis'`.
- `label` (str, optional): Label for the cell path. Default is `'Cell Path'`.

**Returns**:
- `fig` (matplotlib.figure.Figure): The matplotlib figure object.
- `ax` (matplotlib.axes.Axes): The matplotlib axes object.
- `lc` (matplotlib.collections.LineCollection): The LineCollection object representing the gradient line.

**Notes**:
- The function creates a gradient line plot where the color of the line represents the time steps.
- It also annotates the start and end points of the trajectory.

----
#### Reinforcement Learning Functions

### 1. `fitness_function()`
**Purpose**: Evaluates the fitness of an agent (cell) based on its interaction with the gradient field.

### 2. `RL Training Loop`
Iterates through epochs to train agents by optimizing fitness-related parameters like degradation rates and secretion probabilities.

---

For further details, refer to the comments in the source code or create an issue on GitHub for clarification!


