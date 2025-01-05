# PyChemoTax: Reinforcement Learning-Driven Chemotaxis Simulation

PyChemoTax is a Python-based framework designed to simulate chemotaxis-driven cellular dynamics in customizable environments. Leveraging both computational efficiency and biological realism, this project includes reinforcement learning capabilities for optimizing agent behavior, enabling the exploration of emergent phenomena in cell populations.

## Features

- **Simulations of Chemotactic Behavior**: Models agents influenced by chemoattractant gradients using Michaelis-Menten kinetics.
- **Reinforcement Learning (RL) Integration**: A proof-of-concept RL module allows agents to optimize parameters such as degradation rate and secretion probability.
- **Highly Customizable Agents**: Define cell shapes, receptor density, sensitivity, degradation areas, and more.
- **Efficient Simulation**: Employs JIT compilation, parallelism with Dask, and vectorized operations for scalability.
- **Visualization**: Real-time visualization of agent dynamics and gradient fields using Matplotlib.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/<your-repo-name>/pychemotax.git
cd pychemotax
```

Install dependencies:

```bash
pip install -r requirements.txt
```

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
python CompPra_V2.py
```

This will execute a reinforcement learning simulation where agents adapt their behavior over multiple epochs.

### Customizing Parameters

Key parameters are defined in the script and can be modified to suit specific needs:

- **Environment**: Grid size, initial chemoattractant distribution, and diffusion properties.
- **Agents**: Number of agents, shape, receptor density, degradation area, and secretion behavior.
- **Reinforcement Learning**: Number of epochs, time steps per epoch, and fitness function.

### Visualization

The simulation provides real-time visualization of the chemoattractant field and agent dynamics. At the end of the simulation, you can also plot:

- Fitness over time: Displays how the average fitness of agents evolves during training.

```python
# Example of fitness plotting
plt.plot(range(1, epochs), fitness_over_time, marker='o', label='Average Fitness')
plt.xlabel('Epoch')
plt.ylabel('Average Fitness')
plt.title('Fitness Over Time During Reinforcement Learning')
plt.legend()
plt.show()
```

## Known Issues

- Agents may exhibit unexpected movement patterns due to gradient calculation biases. See `compute_gradient` for potential fixes.
- The `refactor_RL` branch contains experimental features and may produce unstable results.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

### Branching Strategy

- **main**: Stable and production-ready.
- **refactor\_RL**: Development and experimental features.

Follow [GitHub Flow](https://guides.github.com/introduction/flow/) for submitting contributions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Thanks to the contributors and community for feedback and testing.
- Inspired by computational biology and the need for scalable, realistic chemotaxis simulations.

## Contact

For questions or suggestions, please create an issue or reach out at [your-email@example.com](mailto\:your-email@example.com).

