# Network Percolation Symmetry

Simulations to explore the relationships between graph/network percolation and symmetries.


## Introduction

Network percolation and symmetry breaking are key concepts in understanding the fundamental properties of complex systems. Percolation theory studies the behavior of connected clusters in a random graph, which can model phenomena such as the spread of diseases, information flow, and the robustness of networks. Symmetry breaking, on the other hand, is a process where a symmetric state leads to an asymmetric state, which is crucial in phase transitions and the emergence of complex structures.

In this repository, we explore the relationship between network percolation and symmetry breaking. By analyzing the symmetry measures of random graphs, we can gain insights into how structural properties of networks change as they transition from a disordered to an ordered state. This represents a fundamental universal model of the symmetry of form in systems, providing a deeper understanding of the underlying principles governing the behavior of various natural and artificial networks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use the code in this repository, you will need to have Python installed along with several libraries. You can install the required libraries using the following command:

```bash
pip install networkx numpy matplotlib pynauty pandas seaborn
```

## Usage

You can run the main script to perform the graph symmetry analysis and visualize the results. The script allows you to specify the number of nodes, the range of mean degrees, and the number of trials.

Example command to run the script:

```bash
python graph_symmetries.py --nodes 50 --z_start 0 --z_end 4 --z_step 0.1 --trials 50
```

## Functions

### `generate_random_graph(n, z)`

Generates a random graph using the Erdős-Rényi model.

- **Parameters:**
  - `n` (int): Number of nodes in the graph.
  - `z` (float): Mean degree.

- **Returns:**
  - `networkx.Graph`: The generated random graph.

### `calculate_symmetry_measures(graph)`

Calculates the symmetry measures of a graph using PyNauty.

- **Parameters:**
  - `graph` (networkx.Graph): The graph for which to calculate symmetry measures.

- **Returns:**
  - `float`: Size of the automorphism group.
  - `int`: Number of orbits.

### `run_trials(n, z_values, trials)`

Runs multiple trials for different z values and computes symmetry measures.

- **Parameters:**
  - `n` (int): Number of nodes in each graph.
  - `z_values` (list of float): List of mean degree values to test.
  - `trials` (int): Number of trials to run for each z value.

- **Returns:**
  - `dict`: Dictionary with z values as keys and lists of symmetry measures as values.

### `visualize_results(z_values, symmetry_data)`

Plots the results of the symmetry analysis.

- **Parameters:**
  - `z_values` (list of float): List of mean degree values.
  - `symmetry_data` (dict): Dictionary containing symmetry measures for each z value.

### `transform_data_for_plotting(symmetry_data)`

Transforms the symmetry data into a format suitable for plotting with Seaborn.

- **Parameters:**
  - `symmetry_data` (dict): Dictionary containing symmetry measures for each z value.

- **Returns:**
  - `pandas.DataFrame`: DataFrame suitable for Seaborn plotting.

### `seaborn_visualize(df)`

Visualizes the given DataFrame using Seaborn's line plot with error bars.

- **Parameters:**
  - `df` (pandas.DataFrame): DataFrame containing the data to plot.

### `seaborn_visualize_dual_axis(df)`

Visualizes the given DataFrame using Seaborn's line plot with dual y-axes.

- **Parameters:**
  - `df` (pandas.DataFrame): DataFrame containing the data to plot.

### `seaborn_visualize_triple_axis(df)`

Visualizes the given DataFrame using Seaborn's line plot with triple y-axes.

- **Parameters:**
  - `df` (pandas.DataFrame): DataFrame containing the data to plot.

## Examples

To run the script with default parameters:

```bash
python graph_symmetries.py
```

To customize the parameters:

```bash
python graph_symmetries.py --nodes 100 --z_start 0 --z_end 5 --z_step 0.2 --trials 100
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

Feel free to modify the content as needed for your specific use case.
