import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pynauty import Graph, autgrp
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import argparse

def generate_random_graph(n, z):
    """
    Generates a random graph using the Erdős-Rényi model.

    Parameters:
    n (int): Number of nodes in the graph.
    z (float): Mean degree.

    Returns:
    networkx.Graph: The generated random graph.
    """
    p = z / (n - 1)
    return nx.erdos_renyi_graph(n, p)


def calculate_symmetry_measures(graph):
    """
    Calculates the symmetry measures of a graph using PyNauty.

    Parameters:
    graph (networkx.Graph): The graph for which to calculate symmetry measures.

    Returns:
    float: Size of the automorphism group.
    int: Number of orbits.
    """
    # Convert networkx graph to pynauty graph
    pynauty_graph = Graph(len(graph), directed=False)
    pynauty_graph.set_adjacency_dict({node: set(neighbors) for node, neighbors in graph.adjacency()})

    # Calculate automorphism group
    autgrp_data = autgrp(pynauty_graph)
    
    # Extract the size of the automorphism group
    aut_group_size = autgrp_data[1]

    # Calculate the number of orbits using the fourth element of the tuple
    orbits = autgrp_data[3]
    orbit_count = len(set(orbits))  # Count the unique values in orbits list

    return aut_group_size, orbit_count



def run_trials(n, z_values, trials):
    """
    Runs multiple trials for different z values and computes symmetry measures.

    Parameters:
    n (int): Number of nodes in each graph.
    z_values (list of float): List of mean degree values to test.
    trials (int): Number of trials to run for each z value.

    Returns:
    dict: Dictionary with z values as keys and lists of symmetry measures as values.
    """
    results = {z: {'aut_group_sizes': [], 'orbit_counts': []} for z in z_values}
    for z in z_values:
        for _ in range(trials):
            graph = generate_random_graph(n, z)
            aut_group_size, orbit_count = calculate_symmetry_measures(graph)
            results[z]['aut_group_sizes'].append(aut_group_size)
            results[z]['orbit_counts'].append(orbit_count)
    return results

def visualize_results(z_values, symmetry_data):
    """
    Plots the results of the symmetry analysis.

    Parameters:
    z_values (list of float): List of mean degree values.
    symmetry_data (dict): Dictionary containing symmetry measures for each z value.
    """
    plt.figure(figsize=(12, 6))

    # Plotting the size of the automorphism group
    plt.subplot(1, 2, 1)
    for z in z_values:
        plt.plot([z] * len(symmetry_data[z]['aut_group_sizes']), symmetry_data[z]['aut_group_sizes'], 'o')
    plt.xlabel('Mean Degree (z)')
    plt.ylabel('Size of Automorphism Group')
    plt.title('Automorphism Group Size vs Mean Degree')

    # Plotting the number of orbits
    plt.subplot(1, 2, 2)
    for z in z_values:
        plt.plot([z] * len(symmetry_data[z]['orbit_counts']), symmetry_data[z]['orbit_counts'], 'o')
    plt.xlabel('Mean Degree (z)')
    plt.ylabel('Number of Orbits')
    plt.title('Number of Orbits vs Mean Degree')

    plt.tight_layout()
    plt.show()


def transform_data_for_plotting(symmetry_data):
    """
    Transforms the symmetry data into a format suitable for plotting with Seaborn.

    Parameters:
    symmetry_data (dict): Dictionary containing symmetry measures for each z value.

    Returns:
    pandas.DataFrame: DataFrame suitable for Seaborn plotting.
    """
    records = []
    for z, measures in symmetry_data.items():
        for gcc_size in measures['gcc_sizes']:
            records.append({'Mean Degree (z)': z, 'Measure': 'GCC Size', 'Value': gcc_size})
        for num_clusters in measures['num_clusters']:
            records.append({'Mean Degree (z)': z, 'Measure': 'Number of Clusters', 'Value': num_clusters})
    return pd.DataFrame(records)


def seaborn_visualize(df):
    """
    Visualizes the given DataFrame using Seaborn's lineplot with error bars.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data to plot.
    """
    plt.figure(figsize=(12, 6))

    # Seaborn lineplot with error bars
    sns.lineplot(data=df, x='Mean Degree (z)', y='Value', hue='Measure', style='Measure', markers=True)
    #, err_style='bars')

    plt.ylabel('Measure Value')
    plt.title('Graph Symmetry Measures vs Mean Degree')
    plt.show()


def seaborn_visualize_dual_axis(df):
    """
    Visualizes the given DataFrame using Seaborn's lineplot with dual y-axes.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data to plot.
    """
    plt.figure(figsize=(12, 6))

    # Create the first subplot
    ax1 = sns.lineplot(data=df[df['Measure'] == 'Size of Automorphism Group'], 
                       x='Mean Degree (z)', y='Value', color='b', label='Size of Automorphism Group')
    ax1.set_ylabel('Size of Automorphism Group', color='b')

    # Create a second y-axis for the number of orbits
    ax2 = ax1.twinx()
    sns.lineplot(data=df[df['Measure'] == 'Number of Orbits'], 
                 x='Mean Degree (z)', y='Value', color='r', ax=ax2, label='Number of Orbits')
    ax2.set_ylabel('Number of Orbits', color='r')

    plt.title('Graph Symmetry Measures vs Mean Degree')
    plt.show()


def seaborn_visualize_triple_axis(df):
    plt.figure(figsize=(12, 6))

    # First plot with no automatic legend
    ax1 = sns.lineplot(data=df[df['Measure'] == 'Size of Automorphism Group'], 
                       x='Mean Degree (z)', y='Value', color='b', label='_nolegend_')

    # Second plot with no automatic legend
    ax2 = ax1.twinx()
    sns.lineplot(data=df[df['Measure'] == 'Number of Orbits'], 
                 x='Mean Degree (z)', y='Value', color='r', ax=ax2, label='_nolegend_')

    # Third plot with no automatic legend
    ax3 = ax1.twinx()
    sns.lineplot(data=df[df['Measure'] == 'GCC Size'], 
                 x='Mean Degree (z)', y='Value', color='g', ax=ax3, label='_nolegend_')
    ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis

    # Collecting all the line handles and labels for the custom legend
    lines = [line for line in ax1.get_lines()] + [line for line in ax2.get_lines()] + [line for line in ax3.get_lines()]
    labels = ['Size of Automorphism Group', 'Number of Orbits', 'GCC Size']

    # Create new handles with thicker lines for the legend
    new_handles = [plt.Line2D([], [], c=line.get_color(), linewidth=4) for line in lines]

    # Creating a single legend with thicker lines
    ax1.legend(handles=new_handles, labels=labels, loc='upper left')


    plt.title('Graph Symmetry Measures and GCC Size vs Mean Degree')
    #plt.show()

    plt.savefig('symmetries.pdf')

    
def visualize_percolation_measures(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Mean Degree (z)', y='Value', hue='Measure', style='Measure', markers=True)
    plt.ylabel('Measure Value')
    plt.title('Percolation Measures vs Mean Degree')
    plt.show()


def visualize_percolation_measures_dual_axis(df):
    plt.figure(figsize=(12, 6))

    # Split the DataFrame for different measures
    df_gcc = df[df['Measure'] == 'GCC Size']
    df_clusters = df[df['Measure'] == 'Number of Clusters']

    # Create the first plot (e.g., GCC Size) on the primary y-axis
    ax1 = sns.lineplot(data=df_gcc, x='Mean Degree (z)', y='Value', color='b', label='GCC Size')
    ax1.set_ylabel('GCC Size', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for the Number of Clusters
    ax2 = ax1.twinx()
    sns.lineplot(data=df_clusters, x='Mean Degree (z)', y='Value', color='r', ax=ax2, label='Number of Clusters')
    ax2.set_ylabel('Number of Clusters', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Percolation Measures vs Mean Degree')
    plt.show()


def run_trial_worker(n, z):
    graph = generate_random_graph(n, z)
    gcc_size = len(max(nx.connected_components(graph), key=len)) / n  # Relative size of GCC
    num_clusters = len(list(nx.connected_components(graph))) - 1  # Number of clusters excluding GCC
    return z, gcc_size, num_clusters



def run_trials_parallel(n, z_values, trials):
    results = {z: {'gcc_sizes': [], 'num_clusters': []} for z in z_values}
    trial_args = [(n, z) for z in z_values for _ in range(trials)]

    with ProcessPoolExecutor() as executor:
        for z, gcc_size, num_clusters in executor.map(run_trial_worker, *zip(*trial_args)):
            results[z]['gcc_sizes'].append(gcc_size)
            results[z]['num_clusters'].append(num_clusters)

    return results



def main():
    parser = argparse.ArgumentParser(description='Run graph symmetry trials.')
    parser.add_argument('--nodes', type=int, default=50, help='Number of nodes (default: 50)')
    parser.add_argument('--z_start', type=float, default=0, help='Start value of mean degree z (default: 0)')
    parser.add_argument('--z_end', type=float, default=4, help='End value of mean degree z (exclusive, default: 4)')
    parser.add_argument('--z_step', type=float, default=0.1, help='Step size for mean degree z (default: 0.1)')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials (default: 50)')

    args = parser.parse_args()

    n = args.nodes
    z_values = np.arange(args.z_start, args.z_end, args.z_step)
    trials = args.trials

    symmetry_data = run_trials_parallel(n, z_values, trials)
    df = transform_data_for_plotting(symmetry_data)
    #seaborn_visualize_dual_axis(df)
    #seaborn_visualize_triple_axis(df)

#    visualize_percolation_measures(df)
    visualize_percolation_measures_dual_axis(df)
    
if __name__ == '__main__':
    main()
