import Utility
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import os

def plan_routes_with_greedy(G, hub_node, delivery_points, time_limit):
    """
    Plans delivery routes using a greedy approach.

    Args:
        G (networkx.Graph): The road network graph.
        hub_node (int): The hub node ID.
        delivery_points (list): List of delivery point node IDs.
        time_limit (float): Time limit for deliveries (in hours).

    Returns:
        completed_deliveries (int): Number of completed deliveries.
        total_time (float): Total time taken for deliveries (in hours).
    """ 
    remaining_locations = delivery_points.copy()
    total_time = 0
    completed_deliveries = 0
    current_node = hub_node

    while remaining_locations:
        # Implementation of Exception Handling 
        try:
            next_node = min(remaining_locations, key=lambda x: nx.shortest_path_length(G, source=current_node, target=x, weight='length')) ## Implementation of lambda functions 
            remaining_locations.remove(next_node) #Removing the node we visit from the list
            shortest_path = nx.shortest_path(G, source=current_node, target=next_node, weight='length') # Listing the route from current to next node
            path_time = 0
            # for loop to calculate total time taken to travel to current to next node 
            for i in range(len(shortest_path) - 1):
                u, v = shortest_path[i], shortest_path[i + 1]
                edge_data = G.get_edge_data(u, v, default={})
                edge_length = edge_data[0]['length']  
                speed = edge_data[0]['speed_kph'] 
                if isinstance(speed, list): 
                    speed = float(speed[0])
                path_time += (edge_length / 1000) / speed
            total_time += path_time
            # time limit condition check  
            if total_time > time_limit:
                print("Time limit exceeded. Ending deliveries.")
                break
            completed_deliveries += 1
            current_node = next_node 
        
        except nx.NetworkXNoPath:
            print(f"No path between {current_node} and {next_node}. Skipping.")
            break
    return completed_deliveries, total_time

def remove_all_duplicates(path):
    """Removes all duplicates while preserving the order of traversal.
    >>> l = remove_all_duplicates([12323,12344,12675,16734,12344, 16734])
    >>> len(l) == len([12323,12344,12675,16734,12344, 16734])
    False
    >>> s = set([12323,12344,12675,16734,12344, 16734])
    >>> len(s)==len(l)
    True
    """
    visited = set()
    unique_path = []
    for node in path:
        if node not in visited:
            visited.add(node)
            unique_path.append(node)
    return unique_path

def solve_tsp_and_calculate_deliveries(G, hub, delivery_points, time_limit, weight='length'):
    """
    Implements TSP approach to get the optimal route to cover all the delivery locations and calculates deliveries and travel time.

    Args:
        G (networkx.Graph): City graph with travel distances and speeds.
        hub (int): Hub node ID.
        delivery_points (list): List of delivery point node IDs.
        time_limit (float): Time limit for deliveries (in hours).
        weight (str): Weight attribute for calculating distances.

    Returns:
        total_time (float): Total travel time in hours.
        completed_deliveries (int): Number of deliveries completed.
        tsp_path (list): The TSP route including the hub.
    """
    tsp_nodes = [hub] + delivery_points
    complete_graph = nx.complete_graph(tsp_nodes)
    # creates a complete graph with all the nodes where delivery has to be made 

    # calculating the distance between two nodes and assiging it to the edge weight 
    for u, v in complete_graph.edges():
        try:
            path_length = nx.shortest_path_length(G, u, v, weight=weight)
            complete_graph[u][v]['weight'] = path_length
        except nx.NetworkXNoPath:
            complete_graph[u][v]['weight'] = float('inf')  
    try:
        # running the tsp algorithm to find the optimal route to traverse all the delivery points 
        tsp_path = nx.approximation.traveling_salesman_problem(complete_graph, cycle=True)
    except nx.NetworkXError:
        print("[ERROR] TSP could not be solved.")
        return None, None, None
    
    tsp_path_unique = remove_all_duplicates(tsp_path)
    total_time = 0  
    completed_deliveries = 0
    for i in range(len(tsp_path_unique) - 1):
        u, v = tsp_path_unique[i], tsp_path_unique[i + 1]
        try:
            shortest_path = nx.shortest_path(G, u, v, weight=weight)
            for j in range(len(shortest_path) - 1):
                segment_u, segment_v = shortest_path[j], shortest_path[j + 1]
                edge_data = G.get_edge_data(segment_u, segment_v, default={})
                path_length = edge_data[0][weight]
                max_speed = edge_data[0]['speed_kph'] 
                if isinstance(max_speed, list):
                    max_speed = float(max_speed[0])
                travel_time = (path_length / 1000) / max_speed
                total_time += travel_time
                if total_time > time_limit:
                    print("Time limit exceeded. Ending deliveries.")
                    return total_time, completed_deliveries, tsp_path_unique
            if v in delivery_points:
                completed_deliveries += 1
        except nx.NetworkXNoPath:
            print(f"No path between {u} and {v}. Skipping segment.")
            continue
    return total_time, completed_deliveries, tsp_path

def run_single_simulation(strategy, G_scc, hub_node, delivery_points, time_limit):
    if strategy == "greedy":
        start_time = time.time()
        completed_deliveries, total_time = plan_routes_with_greedy(G_scc, hub_node, delivery_points, time_limit)
        computation_time = time.time() - start_time
        return {
            "strategy": "greedy",
            "num_deliveries":len(delivery_points),
            "completed_deliveries": completed_deliveries,
            "total_time": total_time,
            "computation_time": computation_time,
        }
    elif strategy == "tsp":
        start_time = time.time()
        total_time, completed_deliveries, _ = solve_tsp_and_calculate_deliveries(G_scc, hub_node, delivery_points, time_limit)
        computation_time = time.time() - start_time
        return {
            "strategy": "tsp",
            "num_deliveries":len(delivery_points),
            "completed_deliveries": completed_deliveries,
            "total_time": total_time,
            "computation_time": computation_time,
        }
    else:
        raise ValueError("Invalid strategy. Use 'greedy' or 'tsp'.")

def run_simulation_task(args):
    """
    Wrapper function to execute a simulation task.

    Args:
        args (tuple): A tuple containing arguments for run_single_simulation.

    Returns:
        dict: Results of the simulation.
    """
    return run_single_simulation(*args)


def simulation_hypothesis_2(G, num_deliveries, time_limit, iterations):
    """
    Runs simulations in parallel for Greedy and TSP approaches across multiple iterations.

    Args:
        G (networkx.Graph): The road network graph.
        num_deliveries (list): List of delivery sizes.
        time_limit (float): Time limit for deliveries (in hours).
        iterations (int): Number of iterations.

    Returns:
        all_results (list): List of results across iterations and strategies.
    """
    G_scc, hub_node = Utility.get_fixed_hub_and_scc(G)
    shortest_paths = Utility.precompute_shortest_paths(G_scc, hub_node)

    all_results = []

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1} starting...")
        tasks = []
        for num in num_deliveries:

            delivery_points = Utility.generate_delivery_points(G_scc, num, hub_node, shortest_paths)
            tasks.append(("greedy", G_scc, hub_node, delivery_points, time_limit))
            tasks.append(("tsp", G_scc, hub_node, delivery_points, time_limit))
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(run_simulation_task, tasks))
        iteration_results = {
            "iteration": iteration + 1,
            "results": results
        }
        all_results.append(iteration_results)
    return all_results

def results_to_dataframe(all_results):
    """
    Converts the resultant output of the simulation to a dataframe 

    Args:
        all_results (list): List of results across iterations and strategies.
    Returns:
        df: A dataframe with columns iteration, results, strategy, num_deliveries, completed_deliveries, total_time, computation_time
    >>> results = [{'iteration': 1, 'results': [{'strategy': 'greedy', 'num_deliveries': 200, 'completed_deliveries': 200, 'total_time': 3.3073054620939213, 'computation_time': 44.862642765045166}, {'strategy': 'tsp', 'num_deliveries': 200, 'completed_deliveries': 200, 'total_time': 2.982897033143903, 'computation_time': 44.8230299949646}, {'strategy': 'greedy', 'num_deliveries': 400, 'completed_deliveries': 400, 'total_time': 4.768264361218804, 'computation_time': 178.12589693069458}, {'strategy': 'tsp', 'num_deliveries': 400, 'completed_deliveries': 400, 'total_time': 4.534652384798603, 'computation_time': 192.61627388000488}, {'strategy': 'greedy', 'num_deliveries': 600, 'completed_deliveries': 600, 'total_time': 5.406827949407598, 'computation_time': 443.03445506095886}, {'strategy': 'tsp', 'num_deliveries': 600, 'completed_deliveries': 600, 'total_time': 5.295299723913028, 'computation_time': 490.08500504493713}, {'strategy': 'greedy', 'num_deliveries': 800, 'completed_deliveries': 800, 'total_time': 6.300763371752854, 'computation_time': 1818.429852962494}, {'strategy': 'tsp', 'num_deliveries': 800, 'completed_deliveries': 800, 'total_time': 6.2962532451729585, 'computation_time': 1926.2362020015717}, {'strategy': 'greedy', 'num_deliveries': 1000, 'completed_deliveries': 1000, 'total_time': 7.176981730294133, 'computation_time': 2100.916377067566}, {'strategy': 'tsp', 'num_deliveries': 1000, 'completed_deliveries': 1000, 'total_time': 6.933875444838426, 'computation_time': 2331.4957840442657}]}]
    >>> df = results_to_dataframe(results)
    >>> columns = ['iteration', 'strategy', 'num_deliveries', 'completed_deliveries', 'total_time', 'computation_time']
    >>> columns == list(df.columns)
    True
    >>> df['iteration'].dtype == int
    True
    >>> len(df) > 0
    True
    """
    flattened_data = []
    for iteration_result in all_results:
        iteration = iteration_result["iteration"]
        for result in iteration_result["results"]:
            flattened_data.append({
                "iteration": iteration,
                "strategy": result["strategy"],
                "num_deliveries": result["num_deliveries"],
                "completed_deliveries": result["completed_deliveries"],
                "total_time": result["total_time"],
                "computation_time": result["computation_time"],
            })
    df = pd.DataFrame(flattened_data)
    return df

def plot_over_iterations(df):
    """
    It plots a graph which compares the total time taken to deliver the orders by two different approaches over mulitple set of iterations
    """
    iterations = sorted(df['iteration'].unique())
    strategies = ['greedy', 'tsp']
    cmap = cm.get_cmap('tab10', len(iterations))
    colors = {iteration: cmap(i) for i, iteration in enumerate(iterations)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy'] == strategy]
        for iteration in iterations:
            iteration_data = strategy_data[strategy_data['iteration'] == iteration]
            axes[i].plot(
                iteration_data['num_deliveries'], 
                iteration_data['total_time'], 
                label=f"Iteration {iteration}", 
                color=colors[iteration], 
                marker='o',
                alpha=0.7
            )
        axes[i].set_title(f"{strategy.capitalize()} Approach", fontsize=14)
        axes[i].set_xlabel("Number of Deliveries", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Total Time (hours)", fontsize=12)
        axes[i].grid(alpha=0.5)
        axes[i].legend(title="Iterations", fontsize=10, loc="best")
    plt.suptitle("Delivery Time vs Number of Deliveries Across Iterations", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_comparison_time_and_comp(df):
    """
    Plots over an single iteration to show comparison of how the tsp time and the greedy time varies over the different number of delivery orders.
    """
    iteration = 1
    iteration_data = df[df['iteration'] == iteration]
    approaches = iteration_data['strategy'].unique()  # ['greedy', 'tsp']
    num_deliveries = iteration_data['num_deliveries'].unique()
    comp_time = iteration_data.pivot(index='num_deliveries', columns='strategy', values='computation_time')
    total_time = iteration_data.pivot(index='num_deliveries', columns='strategy', values='total_time')
    x = np.arange(len(comp_time.index))
    width = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot comp_time
    ax1.bar(x - width/2, comp_time['greedy'], width, label='greedy')
    ax1.bar(x + width/2, comp_time['tsp'], width, label='tsp')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comp_time.index)
    ax1.set_title('Computation Time')
    ax1.legend()
    # Plot total_time
    ax2.bar(x - width/2, total_time['greedy'], width, label='greedy')
    ax2.bar(x + width/2, total_time['tsp'], width, label='tsp')
    ax2.set_xticks(x)
    ax2.set_xticklabels(total_time.index)
    ax2.set_title('Total Time')
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Inputs
    place_name = "Champaign, Illinois, USA"
    num_deliveries = [20,40,60,80,100]
    time_limit = 8  # in hours
    iterations = 1
    G = Utility.initializing_graph(place_name,hypothesis=2)
    G = Utility.apply_traffic_congestion(G,hypothesis=2)
    all_results = simulation_hypothesis_2(G, num_deliveries, time_limit, iterations)
    print(all_results)
    df = results_to_dataframe(all_results)
    print(df.head())
    # df.to_csv('Result.csv')
    output_file = 'Result_Hypothesis_2.csv'
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', index=False, header=False)
    else:
        df.to_csv(output_file, index=False)
    print(f"Results appended to {output_file}.")
    plot_comparison_time_and_comp(df)
    plot_over_iterations(df)