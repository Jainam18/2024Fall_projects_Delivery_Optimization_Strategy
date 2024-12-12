import networkx as nx
import osmnx as ox
import random
from networkx.algorithms.approximation import traveling_salesman_problem as tsp
from scipy.stats import beta, ttest_rel
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

def initialize_graph(place_name):
    """
    Initializes the city graph for a given place.

    Args:
        place_name (str): Name of the place in "City, State, Country" format.

    Returns:
        G (networkx.Graph): The road network graph.
    """
    print(f"Initializing graph for {place_name}...")
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G

def get_fixed_hub_and_scc(G):
    """
    Selects a fixed hub node and extracts the largest SCC containing the hub.

    Args:
        G (networkx.Graph): The road network graph.

    Returns:
        G_scc (networkx.Graph): The largest strongly connected component with the hub.
        hub_node (int): The fixed hub node ID.
    """

    hub_node = list(G.nodes)[0]
    # Extracting the largest strongly connected component (SCC)
    components = nx.strongly_connected_components(G)
    for component in components:
        if hub_node in component:
            G_scc = G.subgraph(component).copy()
            return G_scc, hub_node


def apply_traffic_congestion(G, traffic_impact_probability=0.3):
    for u, v, k, data in G.edges(keys=True, data=True):
        if random.random() < traffic_impact_probability: 
            maxspeed = data.get("speed_kph", 30) 
            if isinstance(maxspeed, str): 
                maxspeed = int(''.join(filter(str.isdigit, maxspeed)))
            elif isinstance(maxspeed, list):  
                if isinstance(maxspeed[0], str):  
                    maxspeed = int(''.join(filter(str.isdigit, maxspeed[0])))  
                else: 
                    maxspeed = float(maxspeed[0])  
            elif isinstance(maxspeed, (int, float)):
                maxspeed = float(maxspeed)
            else:
                maxspeed = 30
            traffic_factor = beta.rvs(2, 5, loc=0, scale=1)
            data["speed_kph"] = maxspeed * (1 - traffic_factor)
    return G

def precompute_shortest_paths(G, hub_node):
    shortest_paths = nx.single_source_dijkstra_path_length(G, hub_node, weight="length")
    return shortest_paths

def generate_delivery_points(G, num_points, hub_node,shortest_paths):
    """
    Generates random delivery points from the graph.

    Args:
        G (networkx.Graph): The road network graph.
        num_points (int): Number of delivery points to generate.
        hub_node (int): The hub node ID.

    Returns:
        delivery_points (list): List of randomly selected delivery point node IDs.
    """
    reachable_nodes = [node for node in G.nodes if node in shortest_paths]
    reachable_nodes.remove(hub_node)
    delivery_points = random.sample(reachable_nodes, min(num_points, len(reachable_nodes)))
    return delivery_points

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
    remaining_points = delivery_points.copy()
    total_time = 0
    completed_deliveries = 0
    current_node = hub_node

    while remaining_points:
        try:
            next_node = min(remaining_points, key=lambda x: nx.shortest_path_length(G, source=current_node, target=x, weight='length'))
            remaining_points.remove(next_node)
            shortest_path = nx.shortest_path(G, source=current_node, target=next_node, weight='length')
            path_time = 0
            for i in range(len(shortest_path) - 1):
                u, v = shortest_path[i], shortest_path[i + 1]
                edge_data = G.get_edge_data(u, v, default={})
                length = edge_data[0]['length']  
                max_speed = edge_data[0]['speed_kph'] 
                if isinstance(max_speed, list): 
                    max_speed = float(max_speed[0])
                path_time += (length / 1000) / max_speed
            total_time += path_time
            if total_time > time_limit:
                print("[INFO] Time limit exceeded. Ending deliveries.")
                break
            completed_deliveries += 1
            current_node = next_node 
        except nx.NetworkXNoPath:
            print(f"[WARNING] No path between {current_node} and {next_node}. Skipping.")
            break
    return completed_deliveries, total_time

def remove_all_duplicates(path):
        """Removes all duplicates while preserving the order of traversal."""
        visited = set()
        unique_path = []
        for node in path:
            if node not in visited:
                visited.add(node)
                unique_path.append(node)
        return unique_path

def solve_tsp_and_calculate_deliveries(G, hub, delivery_points, time_limit, weight='length'):
    """
    Solves TSP and calculates deliveries and travel time.

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
    for u, v in complete_graph.edges():
        try:
            path_length = nx.shortest_path_length(G, u, v, weight=weight)
            complete_graph[u][v]['weight'] = path_length
        except nx.NetworkXNoPath:
            complete_graph[u][v]['weight'] = float('inf')  
    try:
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
                    print("[INFO] Time limit exceeded. Ending deliveries.")
                    return total_time, completed_deliveries, tsp_path_unique
            if v in delivery_points:
                completed_deliveries += 1
        except nx.NetworkXNoPath:
            print(f"[WARNING] No path between {u} and {v}. Skipping segment.")
            continue
    return total_time, completed_deliveries, tsp_path

def results_to_dataframe(all_results):
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


def simulation_hypothesis_2_parallel(G, num_deliveries, time_limit, iterations):
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
    G_scc, hub_node = get_fixed_hub_and_scc(G)
    shortest_paths = precompute_shortest_paths(G_scc, hub_node)

    all_results = []

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1} starting...")
        tasks = []
        for num in num_deliveries:

            delivery_points = generate_delivery_points(G_scc, num, hub_node, shortest_paths)
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

def plot_time_comparison(num_deliveries_list, delivery_time_greedy, delivery_time_tsp, comp_time_greedy, comp_time_tsp):
    """
    Plots separate graphs for delivery time and computational time for Greedy and TSP approaches.

    Args:
        num_deliveries_list (list): List of delivery sizes.
        delivery_time_greedy (list): Delivery times for the Greedy approach.
        delivery_time_tsp (list): Delivery times for the TSP approach.
        comp_time_greedy (list): Computational times for the Greedy approach.
        comp_time_tsp (list): Computational times for the TSP approach.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(num_deliveries_list, delivery_time_greedy, marker='o', label="Greedy Approach", color='blue')
    plt.plot(num_deliveries_list, delivery_time_tsp, marker='s', label="TSP Approach", color='cyan')
    plt.title("Comparison of Delivery Times for Greedy vs TSP", fontsize=16)
    plt.xlabel("Number of Deliveries", fontsize=14)
    plt.ylabel("Delivery Time (hours)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(num_deliveries_list, comp_time_greedy, marker='x', label="Greedy Approach", color='red')
    plt.plot(num_deliveries_list, comp_time_tsp, marker='^', label="TSP Approach", color='orange')
    plt.title("Comparison of Computational Times for Greedy vs TSP", fontsize=16)
    plt.xlabel("Number of Deliveries", fontsize=14)
    plt.ylabel("Computational Time (seconds)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Inputs
    place_name = "Champaign, Illinois, USA"
    num_deliveries = [200,400,600,800,1000]
    time_limit = 8  # in hours
    iterations = 5
    G = initialize_graph(place_name)
    G = apply_traffic_congestion(G)
    all_results = simulation_hypothesis_2_parallel(G, num_deliveries, time_limit, iterations)
    print(all_results)
    df = results_to_dataframe(all_results)
    print(df.head())