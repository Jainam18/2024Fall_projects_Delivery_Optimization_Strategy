from Utility import *
import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta
import pandas as pd


def plan_routes(G, hub_node, delivery_points, time_limit, truck_type, truck_capacity):
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
                if truck_type == 'Truck A (Large)':
                    speed = edge_data[0]['speed_truck_a']
                elif truck_type == 'Truck B1 (Small)' or truck_type=='Truck B2 (Small)':
                    speed = edge_data[0]['speed_truck_a'] 
                else:
                    return "Invalid Truck Type given"
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
        if completed_deliveries==truck_capacity:
            shortest_path = nx.shortest_path(G, source=current_node, target=hub_node, weight='length')
            for i in range(len(shortest_path) - 1):
                u, v = shortest_path[i], shortest_path[i + 1]
                edge_data = G.get_edge_data(u, v, default={})
                edge_length = edge_data[0]['length']  
                speed = edge_data[0]['speed_kph'] 
                if isinstance(speed, list): 
                    speed = float(speed[0])
                path_time += (edge_length / 1000) / speed
            total_time += path_time
            current_node = hub_node
    return completed_deliveries, total_time

# Run a single simulation for Truck A or Truck B
def run_single_simulation(args):
    G, hub_node, delivery_points, time_limit, truck_type, truck_capacity = args
    return plan_routes(G, hub_node, delivery_points, time_limit, truck_type, truck_capacity)


# Monte Carlo Simulation
def run_monte_carlo_simulation(place_name, num_deliveries, num_simulations, time_limit):
    print("Starting Monte Carlo Simulation...")
    G = initializing_graph(place_name, 1)
    G = apply_traffic_congestion(G, 1)
    G_scc, hub_node = get_fixed_hub_and_scc(G)

    shortest_paths = precompute_shortest_paths(G_scc, hub_node)
    results_data = []
    # iterations = []
    # truck_a_percentages = []
    # truck_b_percentages = []
    # truck_a_time = []
    # truck_b_time = []

    for iteration in range(1,num_simulations+1):
        print(f"\nSimulation {iteration}: Running with {num_deliveries} deliveries...")
        for deliveries in num_deliveries:
            delivery_points = generate_delivery_points(G_scc, deliveries, hub_node,shortest_paths)

            # Truck capacities: Large Truck (200), Small Trucks (100 each)
            args_a = (G_scc, hub_node, delivery_points, time_limit, "Truck A (Large)", 200)
            args_b1 = (G_scc, hub_node, delivery_points[:len(delivery_points) // 2], time_limit,  "Truck B1 (Small)",100)
            args_b2 = (G_scc, hub_node, delivery_points[len(delivery_points) // 2:], time_limit,  "Truck B2 (Small)",100)

            with ProcessPoolExecutor(max_workers=3) as executor:
                results = executor.map(run_single_simulation, [args_a, args_b1, args_b2])
            (completed_del_a,total_time_a), (completed_del_b1,total_time_b1), (completed_del_b2,total_time_b2) = results
            total_time_b = max(total_time_b1, total_time_b2)
            completed_del_b = completed_del_b1 + completed_del_b2

            # Calculate percentages based on deliveries completed within the time limit
            truck_a_percentage = (completed_del_a / deliveries) * 100
            truck_b_percentage = (completed_del_b / deliveries) * 100
            results_data.append({
                'iteration': iteration,
                'num_deliveries': deliveries,
                'truck_a_percentage': truck_a_percentage,
                'truck_b_percentage': truck_b_percentage,
                'truck_a_time': total_time_a,
                'truck_b_time': total_time_b
            })
    df_results = pd.DataFrame(results_data)
    return df_results


# Plot comparison graph
def plot_comparison_efficiency(x_orders, truck_a_percentages, truck_b_percentages):
    plt.plot(x_orders, truck_a_percentages, label="Truck A (Large)", color="blue", marker="o")
    plt.plot(x_orders, truck_b_percentages, label="Truck B (Max of B1 and B2)", color="green", marker="x")
    plt.xlabel("Number of Orders")
    plt.ylabel("Percentage of Orders Completed (%)")
    plt.title("Percentage of Orders Completed Within Time Limit")
    plt.legend()
    plt.grid()
    plt.show()

def plot_comparison_time(x_orders, truck_a_time, truck_b_time):
    plt.plot(x_orders, truck_a_time, label="Truck A (Large)", color="blue", marker="o")
    plt.plot(x_orders, truck_b_time, label="Truck B (Max of B1 and B2)", color="green", marker="x")
    plt.xlabel("Number of Orders")
    plt.ylabel("Time taken to deliver all orders")
    plt.title("Comparison of time taken to deliver order by small and big truck")
    plt.legend()
    plt.grid()
    plt.show()


# Run the simulation
if __name__ == "__main__":
    place_name = "Champaign, Illinois, USA"
    num_deliveries = [250,500,750,1000,1250,1500]
    num_simulations = 8
    time_limit = 9 #in hours 8am to 5pm

    df = run_monte_carlo_simulation(
        place_name, num_deliveries, num_simulations, time_limit
    )
    df.to_csv('Results_Hypothesis_1.csv')
    df_iteration = df[df['iteration']==1]
    truck_a_eff = df_iteration['truck_a_percentage']
    truck_b_eff = df_iteration['truck_b_percentage']
    truck_a_time = df_iteration['truck_a_time']
    truck_b_time = df_iteration['truck_b_time']
    plot_comparison_time(num_deliveries,truck_a_time,truck_b_time)
    plot_comparison_efficiency(num_deliveries,truck_a_eff,truck_b_eff)