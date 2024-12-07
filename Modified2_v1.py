import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta, ttest_rel


# Initializing the street network
def initialize_graph(place_name):
    print("[INFO] Initializing the street network for:", place_name)
    G = ox.graph_from_place(place_name, network_type='drive')
    print("[INFO] Adding speed and travel time attributes...")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Set base speeds for Truck A and Truck B
    for u, v, k, data in G.edges(keys=True, data=True):
        base_speed_b = data.get("speed_kph", 50)  # Default speed for Truck B
        data["speed_truck_b"] = base_speed_b
        data["speed_truck_a"] = max(base_speed_b - 16, 16)  # Truck A is 10 mph (16 kph) slower

    print("[INFO] Graph initialization complete!")
    return G


# Applying traffic congestion using a PERT distribution
def apply_traffic_congestion(G, traffic_impact_probability=0.3):
    print("[INFO] Applying traffic congestion...")
    for u, v, k, data in G.edges(keys=True, data=True):
        if random.random() < traffic_impact_probability:  # Apply congestion to a subset of roads
            # Generate a traffic factor using PERT distribution (alpha=2, beta=5, mode=0.3)
            traffic_factor = beta.rvs(2, 5, loc=0, scale=1)
            # Adjust speeds based on traffic factor
            data["speed_truck_b"] *= (1 - traffic_factor)
            data["speed_truck_a"] *= (1 - traffic_factor)

            # Ensuring that the speed remains realistic
            data["speed_truck_b"] = max(data["speed_truck_b"], 5)  # Minimum 5 kph
            data["speed_truck_a"] = max(data["speed_truck_a"], 5)

    print("[INFO] Traffic congestion applied!")
    return G


# Set a fixed hub node and extract SCC
def get_fixed_hub_and_scc(G):
    print("[INFO] Using a fixed hub node...")
    hub_node = list(G.nodes)[0]
    print(f"[INFO] Fixed hub node set: {hub_node}")

    # Extracting the largest strongly connected component (SCC) containing the hub
    print("[INFO] Extracting largest strongly connected component (SCC)...")
    if not nx.is_strongly_connected(G):
        components = nx.strongly_connected_components(G)
        for component in components:
            if hub_node in component:
                G_scc = G.subgraph(component).copy()
                print(f"[INFO] SCC with hub node contains {len(G_scc.nodes)} nodes.")
                return G_scc, hub_node
    else:
        print("[INFO] The entire graph is strongly connected.")
        return G, hub_node


# Precompute shortest paths from the hub
def precompute_shortest_paths(G, hub_node):
    print("[INFO] Precomputing shortest paths from the hub...")
    shortest_paths = nx.single_source_dijkstra_path_length(G, hub_node, weight="travel_time")
    print("[INFO] Precomputation complete!")
    return shortest_paths


# 5. Generate random delivery points
def generate_delivery_points(G, num_deliveries, hub_node, shortest_paths):
    reachable_nodes = [node for node in G.nodes if node in shortest_paths]
    reachable_nodes.remove(hub_node)
    delivery_points = random.sample(reachable_nodes, min(num_deliveries, len(reachable_nodes)))
    print(f"[INFO] Generated {len(delivery_points)} reachable delivery points.")
    return delivery_points


# 6. Plan delivery routes using precomputed shortest paths
def plan_routes_with_precomputed_paths(G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths):
    print(f"\n[INFO] Deliveries started for {truck_type}...")
    remaining_points = delivery_points.copy()
    total_time = 0  # Track total travel time in hours
    trip_count = 1

    while remaining_points:
        print(f"[INFO] Trip No. {trip_count} for {truck_type}...")
        trip_points = remaining_points[:truck_capacity]
        remaining_points = remaining_points[truck_capacity:]
        current_node = hub_node

        while trip_points:
            try:
                next_node = min(trip_points, key=lambda x: shortest_paths[x])
                travel_time = shortest_paths[next_node] / 3600
                total_time += travel_time
                current_node = next_node
                trip_points.remove(next_node)
            except KeyError as e:
                print(f"[WARNING] Node {e} not found in shortest paths. Skipping.")
                trip_points.remove(e)

        # Return to hub
        try:
            travel_time = shortest_paths[hub_node] / 3600
            total_time += travel_time
        except KeyError as e:
            print(f"[ERROR] Hub node {hub_node} not found in shortest paths.")
            break

        print(f"[INFO] {truck_type} - Returning to hub after trip {trip_count}.")
        trip_count += 1

    print(f"[INFO] {truck_type} - All deliveries completed in {total_time:.2f} hours.")
    return total_time


# 7. Run a single simulation for Truck A or Truck B
def run_single_simulation(args):
    G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths = args
    return plan_routes_with_precomputed_paths(G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths)


# 8. Run the Monte Carlo simulation
def run_monte_carlo_simulation(place_name, initial_deliveries, num_simulations, increment, time_limit):
    print("[INFO] Starting Monte Carlo Simulation...")
    G = initialize_graph(place_name)
    G = apply_traffic_congestion(G)  # Apply traffic congestion
    G_scc, hub_node = get_fixed_hub_and_scc(G)

    # Precompute shortest paths
    shortest_paths = precompute_shortest_paths(G_scc, hub_node)

    x_orders = []
    truck_a_percentages = []
    truck_b_percentages = []

    num_deliveries = initial_deliveries

    for simulation in range(1, num_simulations + 1):
        print(f"\n[INFO] Simulation {simulation}: Running with {num_deliveries} deliveries...")
        delivery_points = generate_delivery_points(G_scc, num_deliveries, hub_node, shortest_paths)

        # Parallel execution for Truck A and Truck B
        args_a = (G_scc, hub_node, delivery_points, 100, "Truck A (Large)", shortest_paths)
        half_deliveries = len(delivery_points) // 2
        args_b1 = (G_scc, hub_node, delivery_points[:half_deliveries], 50, "Truck B1 (Small)", shortest_paths)
        args_b2 = (G_scc, hub_node, delivery_points[half_deliveries:], 50, "Truck B2 (Small)", shortest_paths)

        with ProcessPoolExecutor(max_workers=3) as executor:
            results = executor.map(run_single_simulation, [args_a, args_b1, args_b2])

        # Process results
        time_a, time_b1, time_b2 = results
        time_b = max(time_b1, time_b2)  # Use the maximum time of B1 and B2 for Truck B

        # Calculate percentages of completed orders
        truck_a_percentage = 100 if time_a <= time_limit else (time_limit / time_a) * 100
        truck_b_percentage = 100 if time_b <= time_limit else (time_limit / time_b) * 100

        truck_a_percentages.append(truck_a_percentage)
        truck_b_percentages.append(truck_b_percentage)

        x_orders.append(num_deliveries)
        print(f"[INFO] Simulation {simulation} Results: Truck A: {truck_a_percentage:.2f}%, Truck B: {truck_b_percentage:.2f}%.")
        num_deliveries += increment

    plot_comparison(x_orders, truck_a_percentages, truck_b_percentages)
    return truck_a_percentages, truck_b_percentages


# 9. Plot comparison graph
def plot_comparison(x_orders, truck_a_percentages, truck_b_percentages):
    plt.plot(x_orders, truck_a_percentages, label="Truck A (Large)", color="blue", marker="o")
    plt.plot(x_orders, truck_b_percentages, label="Truck B (Max of B1 and B2)", color="green", marker="x")

    plt.xlabel("Number of Orders")
    plt.ylabel("Percentage of Orders Completed (%)")
    plt.title("Percentage of Orders Completed Within Time Limit")
    plt.legend()
    plt.grid()
    plt.show()


# 10. Perform Statistical Test
def perform_statistical_test(truck_a_percentages, truck_b_percentages):
    t_stat, p_value = ttest_rel(truck_b_percentages, truck_a_percentages)
    print("\n[STATISTICAL TEST RESULTS]")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("Result: Reject the null hypothesis. Truck B performs significantly better than Truck A.")
    else:
        print("Result: Fail to reject the null hypothesis. No significant difference in performance.")


# Parameters
place_name = "Champaign, Illinois, USA"
initial_deliveries = 100
num_simulations = 10
increment = 50
time_limit = 9  # in hours

# Run the simulation
if __name__ == "__main__":
    truck_a_percentages, truck_b_percentages = run_monte_carlo_simulation(
        place_name, initial_deliveries, num_simulations, increment, time_limit
    )
    perform_statistical_test(truck_a_percentages, truck_b_percentages)
