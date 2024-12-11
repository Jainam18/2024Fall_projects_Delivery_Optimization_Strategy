import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta


# Initialize the street network
def initialize_graph(place_name):
    print("Initializing the street network for:", place_name)
    G = ox.graph_from_place(place_name, network_type='drive')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Set base speeds for Truck A and Truck B
    for u, v, k, data in G.edges(keys=True, data=True):
        base_speed_b = data.get("speed_kph", 50)  # Default speed for Truck B
        data["speed_truck_b"] = base_speed_b
        data["speed_truck_a"] = max(base_speed_b - 16, 16)  # Assuming that the Truck A is slower by 16 kph

    print("Graph initialization complete!")
    return G

# Apply traffic congestion using a PERT distribution
def apply_traffic_congestion(G, traffic_impact_probability=0.3):
    for u, v, k, data in G.edges(keys=True, data=True):
        if random.random() < traffic_impact_probability:  # Apply congestion to roads
            traffic_factor = beta.rvs(2, 5, loc=0, scale=1)  # Generate traffic factor using PERT distribution

            # Adjust speeds based on traffic factor
            data["speed_truck_b"] *= (1 - traffic_factor)
            data["speed_truck_a"] *= (1 - traffic_factor)

            # Ensure speed is  realistic
            data["speed_truck_b"] = max(data["speed_truck_b"], 5)
            data["speed_truck_a"] = max(data["speed_truck_a"], 5)

        # Calculate travel times for Truck A and Truck B
        distance = data.get("length", 1)
        data["travel_time_truck_a"] = (distance / data["speed_truck_a"])
        data["travel_time_truck_b"] = (distance / data["speed_truck_b"])

    print("Traffic congestion applied!")
    return G

# Set the hub  and extract Strongly Connected Components
def get_fixed_hub_and_scc(G):
    hub_node = list(G.nodes)[0]

    print("Extracting largest strongly connected component (SCC)")
    if not nx.is_strongly_connected(G):
        components = nx.strongly_connected_components(G)
        for component in components:
            if hub_node in component:
                G_scc = G.subgraph(component).copy()
                print(f"SCC with hub node contains {len(G_scc.nodes)} nodes.")
                return G_scc, hub_node
    else:
        print("The entire graph is strongly connected.")
        return G, hub_node

# Precompute shortest paths from the hub
def precompute_shortest_paths(G, hub_node):
    # Shortest paths for Truck A
    shortest_paths_truck_a = nx.single_source_dijkstra_path_length(G, hub_node, weight="travel_time_truck_a")
    # Shortest paths for Truck B
    shortest_paths_truck_b = nx.single_source_dijkstra_path_length(G, hub_node, weight="travel_time_truck_b")
    return shortest_paths_truck_a, shortest_paths_truck_b

# Generate random delivery points
def generate_delivery_points(G, num_deliveries, hub_node, shortest_paths):
    reachable_nodes = [node for node in G.nodes if node in shortest_paths]
    reachable_nodes.remove(hub_node)
    delivery_points = random.sample(reachable_nodes, min(num_deliveries, len(reachable_nodes)))
    print(f"Generated {len(delivery_points)} reachable delivery points.")
    return delivery_points

# Plan delivery routes using precomputed shortest paths
def plan_routes_with_precomputed_paths(G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths):
    print(f"\nDeliveries started for {truck_type}...")
    remaining_points = delivery_points.copy()
    total_time = 0
    trip_count = 0

    while remaining_points:
        trip_points = remaining_points[:truck_capacity]
        remaining_points = remaining_points[truck_capacity:]

        while trip_points:
            try:
                next_node = min(trip_points, key=lambda x: shortest_paths[x])
                travel_time = shortest_paths[next_node] / 3600  # Convert seconds to hours
                total_time += travel_time
                trip_points.remove(next_node)
            except KeyError as e:
                trip_points.remove(e)

        total_time += shortest_paths[hub_node] / 3600
        trip_count += 1

    print(f"{truck_type} - All deliveries completed in {total_time:.2f} hours with {trip_count} trips.")
    return total_time, trip_count

# Run a single simulation for Truck A or Truck B
def run_single_simulation(args):
    G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths = args
    return plan_routes_with_precomputed_paths(G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths)

# Monte Carlo Simulation
def run_monte_carlo_simulation(place_name, initial_deliveries, num_simulations, increment, time_limit):
    print("Starting Monte Carlo Simulation...")
    G = initialize_graph(place_name)
    G = apply_traffic_congestion(G)
    G_scc, hub_node = get_fixed_hub_and_scc(G)

    shortest_paths_truck_a, shortest_paths_truck_b = precompute_shortest_paths(G_scc, hub_node)

    x_orders = []
    truck_a_percentages = []
    truck_b_percentages = []
    truck_a_trips = []
    truck_b_trips = []

    num_deliveries = initial_deliveries

    for simulation in range(1, num_simulations + 1):
        print(f"\nSimulation {simulation}: Running with {num_deliveries} deliveries...")
        delivery_points = generate_delivery_points(G_scc, num_deliveries, hub_node, shortest_paths_truck_a)

        args_a = (G_scc, hub_node, delivery_points, 100, "Truck A (Large)", shortest_paths_truck_a)
        args_b1 = (G_scc, hub_node, delivery_points[:len(delivery_points)//2], 50, "Truck B1 (Small)", shortest_paths_truck_b)
        args_b2 = (G_scc, hub_node, delivery_points[len(delivery_points)//2:], 50, "Truck B2 (Small)", shortest_paths_truck_b)

        with ProcessPoolExecutor(max_workers=3) as executor:
            results = executor.map(run_single_simulation, [args_a, args_b1, args_b2])

        (time_a, trips_a), (time_b1, trips_b1), (time_b2, trips_b2) = results
        time_b = max(time_b1, time_b2)
        trips_b = trips_b1 + trips_b2

        truck_a_percentage = 100 if time_a <= time_limit else (time_limit / time_a) * 100
        truck_b_percentage = 100 if time_b <= time_limit else (time_limit / time_b) * 100

        truck_a_percentages.append(truck_a_percentage)
        truck_b_percentages.append(truck_b_percentage)
        truck_a_trips.append(trips_a)
        truck_b_trips.append(trips_b)
        x_orders.append(num_deliveries)

        print(f"Truck A completed {trips_a} trips.")
        print(f"Truck B completed {trips_b} trips.")

        num_deliveries += increment

    plot_comparison(x_orders, truck_a_percentages, truck_b_percentages)
    return truck_a_percentages, truck_b_percentages, truck_a_trips, truck_b_trips

# Plot comparison graph
def plot_comparison(x_orders, truck_a_percentages, truck_b_percentages):
    plt.plot(x_orders, truck_a_percentages, label="Truck A (Large)", color="blue", marker="o")
    plt.plot(x_orders, truck_b_percentages, label="Truck B (Max of B1 and B2)", color="green", marker="x")
    plt.xlabel("Number of Orders")
    plt.ylabel("Percentage of Orders Completed (%)")
    plt.title("Percentage of Orders Completed Within Time Limit")
    plt.legend()
    plt.grid()
    plt.show()


# Run the simulation
if __name__ == "__main__":
    place_name = "Champaign, Illinois, USA"
    initial_deliveries = 100
    num_simulations = 50
    increment = 50
    time_limit = 9

    truck_a_percentages, truck_b_percentages, truck_a_trips, truck_b_trips = run_monte_carlo_simulation(
        place_name, initial_deliveries, num_simulations, increment, time_limit
    )