from Utility import *
import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta

# Plan delivery routes using precomputed shortest paths
def plan_routes_with_precomputed_paths(G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths):
    '''
       Calculating the total time and number of trips the truck has to make to the hub to bring new set of delivery orders.

    Args-
        
        G (networkx.Graph): The graph representing the road network.
        hub_node (str): The hub node where the truck starts and returns to reload.
        delivery_points (list): A list of delivery point nodes where deliveries need to be made.
        truck_capacity (int): The maximum number of orders/parcels the truck can carry per trip.
        truck_type (str): A string indicating the type of the truck (e.g., "Truck A" or "Truck B").
        shortest_paths (dict): A dictionary containing precomputed shortest path times from the hub to each delivery point.

    Returns:
        tuple: 
          total_time(float) : Total time taken for deliveries (in hours).
          trip_count (int)- number of trips made by truck to reload the orders.

   
    '''
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
    '''
    Runs a single delivery simulation for a specified truck type.

    Args:
        args (tuple): A tuple containing the necessary arguments for the `plan_routes_with_precomputed_paths` function.

    Returns:
        a function named "plan_routes_with_precomputed_paths" which then returns a tuple: Total time taken and number of trips for the specified truck.
    '''
    G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths = args
    return plan_routes_with_precomputed_paths(G, hub_node, delivery_points, truck_capacity, truck_type, shortest_paths)

# Monte Carlo Simulation
def run_monte_carlo_simulation(place_name, initial_deliveries, num_simulations, increment, time_limit):
    '''
    Runs a Monte Carlo simulation to compare delivery efficiency of two truck types over multiple iterations.

    Args:
        place_name (str): The name of the place to initialize the delivery graph.
        initial_deliveries (int): The initial number of delivery points.
        num_simulations (int): The number of simulation runs to perform.
        increment (int): The number of additional deliveries added after each simulation run.
        time_limit (float): The maximum time limit (in hours) allowed for deliveries.

    Returns:
        tuple: Lists of percentages of orders completed within the time limit and the number of trips 
               for each truck type across the simulations.




    '''
    print("Starting Monte Carlo Simulation...")
    G = initializing_graph(place_name,1)
    G = apply_traffic_congestion(G,1)
    G_scc, hub_node = get_fixed_hub_and_scc(G)

    shortest_paths_truck_a, shortest_paths_truck_b = precompute_shortest_paths(G_scc, hub_node,1)

    x_orders = []
    truck_a_percentages = []
    truck_b_percentages = []
    truck_a_trips = []
    truck_b_trips = []

    num_deliveries = initial_deliveries

    for simulation in range(1, num_simulations + 1):
        print(f"\nSimulation {simulation}: Running with {num_deliveries} deliveries...")
        delivery_points = generate_delivery_points(G_scc, num_deliveries, hub_node, shortest_paths_truck_a)

        #we have taken size as 100 for lager truck and smaller trucks have 50 capacity each

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
    '''

    Plotting the graph for percentage of orders complted vs Number of orders.

    Args-
    x_orders(int): total number of orders 
    truck_a_percentages(float): Percentage of orders completed by truck A.
    truck_b_percentages(float):percentages of orders complted by truck type B.
    '''
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
    time_limit = 9 #in hours

    truck_a_percentages, truck_b_percentages, truck_a_trips, truck_b_trips = run_monte_carlo_simulation(
        place_name, initial_deliveries, num_simulations, increment, time_limit
    )