import networkx as nx
import osmnx as ox
import random
import matplotlib.pyplot as plt

def initialize_graph(city_name):
    """
    This function when given a city name in format 'City, State, Country',
    will return a city graph with all the location coordinates as nodes 
    so in return we get node points and edges as two datasets.
    
    Args:
        city_name (str): Name of the city in the format 'City, State, Country'.

    Returns:
        G (networkx.Graph): A graph representing the city's nodes and edges.
    
    >>> G = initialize_graph("Champaign, IL, USA")
    >>> len(G.nodes) > 0
    True
    >>> len(G.edges) > 0
    True
    >>> "location" in G.nodes[0]
    True
    """
    G = ox.graph_from_place(city_name, network_type="drive")
    return G

def find_hub_node(G):
    """
    Find the central hub node which will be the delivery hub 
    decided according to the edge centrality measure.

    Args:
        G (networkx.Graph): A graph of the city.

    Returns:
        hub (int): Node ID of the hub with the highest edge centrality.
    """
    edge_centrality = nx.betweenness_centrality(nx.Graph(G))
    hub_node = max(edge_centrality, key=edge_centrality.get)
    return hub_node

def generate_delivery_points(G, no_of_orders, hub_node_id):
    """
    Generates a number of delivery points from a graph G of a city 
    where deliveries are to be made.

    Args:
        G (networkx.Graph): A graph of the city.
        no_of_orders (int): Number of delivery points to generate.
        hub_node_id: ID of the node containing location of delivery hub.

    Returns:
        delivery_points (list): List of node IDs selected as delivery points.

    >>> import networkx as nx
    >>> G = nx.complete_graph(10) 
    >>> delivery_points = generate_delivery_points(G, 3)
    >>> len(delivery_points)
    3
    >>> all(dp in G.nodes for dp in delivery_points)
    True
    """
    all_nodes = list(G.nodes)
    delivery_nodes = random.sample(all_nodes, no_of_orders)
    if hub_node_id in delivery_nodes:
        delivery_nodes.remove(hub_node_id)
    return delivery_nodes

def route_greedy_closest_first(G, hub, delivery_points):
    """
    Implements the greedy closest-first delivery approach.
    
    Args:
        G (networkx.Graph): A graph of the city.
        hub (int): Node ID of the delivery hub.
        delivery_points (list): List of delivery point node IDs.

    Returns:
        total_distance (float): Total travel distance for the greedy approach.
    """
    visited = set()
    current_node = hub
    total_distance = 0

    while len(visited) < len(delivery_points):
        closest_node = None
        min_distance = float('inf')
        for node in delivery_points:
            if node not in visited:
                distance = nx.shortest_path_length(G, current_node, node, weight='length')
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node
        total_distance += min_distance
        visited.add(closest_node)
        current_node = closest_node

    # Return to hub
    total_distance += nx.shortest_path_length(G, current_node, hub, weight='length')
    return total_distance, visited

def solve_tsp_on_osmnx_graph(G, hub, delivery_points, weight='length'):
    """
    Solve Traveling Salesman Problem on a subset of nodes from an OSMnx graph
    
    Parameters:
    - G: NetworkX graph from OSMnx
    - selected_nodes: List of nodes to visit
    - weight: Edge attribute to use as distance (default is 'length')
    
    Returns:
    - Optimal route through the selected nodes
    - Total route distance
    """
    delivery_points.insert(0, hub)
    # Create a complete graph with edge weights based on shortest path lengths
    complete_graph = nx.complete_graph(delivery_points)
    
    # Compute shortest path lengths between all pairs of selected nodes
    for u, v in complete_graph.edges():
        try:
            # Calculate shortest path length
            path_length = nx.shortest_path_length(G, u, v, weight=weight)
            complete_graph[u][v]['weight'] = path_length
        except nx.NetworkXNoPath:
            # If no path exists, set a very high weight
            complete_graph[u][v]['weight'] = float('inf')
    
    # Solve TSP using approximation algorithm
    try:
        tsp_path = nx.approximation.traveling_salesman_problem(complete_graph, cycle=True)
        
        # Compute the actual route using shortest paths
        full_route = []
        route_distance = 0
        
        for i in range(len(tsp_path) - 1):
            start_node = tsp_path[i]
            end_node = tsp_path[i+1]
            
            # Find the shortest path between nodes
            shortest_path = nx.shortest_path(G, start_node, end_node, weight=weight)
            
            # Compute path length
            path_length = nx.path_weight(G, shortest_path, weight=weight)
            route_distance += path_length
            
            # Extend route (avoid duplicates)
            if i == 0:
                full_route.extend(shortest_path)
            else:
                full_route.extend(shortest_path[1:])
        
        return route_distance, full_route
    
    except nx.NetworkXError:
        print("Could not solve TSP. Ensure nodes are connected.")
        return None, None

def run_simulation(city_name, no_of_orders, trials=10):
    """
    Runs a simulation to compare TSP and greedy approaches.

    Args:
        city_name (str): Name of the city in the format 'City, State, Country'.
        no_of_orders (int): Number of delivery points.
        trials (int): Number of simulation trials.

    Returns:
        results (list): List of tuples (greedy_distance, tsp_distance).
    """
    G = initialize_graph(city_name)
    hub = find_hub_node(G)
    results = []

    for _ in range(trials):
        delivery_points = generate_delivery_points(G, no_of_orders, hub)
        greedy_distance, visited = route_greedy_closest_first(G, hub, delivery_points)
        tsp_distance, tsp_route = solve_tsp_on_osmnx_graph(G, hub, delivery_points)
        # print(greedy_distance)
        # print(visited)
        # tsp_distance = (G, hub, delivery_points)
        results.append((greedy_distance, tsp_distance))
        print(visited)
        print(tsp_route)
        print(results)
    
    return results

if __name__ == '__main__':
    city_name = input("Enter city name in format 'City, State, Country': ")
    no_of_orders = int(input("Enter number of delivery orders: "))
    trials = int(input("Enter number of simulation trials: "))

    results = run_simulation(city_name, no_of_orders, trials)
    
    # # Analyze results
    # avg_greedy = sum(r[0] for r in results) / len(results)
    # avg_tsp = sum(r[1] for r in results) / len(results)

    # print(f"Average distance (Greedy): {avg_greedy:.2f}")
    # print(f"Average distance (TSP): {avg_tsp:.2f}")
    # print("Results for individual trials:")
    # for idx, (greedy, tsp) in enumerate(results, 1):
    #     print(f"Trial {idx}: Greedy = {greedy:.2f}, TSP = {tsp:.2f}")

    # # Visualize comparison
    # greedy_distances = [r[0] for r in results]
    # tsp_distances = [r[1] for r in results]

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, trials + 1), greedy_distances, label="Greedy Closest-First", marker="o")
    # plt.plot(range(1, trials + 1), tsp_distances, label="TSP Optimized", marker="x")
    # plt.xlabel("Trial")
    # plt.ylabel("Total Distance")
    # plt.title("Greedy vs TSP: Total Distance Comparison")
    # plt.legend()
    # plt.show()
