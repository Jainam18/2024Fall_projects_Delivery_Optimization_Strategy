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
    """Generates a n number of delivery points from a graph G of a city where deliveries are to be made.

    Args:
        G (networkx.Graph): A graph of the city.
        no_of_orders (int): Number of delivery points to generate.
        hub_node_id: Id of the node containing location of delivery hub
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

def route_optimization(hub,delivery_location):
    """Given the hub and the delivery locations which is a list compute the route with the vehicle every time moving to the shortest distance location"""
    pass

if __name__=='__main__':
    location = input("")
    G = initialize_graph(location)
    fig, ax = ox.plot_graph(G)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    hub_node_id = find_hub_node(G)
    hub_node = gdf_nodes[hub_node_id]
    no_of_orders = 800
    delivery_location_nodes_id = generate_delivery_points(G,no_of_orders,hub_node_id)
    print(delivery_location_nodes_id)
    delivery_location_nodes = gdf_nodes[delivery_location_nodes_id]
    print(delivery_location_nodes)