import osmnx as osmnx
import networkx as netx
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta


#intializing the graph for a particular place
def initializing_graph(place_name: str, hypothesis: int) -> netx.MultiDiGraph:
    '''
    Intializing the street network graph for a place and then adding speed attribute to edge. For hypothesis 2 after adding speed to edge, the graph is returned but for hypothesis 1 base speeds of truck A and B are set and then the graph is returned.

    Args:
    place_name(str) : string defining the place name for which the graph is required.
    hypothesis(int): integer value representing the hypothesis number

    Returns:
        netx.MultiDiGraph: The initialized street network graph with speed attribute added to the edge for hypothesis 2 and for hypothesis 1 speed of truck A and truck B is also added.
     
    Doctests:
    >>> g = initializing_graph("Piedmont, California, USA", 1)
    Initializing the street network for: Piedmont, California, USA
    Graph initialization is complete!
    >>> isinstance(g, netx.MultiDiGraph)
    True

    >>> initializing_graph("Piedmont, California, USA", 3)
    Initializing the street network for: Piedmont, California, USA
    Invalid hypothesis number. Please choose either 1 or 2.


    '''

    print("Initializing the street network for:", place_name)
    Graph = osmnx.graph_from_place(place_name, network_type='drive')
    Graph = osmnx.add_edge_speeds(Graph)
    #Graph = osmnx.add_edge_travel_times(Graph)
   
    #For every edge in the graph u: The source node, v: The target node, k: The key differentiating multiple edges between u and v, data: A dictionary containing attributes of the edge, such as length, speed limit, etc.
    if hypothesis==1 or hypothesis==2:
      for u, v, k, data in Graph.edges(keys=True, data=True):
        if(hypothesis==2):
          base_speed=data.get("speed_kph", 30)#if no speed is found then speed is assumed as 30kph.
          data["speed_kph"] = base_speed
        
        elif(hypothesis==1):
           # Setting base speeds for Truck A and Truck B
          base_speed_b = data.get("speed_kph", 50)  # if no speed is found from the network graph then default speed for Truck B is 50kph.
          data["speed_truck_b"] = base_speed_b
          data["speed_truck_a"] = max(base_speed_b - 16, 16)  # Assuming that the Truck A is slower than truck B by 16 kph
      print("Graph initialization is complete!")
      return Graph
    else:
      print("Invalid hypothesis number. Please choose either 1 or 2.")
      return None




#Using PERT distribution to calculte the traffic congestion factor
def apply_traffic_congestion(Graph:netx.MultiDiGraph,hypothesis: int, traffic_impact_probability:float=0.3)-> netx.MultiDiGraph: 

    '''
    Calculating the traffic congestion factor and applying it to the graph. For hypothesis 1, applying congestion factor to both truck A and truck B speed. For hypothesis 2 directly applying congestion factor to speed.

    Args:
    Graph(netx.MultiDiGraph) : The network graph of the location
    hypotheis (int): the hypotheis number 
    traffic_impact_probability (float): The probability of applying traffic congestion to roads.

    Returns:
    netx.MultiDiGraph: The graph with traffic congestion applied to the speed.

    Doctests-
    
        >>> G = netx.MultiDiGraph()
        >>> G.add_edge(1, 2, key=0, length=100, speed_kph=60, speed_truck_b=50, speed_truck_a=40)

        >>> random.seed(42)  # Set seed for predictable output
        >>> beta.random_state = 42
        >>> updated_G = apply_traffic_congestion(G, hypothesis=2, traffic_impact_probability=1.0)
        >>> round(updated_G[1][2][0]['speed_kph'], 2)
        47.63

    '''
  
    #For every edge in the graph u: The source node, v: The target node, k: The key differentiating multiple edges between u and v, data: A dictionary containing attributes of the edge, such as length, speed limit, etc.
    for u, v, k, data in Graph.edges(keys=True, data=True):
        if random.random() < traffic_impact_probability:  # Applying congestion to roads, random.random() generates number between 0.0(inclusive) and 1.0(exclusive)
           '''
           PERT distribution:
           α: The distribution is skewed to the right, with more probability mass concentrated towards higher values.
           β: The distribution is skewed to the left, with more probability mass concentrated towards lower values.
           loc (location parameter): 0. This shifts the distribution to start at 0.
           scale (scale parameter): 1. This scales the distribution to fit within the interval [0, 1].
           The distribution will be slightly skewed to the left and will be more close to 0 than 1.
            '''
           traffic_congestion_factor = beta.rvs(2, 5, loc=0, scale=1)  # Generate traffic factor using PERT distribution.

           if hypothesis == 2:
             data["speed_kph"] = data["speed_kph"] * (1 - traffic_congestion_factor)

           elif hypothesis == 1:
             # Adjust speeds based on traffic factor
             data["speed_truck_b"] =data["speed_truck_b"]* (1 - traffic_congestion_factor)
             data["speed_truck_a"] =data["speed_truck_a"]* (1 - traffic_congestion_factor)

             # Ensure speed is  realistic
             data["speed_truck_b"] = max(data["speed_truck_b"], 5)
             data["speed_truck_a"] = max(data["speed_truck_a"], 5)

             # Calculate travel times for Truck A and Truck B
             distance = data.get("length", 1)
             data["travel_time_truck_a"] = (distance / data["speed_truck_a"])
             data["travel_time_truck_b"] = (distance / data["speed_truck_b"])
           else :
            print("Invalid hypothesis number. Please choose either 1 or 2.")
            return None

    print("Traffic congestion applied!")
    return Graph


# Set the hub  and extract Strongly Connected graph.
def get_fixed_hub_and_scc(Graph:netx.MultiDiGraph)-> tuple[netx.MultiDiGraph,int]:
    '''
     Extracting a strongly connected component- in a directed graph, where any two nodes have a directed path in between them. We didn't want to have any unreahable nodes in the graph.

     Args:
     Graph(netx.MultiDiGraph) : network graph of location

     returns:
     tuple(netx.MultiDiGraph,int):
      netx.MultiDiGraph: Strongly connected graph
      int: hub node
    '''
    hub_node = list(Graph.nodes)[0]

    print("Extracting largest strongly connected graph")
    if not netx.is_strongly_connected(Graph):
        components = netx.strongly_connected_components(Graph)
        for component in components:
            if hub_node in component:
                Graph_scc = Graph.subgraph(component).copy()
                print("Strongly connected component with hub node contains {len(Graph_scc.nodes)} nodes.")
                return Graph_scc, hub_node
    else:
        print("The entire graph is strongly connected.")
        return Graph, hub_node



def precompute_shortest_paths(Graph:netx.MultiDiGraph, hub_node:int,hypothesis:int):
    '''
    Finding out the nearest node from hub

    Args:
    Graph(netx.MultiDiGraph):Graph of location
    hub_node(int): the hub/warehouse from where the trucks will pick up parcels

    return:
    dictionary: shortest path from hub to all nodes in the graph

    Doctests-
    >>> G = netx.MultiDiGraph()
    >>> G.add_edges_from([(1, 2, {'length': 5}), \
                              (2, 3, {'length': 10})])
        
    >>> precompute_shortest_paths(G, 1, hypothesis=2)
        {1: 0, 2: 5, 3: 15}

    '''
    if hypothesis==2:
       shortest_paths = netx.single_source_dijkstra_path_length(Graph, hub_node, weight="length")
       return shortest_paths
    elif hypothesis==1:
       # Shortest paths for Truck A
       shortest_paths_truck_a = netx.single_source_dijkstra_path_length(Graph, hub_node, weight="travel_time_truck_a")
       # Shortest paths for Truck B
       shortest_paths_truck_b = netx.single_source_dijkstra_path_length(Graph, hub_node, weight="travel_time_truck_b")
       return shortest_paths_truck_a, shortest_paths_truck_b
    else:
         print("Invalid hypothesis number. Please choose either 1 or 2.")
         return None
       


def generate_delivery_points(Graph, num_deliveries, hub_node, shortest_paths):

    '''
    Randomly generating delivery points.
    Args:
    Graph(netx.MultiDiGraph):network Graph of location
    num_deliveries(int): number of delivery points to be generated
    hub_node(int): the hub/warehouse from where the trucks will pick up parcels

    return:
    list: list of delivery points

    doctests-
    >>> G = netx.MultiDiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    >>> shortest_paths = {1: 0, 2: 5, 3: 10, 4: 15, 5: 20}
    >>> hub_node = 1

    >>> generate_delivery_points(G, 2, hub_node, shortest_paths)
        Generated 2 reachable delivery points.
        [5, 2]
    '''
    reachable_nodes = [node for node in Graph.nodes if node in shortest_paths]
    reachable_nodes.remove(hub_node)
    delivery_points = random.sample(reachable_nodes, min(num_deliveries, len(reachable_nodes)))
    print("Generated", len(delivery_points)," reachable delivery points.")
    return delivery_points