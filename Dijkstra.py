# Slate's code
import heapq
import time
import matplotlib.pyplot as plt
import networkx as nx

# load dimacs graph
def load_dimacs_graph(filepath):
    graph = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("a"):
                _, u, v, w = line.split()
                u = int(u)
                v = int(v)
                w = int(w)

                if u not in graph:
                    graph[u] = []
                graph[u].append((v, w))
    return graph

# load coordinates
def load_coordinates(filepath):
    coords = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v"):
                _, node_id, lon, lat = line.split()
                node_id = int(node_id)
                lon = float(lon) / 1e6
                lat = float(lat) / 1e6
                coords[node_id] = (lon, lat)
    return coords

# dikstra
def dijkstra_with_prev(graph, start, end=None):
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0

    pq = [(0, start)]

    while pq:
        current_dist, node = heapq.heappop(pq)

        if end is not None and node == end:
            break

        if current_dist > dist[node]:
            continue

        for neighbor, weight in graph.get(node, []):
            new_dist = current_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    return dist, prev

def reconstruct_path(prev, start, end):
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]

# visualization
def visualize_path_geo(path, coords):
    G = nx.DiGraph()

    for i in range(len(path) - 1):
        G.add_edge(path[i], path[i+1])

    pos = {node: coords[node] for node in path}

    plt.figure(figsize=(10, 10))

    nx.draw(
        G, pos,
        node_size=10,
        edge_color="blue",
        arrows=False
    )

    nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_color="green", node_size=60)
    nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color="red", node_size=60)

    plt.title("Shortest Path Using Real Coordinates (NY DIMACS)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

gr_file = "USA-road-d.NY.gr"
co_file = "USA-road-d.NY.co"

print("Loading graph...")
graph = load_dimacs_graph(gr_file)
coords = load_coordinates(co_file)
print("Graph nodes:", len(graph))
print("Coordinate entries:", len(coords))

start_node = 1
end_node = 100000

print("Running Dijkstra...")
start_time = time.time()
dist, prev = dijkstra_with_prev(graph, start_node, end_node)
elapsed = time.time() - start_time

distance = dist[end_node]
km = distance / 1000.0

print("Shortest distance:", distance)
print("Distance in km:", km)
print("Runtime:", elapsed)

path = reconstruct_path(prev, start_node, end_node)
print("Path length:", len(path))

# visualize_path_geo(path, coords) # uncomment this to pull up the visualization