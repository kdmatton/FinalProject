# Connor's Code
import networkx as nx
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import scipy as sp

# -----------------------------
# DAG generator
# -----------------------------
def fast_random_dag(n, avg_outdegree):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for u in range(n):
        max_targets = min(avg_outdegree, n - u - 1)
        if max_targets <= 0:
            continue
        targets = random.sample(range(u + 1, n), max_targets)
        for v in targets:
            G.add_edge(u, v, weight=np.random.uniform(1, 5))
    return G

# -----------------------------
# Benchmark
# -----------------------------
def benchmark_dag_shortest_path(G, source):
    start = time.time()
    nx.single_source_shortest_path_length(G, source)
    end = time.time()
    return end - start

# -----------------------------
# Layout-based DAG visualization (subgraph)
# -----------------------------
def visualize_dag_layout_subgraph(G, source, target, neighbor_hops=2):
    """
    Visualizes a subgraph around the shortest path with a spring layout.
    neighbor_hops: how many neighbors around the path to include
    """
    try:
        path = nx.shortest_path(G, source=source, target=target, weight="weight")
    except nx.NetworkXNoPath:
        print("No path exists — cannot highlight shortest path.")
        return

    # Include nodes on the path and their neighbors up to neighbor_hops
    sub_nodes = set(path)
    for _ in range(neighbor_hops):
        for u in list(sub_nodes):
            sub_nodes.update(G.predecessors(u))
            sub_nodes.update(G.successors(u))
    subG = G.subgraph(sub_nodes).copy()

    print(f"Visualizing subgraph with {subG.number_of_nodes()} nodes and {subG.number_of_edges()} edges")

    # Compute layout
    pos = nx.spring_layout(subG, k=0.3, iterations=50)

    plt.figure(figsize=(12, 12))
    nx.draw(subG, pos, node_size=50, edge_color="gray", alpha=0.3)

    # Highlight the shortest path
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_nodes(subG, pos, nodelist=path, node_color="red", node_size=150)
    nx.draw_networkx_edges(subG, pos, edgelist=path_edges, edge_color="red", width=2)

    plt.title(f"DAG Subgraph with Shortest Path ({source} → {target})")
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    n = 200
    # n = 100000
    avg_outdegree = 3
    source = 0
    target = n - 1

    print("=== Generating Large DAG ===")
    DAG = fast_random_dag(n, avg_outdegree)
    print(f"Nodes: {DAG.number_of_nodes()}")
    print(f"Edges: {DAG.number_of_edges()}")

    print("\n=== Benchmarking DAG Shortest Path ===")
    t_dag = benchmark_dag_shortest_path(DAG, source)
    print(f"DAG shortest path runtime: {t_dag:.6f} seconds")

    print("\n=== Visualizing Subgraph Around Shortest Path ===")
    visualize_dag_layout_subgraph(DAG, source, target, neighbor_hops=2)