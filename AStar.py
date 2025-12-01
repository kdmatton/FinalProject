#AJ Wood Code
import math
import random
import time
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt


# ------------------------------------------------------
# CONFIG: paths to your files (they are in the same folder)
# ------------------------------------------------------
CO_PATH = "USA-road-d.NY.co"
GR_PATH = "USA-road-d.NY.gr"


# ------------------------------------------------------
# Loading the data
# ------------------------------------------------------

def read_coords(path):
    """
    Read coordinates from the .co file.
    Expected formats:
      - 'v id x y'
      - or 'id x y'
    Returns: dict {node_id: (x, y)}
    """
    coords = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ("c", "p", "#"):
                continue
            parts = line.split()

            # Format: v id x y
            if parts[0].lower() == "v" and len(parts) >= 4:
                nid = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                coords[nid] = (x, y)

            # Some formats: id x y
            elif len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[nid] = (x, y)
                except ValueError:
                    continue

    return coords


def read_graph(path):
    """
    Read edges from the .gr file.
    Expected format:
      - 'a u v w'  (arc from u to v with weight w)
    or:
      - 'u v w'
    Returns: (adjacency list dict, edge_count)
    """
    adj = defaultdict(list)
    edge_count = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ("c", "p", "#"):
                continue
            parts = line.split()

            # Format: a u v w
            if parts[0].lower() == "a" and len(parts) >= 4:
                try:
                    u = int(parts[1])
                    v = int(parts[2])
                    w = float(parts[3])
                    adj[u].append((v, w))
                    edge_count += 1
                except ValueError:
                    continue

            # Format: u v w
            elif len(parts) >= 3:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    w = float(parts[2])
                    adj[u].append((v, w))
                    edge_count += 1
                except ValueError:
                    continue

    return adj, edge_count


# ------------------------------------------------------
# A* algorithm
# ------------------------------------------------------

def euclidean_heuristic(a, b):
    """Euclidean distance between points a=(x1,y1), b=(x2,y2)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(adj, coords, start, goal, max_expansions=None):
    """
    Run A* from start to goal on directed graph adj using coords for heuristic.

    adj: dict {u: [(v, weight), ...]}
    coords: dict {node: (x, y)}
    start, goal: node IDs
    max_expansions: optional safety cap (int)

    Returns:
        path (list of node IDs) or None,
        total_cost,
        nodes_expanded,
        queue_pops,
        elapsed_time_seconds
    """
    t0 = time.perf_counter()

    if start == goal:
        return [start], 0.0, 0, 1, time.perf_counter() - t0

    # priority queue of (f, g, node, parent)
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, start, None))

    came_from = {}  # node -> parent
    gscore = {start: 0.0}
    closed = set()

    nodes_expanded = 0
    queue_pops = 0

    def h(n):
        if n in coords and goal in coords:
            return euclidean_heuristic(coords[n], coords[goal])
        return 0.0

    while open_heap:
        f, g, node, parent = heapq.heappop(open_heap)
        queue_pops += 1

        if node in closed:
            continue

        if parent is not None:
            came_from[node] = parent

        if node == goal:
            # reconstruct path
            path = [node]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            elapsed = time.perf_counter() - t0
            return path, g, nodes_expanded, queue_pops, elapsed

        closed.add(node)
        nodes_expanded += 1

        if max_expansions is not None and nodes_expanded > max_expansions:
            elapsed = time.perf_counter() - t0
            return None, float("inf"), nodes_expanded, queue_pops, elapsed

        for (nbr, w) in adj.get(node, []):
            if nbr in closed:
                continue
            tentative_g = g + w
            if tentative_g < gscore.get(nbr, float("inf")):
                gscore[nbr] = tentative_g
                f_nbr = tentative_g + h(nbr)
                heapq.heappush(open_heap, (f_nbr, tentative_g, nbr, node))

    elapsed = time.perf_counter() - t0
    return None, float("inf"), nodes_expanded, queue_pops, elapsed


# ------------------------------------------------------
# Plotting function
# ------------------------------------------------------

def plot_path(coords, path, start, goal, cost, elapsed, expanded, out_file="astar_path.png"):
    """
    Plot a sampled background of nodes and overlay the A* path.
    """
    # sample background nodes so plotting doesn't die
    sample_size = 3000
    items = list(coords.items())
    if len(items) > sample_size:
        sampled = random.sample(items, sample_size)
    else:
        sampled = items

    xs = [xy[0] for _, xy in sampled]
    ys = [xy[1] for _, xy in sampled]

    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, s=1, alpha=0.5)

    path_xy = [coords[n] for n in path if n in coords]
    px = [p[0] for p in path_xy]
    py = [p[1] for p in path_xy]

    plt.plot(px, py, linewidth=2.5, marker='o', markersize=3)

    plt.title(
        f"A* {start} -> {goal} | "
        f"nodes={len(path)} cost={int(cost)} "
        f"time={elapsed:.4f}s expanded={expanded}"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to: {out_file}")


# ------------------------------------------------------
# Main program
# ------------------------------------------------------

def main():
    print("Loading coordinates from", CO_PATH)
    coords = read_coords(CO_PATH)
    print(f"Loaded coordinates for {len(coords)} nodes.")

    print("Loading graph edges from", GR_PATH)
    adj, edge_count = read_graph(GR_PATH)
    print(f"Loaded graph with {edge_count} edges and {len(adj)} nodes with outgoing edges.\n")

    # Show some sample node IDs
    sample_nodes = list(coords.keys())[:10]
    print("Example node IDs (first 10):", sample_nodes)
    print("Use any valid node ID from the graph as start/end.")

    # Get start and goal from user
    def ask_node(prompt):
        while True:
            val = input(prompt).strip()
            try:
                nid = int(val)
                if nid not in coords:
                    print("That node ID does not exist in the coordinates. Try again.")
                    continue
                return nid
            except ValueError:
                print("Please enter a valid integer node ID.")

    start = ask_node("Enter START node ID: ")
    goal = ask_node("Enter GOAL  node ID: ")

    print(f"\nRunning A* from {start} to {goal} ...")
    path, cost, expanded, visited, elapsed = astar(adj, coords, start, goal)

    if path is None:
        print("\nNo path found between those nodes.")
        print(f"Nodes expanded: {expanded}")
        print(f"Queue pops: {visited}")
        print(f"Time: {elapsed:.6f} seconds")
        return

    print("\n=== A* RESULTS ===")
    print(f"Start node: {start}")
    print(f"Goal node:  {goal}")
    print(f"Path length (in nodes): {len(path)}")
    print(f"Total path cost:        {cost}")
    print(f"Nodes expanded:         {expanded}")
    print(f"Queue pops:             {visited}")
    print(f"Time taken:             {elapsed:.6f} seconds")

    print("\nFirst up to 50 nodes on the path:")
    print(path[:50])

    plot_path(coords, path, start, goal, cost, elapsed, expanded)


if __name__ == "__main__":
    main()
