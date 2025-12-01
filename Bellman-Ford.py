#Kai Matton Code
import time

def parse_dimacs(filepath):
    """Parse DIMACS graph format file"""
    edges = []
    adj = {}
    num_nodes = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'p' and len(parts) >= 4:
                num_nodes = int(parts[2])
            elif parts[0] == 'a':
                u = int(parts[1])
                v = int(parts[2])
                w = float(parts[3])
                edges.append((u, v, w))
                # Build adjacency list for path cost calculation
                adj.setdefault(u, []).append((v, w))
                num_nodes = max(num_nodes, u, v)
    return num_nodes, edges, adj

def bellman_ford(start, end, edges, num_nodes):

    INF = float('inf')
    dist = [INF] * (num_nodes + 1)
    pred = [None] * (num_nodes + 1)
    dist[start] = 0

    start_time = time.perf_counter()

    # Track when end node distance last changed
    previous_end_dist = INF

    # Relax edges up to (num_nodes - 1) times
    for iteration in range(num_nodes - 1):
        changed = False

        # Relax all edges
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                changed = True

        # Check if we've found the shortest path to end node
        if dist[end] != INF and dist[end] == previous_end_dist:
            # End node distance hasn't changed - we've found shortest path
            break

        previous_end_dist = dist[end]

        # If no edges were relaxed, algorithm is done
        if not changed:
            break

    elapsed = time.perf_counter() - start_time

    return dist, pred, elapsed


def reconstruct_path(pred, start, end):
    """Reconstruct path from start to end using predecessor array"""
    if start == end:
        return [start]

    path = []
    cur = end

    # Trace back from end to start
    while cur is not None:
        path.append(cur)
        if cur == start:
            break
        cur = pred[cur]

    path.reverse()

    # Verify path is valid
    return path if path and path[0] == start else []


if __name__ == '__main__':
    filepath = 'USA-road-d.NY.gr'

    print("Parsing graph file...")
    num_nodes, edges, adj = parse_dimacs(filepath)
    print(f"Graph loaded: {num_nodes} nodes, {len(edges)} edges")

    start = 1
    end = 200

    # Run Bellman-Ford algorithm
    print(f"\n{'='*60}")
    print("BELLMAN-FORD ALGORITHM")
    print(f"{'='*60}")
    print(f"Finding shortest path from node {start} to node {end}...")
    print("(This may take a while with Bellman-Ford...)")

    dist_bf, pred_bf, runtime_bf = bellman_ford(start, end, edges, num_nodes)

    print(f"\nTime to find shortest path: {runtime_bf:.6f} seconds")

    if dist_bf[end] < float('inf'):
        path_bf = reconstruct_path(pred_bf, start, end)

        # Compute total path cost explicitly
        total_cost = 0
        for i in range(len(path_bf) - 1):
            u = path_bf[i]
            v = path_bf[i+1]
            # Find edge weight from adjacency list
            for neighbor, w in adj.get(u, []):
                if neighbor == v:
                    total_cost += w
                    break

        print(f"Shortest distance (from dist array): {dist_bf[end]}")
        print(f"Total path cost (sum of edge weights along path): {total_cost}")
        print(f"Path length (number of nodes): {len(path_bf)}")
        print(f"Path: {path_bf[:10]}{'...' if len(path_bf) > 10 else ''}")
    else:
        print("Target node is unreachable from start node")

    # Comparison / summary
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Bellman-Ford time:  {runtime_bf:.6f} seconds")
