#Kai Matton Code - Fixed Version
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
    """
    Bellman-Ford algorithm for shortest path.
    
    Returns:
        dist: array of shortest distances from start node
        pred: array of predecessors for path reconstruction
        elapsed: runtime in seconds
        has_negative_cycle: boolean indicating if a negative cycle was detected
    """
    INF = float('inf')
    dist = [INF] * (num_nodes + 1)
    pred = [None] * (num_nodes + 1)
    dist[start] = 0

    start_time = time.perf_counter()

    # Relax edges up to (num_nodes - 1) times
    for iteration in range(num_nodes - 1):
        changed = False

        # Relax all edges
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                changed = True

        # If no edges were relaxed, algorithm is done
        # This is the ONLY safe early termination condition
        if not changed:
            print(f"Converged early at iteration {iteration + 1}")
            break

    # Check for negative cycles (n-th iteration)
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    elapsed = time.perf_counter() - start_time

    return dist, pred, elapsed, has_negative_cycle


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


def verify_path_cost(path, adj):
    """
    Verify the total cost of a path by summing edge weights.
    
    Returns:
        total_cost: sum of edge weights along the path
        valid: boolean indicating if all edges in the path exist
    """
    total_cost = 0
    valid = True
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        # Find edge weight from adjacency list
        edge_found = False
        for neighbor, w in adj.get(u, []):
            if neighbor == v:
                total_cost += w
                edge_found = True
                break
        
        if not edge_found:
            valid = False
            print(f"Warning: Edge {u} -> {v} not found in graph!")
            break
    
    return total_cost, valid


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

    dist_bf, pred_bf, runtime_bf, has_neg_cycle = bellman_ford(start, end, edges, num_nodes)

    print(f"\nTime to find shortest path: {runtime_bf:.6f} seconds")

    # Check for negative cycle
    if has_neg_cycle:
        print("\n⚠️  WARNING: Graph contains a negative cycle!")
        print("Shortest path distances are not well-defined.")
    else:
        print("✓ No negative cycles detected")

    # Display results
    if dist_bf[end] < float('inf'):
        path_bf = reconstruct_path(pred_bf, start, end)

        # Verify path cost
        total_cost, path_valid = verify_path_cost(path_bf, adj)

        print(f"\nShortest distance (from dist array): {dist_bf[end]}")
        print(f"Total path cost (sum of edge weights): {total_cost}")
        
        # Sanity check: these should match
        if abs(dist_bf[end] - total_cost) > 1e-6:
            print("⚠️  WARNING: Distance mismatch! Algorithm may have a bug.")
        else:
            print("✓ Distance verification passed")
        
        if not path_valid:
            print("⚠️  WARNING: Path contains invalid edges!")
        
        print(f"Path length (number of nodes): {len(path_bf)}")
        print(f"Path: {path_bf[:10]}{'...' if len(path_bf) > 10 else ''}")
    else:
        print("\n❌ Target node is unreachable from start node")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Algorithm:          Bellman-Ford")
    print(f"Runtime:            {runtime_bf:.6f} seconds")
    print(f"Nodes explored:     {num_nodes}")
    print(f"Total edges:        {len(edges)}")
    if dist_bf[end] < float('inf'):
        print(f"Shortest distance:  {dist_bf[end]}")
        print(f"Path hops:          {len(path_bf) - 1}")
    print(f"{'='*60}")