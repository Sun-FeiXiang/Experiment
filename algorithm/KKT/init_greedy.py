import time

from diffusion.Networkx_diffusion import spread_run_IC


def greedy(g, k, p=0.01, mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
    S, spread, timelapse, start_time = [], [], [], time.time()
    # Find k nodes with largest marginal gain
    for _ in range(k):
        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in set(range(g.number_of_nodes())) - set(S):
            # Get the spread
            s = spread_run_IC(g, S + [j], p, mc)
            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j
        # Add the selected node to the seed set
        S.append(node)
        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
    return (S, spread, timelapse)