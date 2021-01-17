import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import time
from igraph import *

from algorithm.Spread.igraph_spread import IC


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
        for j in set(range(g.vcount())) - set(S):

            # Get the spread
            s = IC(g, S + [j], p, mc)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        S.append(node)

        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)

    return (S, spread, timelapse)


if __name__ == "__main__":
    # Create simple network with 0 and 1 as the influential nodes
    source = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5]
    target = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9]

    g = Graph(directed=True)
    g.add_vertices(range(10))
    g.add_edges(zip(source, target))
    # nodes = g.vs
    # degrees = g.degree()

    # Plot graph
    # g.vs["label"], g.es["color"], g.vs["color"] = range(10), "#B3CDE3", "#FBB4AE"
    # plot(g, bbox=(200, 200), margin=20, layout=g.layout("kk")).save("a.png")

    greedy_output = greedy(g, 2, p=0.2, mc=1000)
    print("greedy output: " + str(greedy_output[0]))
