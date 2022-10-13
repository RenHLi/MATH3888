import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import scipy as sp
import random as rn
from heapq import nlargest
from line_profiler import LineProfiler

# some basic settings for plotting figures
import matplotlib.pyplot as plt

import community as community_louvain

from itertools import combinations
from collections import Counter

## Test Graph

G1 = nx.fast_gnp_random_graph(400, p= 0.5,seed=5)
nx.draw(G1, with_labels = True)

l_part = nx_comm.louvain_communities(G1, resolution = 1, seed = 1)

def part_graph_2(G_1, partition):
    N = len(partition) # Number of communities 
    labelled_partition={i:part for i,part in enumerate(partition)}
    node_dict={node:i  for i in labelled_partition for node in labelled_partition[i]}
    G = nx.Graph()
    for i in range (N): 
        G.add_node(i)

    nodes=[*G_1.nodes]


    intra_comm_combs=map(lambda x: combinations(x,2),partition)
    intra_comm_2=[*map(list,intra_comm_combs)]
    intra_comm_3=sum(intra_comm_2,[])
    
    intra_comm_edges={*map(frozenset,intra_comm_3)}
    all_edges={*map(frozenset,G1.edges)}

    inter_comm_edges=all_edges-intra_comm_edges
    
    inter_comm_tuple=map(tuple,inter_comm_edges)

    inter_edges=map(lambda x:frozenset((node_dict[x[0]],node_dict[x[1]])),inter_comm_tuple)

    edge_weights=Counter(inter_edges)

    for t in edge_weights:
        a,b=t
        G.add_edge(a, b, weight = edge_weights[t])

    return G



lp = LineProfiler()
lp_wrapper = lp(part_graph_2)
lp_wrapper(G1,l_part)
lp.print_stats()


def part_graph(G_1, partition):
    N = len(partition) # Number of communities 
    labelled_partition={i:part for i,part in enumerate(partition)}
    node_dict={node:i  for i in labelled_partition for node in labelled_partition[i]}
    G = nx.Graph()
    for i in range (N): 
        G.add_node(i)

    nodes=[*G_1.nodes]


    intra_comm_combs=map(lambda x: combinations(x,2),partition)
    intra_comm_2=[*map(list,intra_comm_combs)]
    intra_comm_3=sum(intra_comm_2,[])
    
    intra_comm_edges={*map(frozenset,intra_comm_3)}
    all_edges={*map(frozenset,G1.edges)}

    inter_comm_edges=all_edges-intra_comm_edges
    
    for ed in inter_comm_edges:
        n1,n2=ed
        p1=node_dict[n1]
        p2=node_dict[n2]
        if G.has_edge(p1, p2): 
            G[p1][p2]["weight"] +=1
        else:
            G.add_edge(p1, p2, weight = 1)

    return G


lp = LineProfiler()
lp_wrapper = lp(part_graph)
lp_wrapper(G1,l_part)
lp.print_stats()