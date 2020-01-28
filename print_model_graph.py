import networkx as nx
from typing import List, Dict
import random
import numpy
import pickle
import argparse
import matplotlib.pylab as plt

from definitions import path_join, EXPERIMENTS_DIR


random.seed(12345)
numpy.random.seed(12345)


def build_nx_graph(nodes: List, edges: Dict, epsilon=1e-5):
    graph = nx.DiGraph()

    graph.add_nodes_from(nodes)

    for edge in edges:
        weight = edge[2]
        if abs(weight) > epsilon:
            rounded = round(weight, 4)
            graph.add_edge(edge[0], edge[1], weight=rounded)

    pos = nx.fruchterman_reingold_layout(graph)
    nx.draw(graph, pos=pos, with_labels=True, with_weights=True)

    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)


def read_pickled_edges(experiment_dir):
    edges_file = path_join(experiment_dir, 'edges')
    levels_file = path_join(experiment_dir, 'levels')

    with open(edges_file, 'rb') as f:
        edges = pickle.load(f)

    with open(levels_file, 'rb') as f:
        levels = pickle.load(f)

    return edges, levels


def main(args):
    experiment_dir = path_join(EXPERIMENTS_DIR, args.experiment_name)
    edges, levels = read_pickled_edges(experiment_dir)
    build_nx_graph(levels, edges)

    file_name = 'sd_model.png'
    file_path = path_join(experiment_dir, file_name)

    plt.savefig(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="",
    )

    args = parser.parse_args()
    main(args)
