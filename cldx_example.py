# from cldx import CausalLoopDiagram
import networkx as nx
import pylab as plt



# import pydot
# print(pydot.find_graphviz())

G = nx.DiGraph()

# adding just one node:
G.add_node("a")
# a list of nodes:
G.add_nodes_from(["b", "c", "d"])

print("Nodes of graph: ")
print(G.nodes())
print("Edges of graph: ")
print(G.edges())

G.add_edge(1, 2, weight=0.33)
edge = ("d", "e")
G.add_edge(*edge, weight=0.66)
edge = ("a", "b")
G.add_edge(*edge, weight=1.5)

# adding a list of edges:
G.add_edges_from([("a","c"),("c","d"), ("a",1), (1,"d"), ("a",2)])

pos = nx.fruchterman_reingold_layout(G)
nx.draw(G, pos=pos, with_labels=True, with_weights=True)
edge_labels = nx.get_edge_attributes(G, 'weight')
# pos = nx.get_node_attributes(G, 'pos')
print(pos)
print(edge_labels)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


# pos=nx.get_node_attributes(G,'pos')

plt.savefig("simple_path.png") # save as png
plt.show() # display
