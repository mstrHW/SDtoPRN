from io import StringIO
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx


graph = nx.MultiGraph()

graph.add_node('x', x=100, y=0)
graph.add_node('h', x=100, y=100)
graph.add_node('y', x=100, y=200)
# graph.add_node(3, x=200, y=200)
graph.add_edge('x', 'y', length='1', connectionstyle='arc3, rad=10')
graph.add_edge('x', 'h', length='W')
graph.add_edge('h', 'y', length='dt')
# graph.add_edge(0, 2, length='W')
# graph.graph['graph'] = {'rankdir': 'TD'}
# graph.graph['node'] = {'shape': 'circle'}
# graph.graph['edges'] = {'arrowsize': '4.0'}
# for n in graph:
#     graph.nodes[n]['pos'] = '"%d,%d"'%(graph.nodes[n]['x'], graph.nodes[n]['y'])
# p = nx.drawing.nx_pydot.to_pydot(graph)

# for n in graph:
#     graph.nodes[n]['pos'] = "{},{}!".format(
#         graph.nodes[n]['x'], graph.nodes[n]['y'])


def get_pos(graph):
    op = {n: [graph.nodes[n]['x'], graph.nodes[n]['y']] for n in graph}
    return op


pos = get_pos(graph)

for n in graph:
    graph.nodes[n]['pos'] = "{},{}".format(graph.nodes[n]['x'], graph.nodes[n]['y'])
#
# for n in graph:
#     graph.nodes[n]['pos'] = (graph.nodes[n]['x'], graph.nodes[n]['y'])

print(len(pos))
print(len(graph))
# pos = nx.spring_layout(graph)
# print(pos)
# nx.drawing.(graph, "test_fix.dot")
# nx.draw(graph, pos, with_labels=True)
# nx.draw(graph, pos, with_labels=True, connectionstyle='arc3, rad=0.1') #node_size=500, font_weight='bold',
# edges = nx.draw_networkx_edge_labels(graph, pos)
# edge_labels = dict([((u, v,), d['length'])for u, v, d in graph.edges(data=True)])
# render pydot by calling dot, no file saved to disk
# treat the dot output string as an image file
# plt.savefig("Graph.png", format="PNG")
# plt.show()
A = nx.drawing.nx_agraph.to_agraph(graph)
A.layout()
# plt.figure(figsize=(16, 9))
A.draw('multi.png', format='jpg')
print(A)