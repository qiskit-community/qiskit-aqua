import networkx as nx
import string
def draw_graph_ndarray(w):
    G = nx.from_numpy_matrix(w)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))
    G = nx.drawing.nx_agraph.to_agraph(G)
    G.node_attr.update(color="red", style="filled")
    G.edge_attr.update(color="blue", width="2.0")
    G.draw('/tmp/out.png', format='png', prog='neato')
