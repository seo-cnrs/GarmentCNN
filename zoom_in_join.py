# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:46:10 2024

@author: Boyang
"""

import networkx as nx
import matplotlib.pyplot as plt
import trimesh
from collections import OrderedDict

def get_obj(file):
    mesh=trimesh.load(file, process=False)

    faces=mesh.faces
    verts=mesh.vertices
    
    return verts, faces

def get_gemm_edges(faces):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    edge_nb = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                nb_count.append(0)
                edges_count += 1
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
    return edge_nb, edges

def plot_submesh_topo(verts, segments, edges, central_node=61, radius=2):
    # Create a graph
    G = nx.Graph()
    
    # Add nodes with labels
    nodes = dict(zip(list(range(len(verts))), segments))
    #nodes = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
    G.add_nodes_from(nodes)

    # Add edges
    G.add_edges_from(edges)

    # Extract subgraph centered around the specific node
    subgraph = nx.ego_graph(G, central_node, radius)

    # Draw the subgraph with both index and label
    pos = nx.spring_layout(subgraph)
    labels = {node: f"{node}\n{nodes[node]}" for node in subgraph.nodes()}  # Combine index and label in the label string
    nx.draw(subgraph, pos, with_labels=True, labels=labels, node_size=700, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)

    # Show the plot
    plt.show()
    
def print_group_edges():
    # Assume you have a mixed list of edges
    mixed_edges = [(0, 1), (1, 2), (2, 0),(2,9),(9,0), (3, 4), (4, 5), (5, 3), (6, 7), (7, 8), (8, 6)]
    
    # Create a graph from the mixed edges
    G = nx.Graph()
    G.add_edges_from(mixed_edges)
    
    # Initialize a list to store connected components
    connected_components = []
    
    # Initialize a dictionary to store information for each group
    group_info_dict = {}
    
    # Traverse the graph and identify connected components
    for i, component in enumerate(nx.connected_components(G)):
        connected_components.append(component)
    
        # Find triangles within the connected component
        triangles = [cycle for cycle in nx.cycle_basis(G.subgraph(component)) if len(cycle) == 3]
    
        # Store information in the dictionary
        group_info_dict[f'Graph {i + 1}'] = {
            'Nodes': list(component),
            'Edges': list(G.subgraph(component).edges()),
            'Number of Edges': G.subgraph(component).number_of_edges(),
            'Triangles': triangles
        }
    
    # Display the result
    print("Connected Components:")
    for key, info in group_info_dict.items():
        print(f"{key}: {info['Nodes']} (Edges: {info['Edges']}, Number of Edges: {info['Number of Edges']}, Triangles: {info['Triangles']})")
        
def get_seg(seg_path):
    # load SEG file
    #seg_path = os.path.join('tee_0ADQ2LDQJA_sim_segmentation.txt')   # Replace with the path to your text file

    # Read the segmentaion file 
    with open(seg_path, 'r') as file:
        lines = file.readlines()
        # Removing newline characters from each line and storing in a list
        vertex_seg_before_decimation = [line.strip() for line in lines]

    order=list(OrderedDict.fromkeys(vertex_seg_before_decimation))
    print(order)
    segments=[order.index(e)-1 for e in vertex_seg_before_decimation] #TODO: stitch made to -1
    return segments

if __name__ == '__main__':
    import os
    name="tee_08P2D24RBB_sim"
    central_node=70
    
    #print_group_edges()
    
    filename=os.path.join("./tee_100/train", name+".obj")
    seg_path=os.path.join("./tee_100", name+"_segmentation.txt")
    
    verts, faces = get_obj(filename)
    segments=get_seg(seg_path)
    
    edge_nb, edges = get_gemm_edges(faces)
    
    plot_submesh_topo(verts, segments, edges, central_node)