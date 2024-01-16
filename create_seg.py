# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:25:41 2024

@author: Boyang YU 
"""

import trimesh
import os
#from collections import OrderedDict
import numpy as np
import networkx as nx

import sys
import glob


def get_obj(file):
    #mesh=trimesh.load("tee_0ADQ2LDQJA_sim_3D.obj",process=False)
    mesh=trimesh.load(file, process=False)

    faces=mesh.faces
    verts=mesh.vertices
    
    return verts, faces


def load_labels(path, basename, simplified):
    # original per vertex label or simplified one
    
    if not simplified:
        # load SEG file
        seg_path = os.path.join(path, basename + '_segmentation.txt')
        
        # Read the segmentaion file 
        with open(seg_path, 'r') as file:
            lines = file.readlines()
            # Removing newline characters from each line and storing in a list
            vertex_seg_before_decimation = [line.strip() for line in lines]
            
        #order=list(OrderedDict.fromkeys(vertex_seg_before_decimation)) # TODO: set a fixed one
        #print(order)
        segments=[order.index(e)-1 for e in vertex_seg_before_decimation] #stitch made to -1

    else: # otherwise the simplified vseg already inside the simplified folder
        vseg_path = os.path.join(os.path.join(path, 'vseg'), basename + '.vseg')
        segments = np.loadtxt(vseg_path, delimiter=',', skiprows=0, dtype=int)
        
    return segments


def get_gemm_edges(faces, export_name_edges):
    """
    gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
    sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
    for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
    """
    edges = []
    edge2key = {}
    edges_count=0
    nb_count=[]

    edge_nb=[]


    for face in faces: 
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)

        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            # add a new edge
            if edge not in edge2key:
                edge2key[edge] = edges_count # this should be id
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                nb_count.append(0)
                edges_count+=1
                
        for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                # register the index of 2 adjacent edges in anticlc 
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]] 
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2 # next time edge in another face, then it will fill 2 3
            
    gemm_edges = np.array(edge_nb, dtype=np.int64)
    #assert(len(mesh.edges_unique)== len(edges))
    
    np.savetxt(export_name_edges, edges, delimiter=',',fmt='%s')
    return gemm_edges, edges, edge2key
    

def get_loop_verts_labels(idx,edges,gemm_edges,segments):
        abcd=gemm_edges[idx]
        verts=[]
        for id_e in abcd: # four edge index
            verts+=edges[id_e] 
            
        loop_labels=[segments[v] for v in set(verts)]


        labels=set(loop_labels) # labels involved for this ring

        if -1 in labels:
            labels.remove(-1)
        return labels



def get_adjedge_label(vert_idx,solved_seams, edge2key, edgeLabels):
    for edge in solved_seams:
        if vert_idx in edge:
            idx=edge2key[tuple(edge)]
            label=edgeLabels[idx]
            return label
        

def group_edges(mixed_edges):

    # Create a graph from the mixed edges
    G = nx.Graph()
    G.add_edges_from(mixed_edges)

    # Initialize a list to store connected components
    connected_components = []

    # Initialize a dictionary to store the number of edges and triangles for each group
    group_info = []

    # Traverse the graph and identify connected components
    for i, component in enumerate(nx.connected_components(G)):
        connected_components.append(component)

        # Find triangles within the connected component
        triangles = [cycle for cycle in nx.cycle_basis(G.subgraph(component)) if len(cycle) == 3]
        
        #'Nodes', 'Edges','Number of Edges','Triangles'
        group_info.append((list(component),list(G.subgraph(component).edges()),G.subgraph(component).number_of_edges(),triangles))

    return group_info

def angle_at_vertex(vertices, vertex_index):
    A, B, C = vertices[vertex_index], vertices[(vertex_index + 1) % 3], vertices[(vertex_index + 2) % 3]
    AB, AC = B - A, C - A
    return np.degrees(np.arccos(np.clip(np.dot(AB, AC) / (np.linalg.norm(AB) * np.linalg.norm(AC)), -1.0, 1.0)))

def find_vertex_with_largest_angle(vertices):
    angles = [angle_at_vertex(vertices, i) for i in range(3)]
    return np.argmax(angles)



def create_eseg_file(verts, edges, segments, gemm_edges, edge2key, export_name_eseg):
    # naive seg labels
    edgeLabels = np.ones(len(edges), dtype=int)*999
    solved_seams=[]
    tosolve_seams=[]
    for idx, edge in enumerate(edges):
        
        l0=int(segments[edge[0]])
        l1=int(segments[edge[1]])
        
        if l0!=l1: # take none -1 label
            if l0 == -1:
                edgeLabels[idx]= l1
            else:
                edgeLabels[idx]= l0
                
        elif l0==l1!=-1:
            edgeLabels[idx]= l0 # take any one of it
        else:
            # here are both are -1, so is the stitch
            labels=get_loop_verts_labels(idx,edges,gemm_edges,segments)
            
            if len(labels)==2: # plain stitch
                edgeLabels[idx]= min(labels) #TODO: decide the rule then
                solved_seams.append(edge)
            else: # only one none -1 label left or even empty
                tosolve_seams.append(edge)
                
    #print("tosolve_seams", tosolve_seams)            
    if tosolve_seams:
        tosolve_verts=np.array(tosolve_seams).reshape(-1)
        unique_elements, counts = np.unique(tosolve_verts, return_counts=True)
        
        around_join = group_edges(tosolve_seams)
        join=[]
        
        for verts_involved, edges_involved, nb_e, triangles in around_join:
            # TODO, add a link to an image of all the cases
            if nb_e==5 or nb_e==3:
                triangle=triangles[0]
                #print(triangle)
                triangle_vertices= verts[np.array(triangle)]
                vertex_with_largest_angle = find_vertex_with_largest_angle(triangle_vertices)
                join.append(triangle[vertex_with_largest_angle])
            elif nb_e==8 or nb_e==7 or nb_e==6:
                unique_elements, counts = np.unique(np.array(edges_involved).reshape(-1), return_counts=True)
                join.append(unique_elements[0])
            else:
                print("unexpected case")
            
        #print("join points",join)
        
        for edge in tosolve_seams:
            idx=edge2key[tuple(edge)] 
            # case 1, edge has one vertext is the join point
            if edge[0] in join:
                 edgeLabels[idx]= get_adjedge_label(edge[1],solved_seams, edge2key, edgeLabels)        
            elif edge[1] in join: 
                 edgeLabels[idx]=get_adjedge_label(edge[0],solved_seams, edge2key, edgeLabels)
            # case 2, no join
            else:
                labels=get_loop_verts_labels(idx,edges,gemm_edges,segments)    
                assert(len(labels)==1)
                edgeLabels[idx]=list(labels)[0]    
    
    np.savetxt(export_name_eseg, edgeLabels,  fmt='%s')

def create_files(path, simplified=False):
    
    # preocess the original Maria's dataset
    if not simplified:
        os.makedirs(os.path.join(path,"edges"))
        os.makedirs(os.path.join(path,"seg"))
        os.makedirs(os.path.join(path,"sseg"))
    
    #print(path)
    for filename in glob.glob(os.path.join(path, 'train/*.obj')):
        #print("name: ",filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        print(basename)
        
        labels=load_labels(path, basename, simplified) # read vseg or txt
        
        export_name_eseg = os.path.join(os.path.join(path, 'seg'), basename + '.eseg') 
        export_name_edges = os.path.join(os.path.join(path, 'edges'), basename + '.edges')
        #print(export_name_eseg, export_name_edges)
        
        verts, faces = get_obj(filename)
        
        assert(len(labels)==len(verts))
        gemms, edges, edge2key = get_gemm_edges(faces, export_name_edges)
        create_eseg_file(verts, edges, labels, gemms, edge2key, export_name_eseg)


            
if __name__ == '__main__':
    #create_files(sys.argv[1])
    
    #path="./tee_samples"
    #create_files(path) 
    
    garment_type="tee"
    if garment_type=="tee":  # TODO: should we unify all the patterns
        order=['stitch', 'lbsleeve', 'lfsleeve', 'rbsleeve', 'rfsleeve', 'back', 'front']
    
    create_files(sys.argv[1], sys.argv[1].split("_")[-1]=="simplified")
    print("done")