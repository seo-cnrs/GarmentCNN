

'''
Monday January 15th 2024
Arash Habibi
main.py

This script supposes that the blender file is in a project folder and that
this project folder has at least 4 other subfolders :
- scenes (containing blender files)
- scripts folder which contains all the other scripts (Topology.py, VertexClusters.py, QEM.py, PolyReduce.py)
- obj folder which contains all input files (object files, material files, and segmentation text files)
- obj_export which is where the results should be put.
Then the object file name should be added to the list defined on line 333 of this file and Voilà !
'''

import os
import sys
import numpy as np
import bpy

scene_path = os.getcwd()
project_path='/'
words=scene_path.split('/')
for i in range(1,len(words)-1):
    project_path += words[i]+'/'
sys.path.append(project_path+"scripts")

from PolyReduce import *
from Topology import *
#from PerVertexAttributeDisplay import *


#------------------------------------------------------------=

def flattenList(lst):
    '''
    lst is a list possibly containing sublists, subsublists etc.
    return value : a list without sublists
    The returned list's elements are the union of the elements of the sublists of lst.
    '''
    res = []
    for l in lst:
        if hasattr(type(l),'__len__'):   # if l is a sequence
            res += flattenList(l)
        else:
            res += [l]
    return res

#------------------------------------------------------------=

def deleteAllObjects():
    bpy.ops.object.mode_set(mode="OBJECT")
    objects = bpy.data.objects
    for i in range(len(objects)-1,-1,-1):
        if objects[i].type == "MESH":
            bpy.data.objects.remove(objects[i], do_unlink=True)

#------------------------------------------------------------=

def unselectAllComponents(poly_mesh):
    '''
    poly_mesh is a polygon mesh
    return value : None
    This function unselects all polygons, edges and vertices
    '''
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode="OBJECT")
    '''
    bpy.ops.object.mode_set(mode="OBJECT")

    nb_faces = len(poly_mesh.polygons)
    for i in range(nb_faces):
        poly_mesh.polygons[i].select = False

    nb_edges = len(poly_mesh.edges)
    for i in range(nb_edges):
        poly_mesh.edges[i].select = False

    nb_vertices = len(poly_mesh.vertices)
    for i in range(nb_vertices):
        poly_mesh.vertices[i].select = False

    bpy.ops.object.mode_set(mode="EDIT")
    '''

#------------------------------------------------------------=

def selectComponents(poly_mesh, comps, typ="VERT"):
    '''
    poly_mesh is a polygon mesh
    comps : a list of integers (indices of vertices, edges or faces)
    typ : either "VERTEX", "EDGE" or "FACE".
    return value : None
    This function unselects all polygons, edges and vertices,
    then selects the components of type type specified in comps.
    '''
    assert typ=="VERT" or typ=="EDGE" or typ=="FACE"
    if typ=="VERT":
        nmax = len(poly_mesh.vertices)
        struct = poly_mesh.vertices
    elif typ=="EDGE":
        nmax = len(poly_mesh.edges)
        struct = poly_mesh.edges
    elif typ=="FACE":
        nmax = len(poly_mesh.polygons)
        struct = poly_mesh.polygons

    bpy.ops.object.mode_set(mode="OBJECT")
    unselectAllComponents(poly_mesh)
    for i in comps:
        struct[i].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type=typ)

#------------------------------------------------------------=

def faceMaterialSegmentation(poly_mesh):
    '''
    poly_mesh : a polygon mesh
    return value : a list of integer lists
    The returned list represents nseg integer lists. Each of these
    lists corresponds to a set of face indices which share the same
    material.
    '''
    face_segments = []
    all_faces = list(range(len(poly_mesh.polygons)))
    remaining_faces = list(all_faces)
    iter=0
    while len(remaining_faces)>0 and iter<10:
        unselectAllComponents(poly_mesh)
        bpy.ops.object.mode_set(mode="OBJECT")
        ind_poly = remaining_faces[0]
        poly_mesh.polygons[ind_poly].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.mesh.select_similar(type='FACE_MATERIAL', compare='EQUAL', threshold=0.0)
        bpy.ops.object.mode_set(mode="OBJECT")
        faces=[]
        for nf in range(len(poly_mesh.polygons)):
            if poly_mesh.polygons[nf].select:
                faces += [nf]
        face_segments += [faces]
        ff = flattenList(face_segments)
        remaining_faces = [nf for nf in all_faces if nf not in ff]
        iter += 1
    assert len(remaining_faces)==0
    return face_segments

#------------------------------------------------------------

def segmentIndex(segments, max_value):
    '''
    segments : a list of list of integers. Each list of integers represents a segment
    and contains the elements which belong to this segment.
    max_value is the highest integer value in these lists.
    return value : A list of max_value integers
    Each integer in the returned list represents the index of the
    segment to which it belongs.
    '''
    indices=[None]*(max_value+1)  # +1 because of 0
    for segm_index in range(len(segments)):
        seg_content = segments[segm_index]
        for ind in seg_content:
            indices[ind]=segm_index
    assert None not in indices
    return indices

#------------------------------------------------------------

def vertexSegmentsFromFaceSegments(poly_mesh, face_segments, topo):
    '''
    poly_mesh : a polygon mesh with nf faces and nv vertices
    faces : a list of lists of integers between 0 and nf-1 (inclusive)
    return value : A list of lists of integers between 0 and nv-1
    Each of the lists of face_segments represents a segment
    In the returned list, each vertex whose adjacents faces all
    belong to the same segment are labelled with the same
    segment index. All vertices whose adjacent faces do not belong
    to the same segment are lablled as stitch vertices.
    The stitch segment list is the last element of the
    returned segment list.
    '''
    face_segment_index = segmentIndex(face_segments, len(poly_mesh.polygons)-1)
    nb_face_segments=len(face_segments)
    vertex_segments=[]
    for i in range(nb_face_segments+1):
        vertex_segments += [[]]
    stitch_index=nb_face_segments
    for nv in range(len(poly_mesh.vertices)):
        neighboring_faces=topo._vertex_polygons[nv]
        face_seg_index=-1
        same_segment=True
        for i in range(len(neighboring_faces)):
            if i==0:  face_seg_index=face_segment_index[neighboring_faces[0]]
            elif face_segment_index[neighboring_faces[i]] != face_seg_index:
                same_segment=False
        assert face_seg_index!=-1
        if same_segment:
            vertex_segments[face_seg_index] += [nv]
        else: # stitch
            vertex_segments[stitch_index] += [nv]
    return vertex_segments


#---------------------------------------------------------

def selectSegmentVertices(poly_mesh, segment_name, segments):
    '''
    poly_mesh : a polygon mesh with n vertices
    segments : a list of n strings. One for each vertex, indicating the name of the
        segment to which belongs the corresponding vertex.
    segment_name : the name of one of these segments
    return value : None
    This function unselects all vertices and then selects all vertices whose
    segment name is segment_name.
    '''
    unselectAllComponents(poly_mesh)
    nb_vertices = len(poly_mesh.vertices)

    bpy.ops.object.mode_set(mode="OBJECT")

    for i in range(nb_vertices):
        if segments[i]==segment_name:
            poly_mesh.vertices[i].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="VERT")

#---------------------------------------------------------

def faceSegments(poly_mesh, segments):
    '''
    poly_mesh : a polygon mesh
    segment_name :
    '''
    all_faces = list(range(len(poly_mesh.polygons)))
    face_segments = [[]]
    remaining_faces = all_faces
    while len(remaining_faces)>0:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        ind_poly = remaining_faces[0]
        poly_mesh.polygons[ind_poly].select = True
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.mesh.select_similar(type='FACE_MATERIAL', compare='EQUAL', threshold=0.0)
    #bpy.ops.mesh.select_similar(type='VERT_NORMAL', compare='EQUAL', threshold=0.0)

#------------------ import object file ---------------------------=

#---------------------------------------------------------

def closestVertexOnMesh(poly_mesh, p):
    '''
    poly_mesh is a polygon_mesh
    p is a point in space
    return value : the index of the vertex of poly_mesh which
        is the closest to p.
    '''
    nb_vertices = len(poly_mesh.vertices)
    dist_min = 10000000
    ind_dist_min = -1
    for nv in range(nb_vertices):
        pp = poly_mesh.vertices[nv].co
        diff = (pp[0]-p[0], pp[1]-p[1], pp[2]-p[2])
        dist2 = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
        if dist2 < dist_min:
            dist_min = dist2
            ind_dist_min = nv
    return ind_dist_min

#---------------------------------------------------------

def segmentNameTranslation(poly_mesh1, segments1, poly_mesh1_neighbors, poly_mesh2, segments2):
    '''
    poly_mesh1 : a polygon mesh rather coarse with n1 vertices
    segments1 : a list of n1 integers or strings representing segment_names
    poly_mesh1_neighbors : a list of n1 lists of integers indicating, for each vertex,
        the list of that vertex's neighbors
    poly_mesh2 : a polygon mesh rather refined with n2 vertices
    segments2 : a list of n2 integers or strings representing segment_names
    return value : a dictionary that associates each segment name of poly_mesh1
        to a segment_name of poly_mesh2.
    '''
    segment1_names = list(set(segments1))
    segment2_names = list(set(segments2))
    lookup_table={}

    for s in segment1_names:
        # s_vertex_indices : the indices of poly_mesh1 whose segment name is s
        s_vertex_indices = [i for i in range(len(segments1)) if segments1[i]==s]

        # now, exclude vertices that are on a border with another segment
        s_middle_vertex_indices=[]
        all_neighbors_are_s=True
        for nv in s_vertex_indices:
            neighbs = poly_mesh1_neighbors[nv]
            for nn in neighbs:
                if segments1[nn]!=s:
                    all_neighbors_are_s=False
            if all_neighbors_are_s:
                s_middle_vertex_indices+=[nv]

        # If by doing this, we don't have any vertices left,
        # we give up and take all the vertices
        if len(s_middle_vertex_indices)<1:
            s_middle_vertex_indices=s_vertex_indices

        # We choose 10 of these vertices (or less if there is less than 10)
        sample_vertex_indices = s_middle_vertex_indices[:10]

        # For each sample, find the closest vertex of polymesh2
        vote=[0]*len(segment2_names)
        for nv in sample_vertex_indices:
            p = poly_mesh1.vertices[nv].co
            ind2 = closestVertexOnMesh(poly_mesh2,p)
            seg2 = segments2[ind2]
            ind_seg = segment2_names.index(seg2)
            vote[ind_seg]+=1
        m = max(vote)
        ind = vote.index(m)
        lookup_table[s]=segment2_names[ind]
    return lookup_table


#========================================================================
#========================================================================
#========================================================================



# model_names = ["Tshirt_80"]
model_names = ["LongSkirt_sim_red80","pants_straight_sides_0BKEIFTY5O_sim_red80","pants_straight_sides_1Q02BIICOB_sim_red80","tee_0JVZ240BF7_sim_red80", "tee_4OFG3KGWU9_sim_red80"]

for model in model_names:
    deleteAllObjects()
    dirpath = os.getcwd()+"/../obj"
    objfile = model+".obj"
    bpy.ops.wm.obj_import(filepath=dirpath+"/"+objfile,use_split_objects=False)
    object = bpy.context.active_object
    poly_mesh = object.data

    bpy.ops.object.mode_set(mode="OBJECT")
    initial_segments = np.loadtxt(dirpath+"/"+model+'_segmentation.txt', delimiter=',', skiprows=0, dtype=str)

   #------------------ duplicate object for later ---------------------

    fullres_copy = object.copy()
    fullres_copy.data = object.data.copy()
    fullres_copy.animation_data_clear()
    fullres_copy.name = object.name + "_fullres"
    fullres_copy.hide_viewport = True
    bpy.context.collection.objects.link(fullres_copy)

   #------------------ identify edges to reduce ------------------------

    selectSegmentVertices(object.data, "stitch", initial_segments)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="EDGE")
    edges=[]
    for i in range(len(poly_mesh.edges)):
        if poly_mesh.edges[i].select:
            edges += [i]

    #------------------ actual reduction -------------------------------

    pr = PolyReduce(poly_mesh, edges_to_reduce=edges)
    pr.reduce(collapse_rate=0.5)

    #------------------ restoring segments -----------------------------

    face_segments = faceMaterialSegmentation(poly_mesh)
    # selectComponents(poly_mesh, face_segments[5], typ="FACE")

    topo = Topology(poly_mesh)
    vertex_segments = vertexSegmentsFromFaceSegments(poly_mesh, face_segments, topo)
    # selectComponents(poly_mesh, vertex_segments[6], typ="VERT")
    # teeshirt : 0 : arriere gauche, 1 : devant gauche, 2 : arriere droit, 3 : devant droit, 4: dos, 5 : devant 6 : stitches

    vertex_segment_index = segmentIndex(vertex_segments, len(poly_mesh.vertices)-1)
    lut = segmentNameTranslation(poly_mesh, vertex_segment_index, topo._vertex_neighbors, fullres_copy.data, initial_segments)
    bpy.ops.object.select_all(action='DESELECT')
    fullres_copy.select_set(True)
    bpy.ops.object.delete()

    #------------------ saving outputs ---------------------------------
    # objfile

    dirpath = os.getcwd()+"/../obj_export"
    objfile = model+".obj"
    object.select_set(True)
    bpy.ops.wm.obj_export(filepath=dirpath+"/"+objfile,export_selected_objects=True)
    # https://blender.stackexchange.com/questions/84934/what-is-the-python-script-to-export-the-selected-meshes-in-obj

    # segmentation file
    output_segment_file = open(dirpath+"/"+model+"_segmentation.txt","w")
    for nv in range(len(poly_mesh.vertices)):
        seg_index = vertex_segment_index[nv]
        seg_name = lut[seg_index]
        output_segment_file.write(seg_name+"\n")
    output_segment_file.close()
