'''
Tuesday January 16th 2024
Arash HABIBI
PolyReduce.py

The Quadric Error Metrics (Qem) enables to evaluate, for each edge
of a polygon, the amount of deformation produced if this edge
was collapsed. It also enables to find the best position to put
the produced vertex along the collapsed edge.

Next step : use the QEM error estimation and length value to
sort the edges beginning with the best choices.

In order to avoid component reordering, we avoid to effectively
collapse the edges. We merely superimpose them in vertex clusters.

When the desired number of edges is attained, we loop through the
edges and merge each vertex cluster in one vertex.

In the previous version (PolyReduce1.py, some edges would become
very long swallowing the neighboring small vertices. Even when
coef_qem is 1. Doing so, they tend the whole tissue in a specific
direction. In this version, we add another criterion. We add an
excentricity coefficient to each vertex cluster. This is the sum
of all the adjacent edges unit vectors leading to this vertex.
If this sum has a high magnitude, it means that the vertex is pulling
all these edges towards that direction. We need to avoid high
excentricities. For an edge adjacent to such a vertex :
- if the excentricity goes towards the edge, it is a good reason
not to collapse that edge, otherwise it would accentuate the excentricity.
- if the excentricity turns away from the edge, it is a good
reason to collapse this edge, even if it is long, and to take
a collapse position far from this vertex.
'''

import bpy
import numpy as np
import sys
import os
import math

scene_path = os.getcwd()
project_path='/'
words=scene_path.split('/')
for i in range(1,len(words)-1):
    project_path += words[i]+'/'
sys.path.append(project_path+"scripts")

from Topology import *
from QEM import *
from VertexClusters import *

pr_very_small = 0.00001
pr_very_large = 1e20

#======================================================================

def unselectAllComponents(poly_mesh):
    '''
    poly_mesh is a polygon mesh
    return value : None
    This function unselects all polygons, edges and vertices
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

#======================================================================

class PolyReduce:

    #-----------------

    def __init__(self, poly_mesh, collapse_along_edge=True, edges_to_reduce=[], edges_to_preserve=[], preserve_borders=True):
        '''
        - poly_mesh is a polygon_mesh with Ne edges
        - collapse_along_edge : boolean
        if it is True, when an edge is collapsed the position of the
        produced vertex is sought along the edge. Otherwise, it
        is sought everywhere (not implemented yet)
        - edges_to_reduce and edges_to_preserve are lists of integers
        all between 0 and Ne-1 representing a set of edge indices.
        edges_to_reduce are the set of indices of edges to be reduced.
        edges_to_preserve : the list of edges that should not be collapsed.
        if edges_to_collapse is empty, it means that the whole
        polygon should be simplified except edges_to_preserve.
        '''
        self._object = objectFromMeshName(poly_mesh.name)
        self._poly_mesh = poly_mesh
        self._polygon_qems = [None] * len(poly_mesh.polygons)
        self._border_plane_qems = [None] * len(poly_mesh.edges)
        self._vertex_qems = [None] * len(poly_mesh.vertices)
        self._edge_qems = [None] * len(poly_mesh.edges)
        # If a qem value is None, it means that it has never been touched.
        # If a qem value is 0 it corresponds to a degenerate component
        self._edge_collapse_pos = [None] * len(poly_mesh.edges)
        self._edge_collapse_err = [pr_very_large] * len(poly_mesh.edges)
        self._collapsed_edges = []
        self._initial_lengths = [None] * len(poly_mesh.edges)
        self._threshold_excentricity = 1

        self._topo = Topology(poly_mesh)
        border_edges = self._topo._edges_touching_border
        self.defineInitialLengths()

        if edges_to_reduce==[]:
            self._edges = [i for i in range(len(poly_mesh.edges))
                           if i not in edges_to_preserve and (i not in border_edges or not preserve_borders) ]
        else:
            self._edges = [i for i in edges_to_reduce
                           if i not in edges_to_preserve and (i not in border_edges or not preserve_borders) ]

        # select all edges that must be reduced.
        unselectAllComponents(poly_mesh)
        bpy.ops.object.mode_set(mode="OBJECT")
        for ne in self._edges:
            poly_mesh.edges[ne].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="EDGE")

        for ne in self._edges:
            assert ne in range(len(poly_mesh.edges))
            self._edge_qems[ne] = QEM(poly_mesh, "EDGE", ne, self._topo,
                                      polygon_qems=self._polygon_qems,
                                      vertex_qems=self._vertex_qems,
                                      border_plane_qems=self._border_plane_qems)

        self._qem_errors = [None] * len(poly_mesh.edges)
        self._square_lengths = [None] * len(poly_mesh.edges)
        self._collapse_positions = [None] * len(poly_mesh.edges)
        self._criteria = []
        self._coef_qem = 0.5
        self.updateCriteria(self._edges)
        self._criteria.sort()

        self._vertex_clusters = VertexClusters(poly_mesh, self._topo)

    #-----------------

    def check(self, label="", poly=False, vert=False, edge=False, error=False, criteria=False, vertex_clusters=False, verbose=False, npoly=-1, nvert=-1, nedge=-1):
        print()
        print("====== Check PolyReduce", label, "==========")
        print("object name :", self._object.name)
        print("mesh name :", self._poly_mesh.name)

        if poly:
            print("polygon qems : ")
            if npoly==-1:
                for i in range(len(self._poly_mesh.polygons)):
                    if verbose:
                        if self._polygon_qems[i]==None:  print(i, "----"*10)
                        else: self._polygon_qems[i].check(str(i))
                    else:
                        if self._polygon_qems[i]==None:  print(i, "----")
                        else: print(i, "Poly")
            else:
                if verbose:
                    if self._polygon_qems[npoly]==None:  print(npoly, "----"*10)
                    else: self._polygon_qems[npoly].check(str(npoly))
                else:
                    if self._polygon_qems[npoly]==None:  print(npoly, "----")
                    else: print(npoly, "Poly")

        if vert:
            print("vertex qems : ")
            if nvert==-1:
                for i in range(len(self._poly_mesh.vertices)):
                    if verbose:
                        if self._vertex_qems[i]==None:  print(i, "----"*10)
                        else: self._vertex_qems[i].check(str(i))
                    else:
                        if self._vertex_qems[i]==None:  print(i, "----")
                        else: print(i, "Vert")
            else:
                    if verbose:
                        if self._vertex_qems[nvert]==None:  print(nvert, "----"*10)
                        else: self._vertex_qems[nvert].check(str(nvert))
                    else:
                        if self._vertex_qems[nvert]==None:  print(nvert, "----")
                        else: print(nvert, "Vert")

        if edge:
            print("edge qems : ")
            if nedge==-1:
                for i in range(len(self._poly_mesh.edges)):
                    if verbose:
                        if self._edge_qems[i]==None:  print(i, "----"*10)
                        else: self._edge_qems[i].check(str(i))
                    else:
                        if self._edge_qems[i]==None:  print(i, "----")
                        else: print(i, "Edge")
            else:
                if verbose:
                    if self._edge_qems[nedge]==None:  print(nedge, "----"*10)
                    else: self._edge_qems[nedge].check(str(nedge))
                else:
                    if self._edge_qems[nedge]==None:  print(nedge, "----")
                    else: print(nedge, "Edge")

        if error:
            print("edge collapse errors : ")
            for i in range(len(self._poly_mesh.edges)):
                if self._edge_qems[i]==None:  print(i, "----")
                else: print(i, self._edge_collapse_err[i])

        if criteria:
            print("Sorting criteria : ")
            print("Cost \t qem_error \t edge length \t collapse position \t edge index")
            if nedge==-1:
                for i in range(len(self._criteria)):
                    print(self._criteria[i])
            else:
                lst=[i for i in range(len(self._criteria)) if self._criteria[i][4]==nedge]
                if len(lst)>0:
                    print(self._criteria[lst[0]])
                else:
                    print("PolyReduce::check : no edge number ", nedge)

        if vertex_clusters:
            self._vertex_clusters.check("vertex clusters")

    #-----------------

    def defineInitialLengths(self):
        '''
        Fill in the _initial_lengths list with the initial
        length of all edges.
        '''
        for ne in range(len(self._poly_mesh.edges)):
            ind_v1 = self._poly_mesh.edges[ne].vertices[0]
            ind_v2 = self._poly_mesh.edges[ne].vertices[1]
            p1 = np.array(self._poly_mesh.vertices[ind_v1].co)
            p2 = np.array(self._poly_mesh.vertices[ind_v2].co)
            dp = p2 - p1
            self._initial_lengths[ne] = np.dot(dp,dp)

    #-----------------

    def getCriteriaIndex(self, num_edge):
        '''
        num_edge : an integer : an edge index.
        return value : an integer
        The order or the edges in the criteria list changes during the
        course of the reduction. This function looks for the edge number
        num_edge in the criteria list. If the edge is found, the return
        value is the index at which the edge was found. If not, -1 is
        returned.
        '''
        lst = [i for i in range(len(self._criteria)) if self._criteria[i][4]==num_edge]
        assert len(lst)<2
        if len(lst)==0:
            return -1
        else:
            return lst[0]


    #-----------------

    def removeCriteriaForCollapsedEdges(self):
        '''
        All edges that belong to self._collapsed_edges must not
        belong to the criteria list.
        '''
        for i in range(len(self._criteria)-1,-1,-1):
            if self._criteria[i][4] in self._collapsed_edges:
                del self._criteria[i]

    #-----------------

    def updateCriteria(self, edges):
        '''
        # edges : a list of integers between 0 and len(poly_mesh.edges)-1
        # return value : None
        # side effect : the criteria list is defined.
        # The criteria list enables to sort the edges in decreasing order of
        # collapse priority. The criteria has as many rows as the number of edges.
        # column 0 : cost,
        # column 1 : qem_error,
        # column 2 : edge square length
        # column 3 : best position for collapse,
        # column 4 : edge index
        # The cost is a linear combination of length and qem error with coef qem_coef.
        '''
        assert 0 <= self._coef_qem <= 1

        max_length_square = 0
        max_qem_error = 0
        max_length_increase = 0
        ind_max_length_increase = -1

        # Calculate qem_error and edge length only for edges
        for ne in edges:
            ind_v1 = self._poly_mesh.edges[ne].vertices[0]
            ind_v2 = self._poly_mesh.edges[ne].vertices[1]
            p1 = np.array(self._poly_mesh.vertices[ind_v1].co)
            p2 = np.array(self._poly_mesh.vertices[ind_v2].co)
            dp = p2 - p1
            square_length = np.dot(dp,dp)
            if square_length > pr_very_small:
                (pos, qem_error) = self._edge_qems[ne].bestCollapsePosition()
                self._square_lengths[ne] = square_length
                self._qem_errors[ne] = abs(qem_error)
                self._collapse_positions[ne] = pos

        # Calculate the maximum qem_error and max_edge_length for all edges

        for ne in self._edges:
            ind_v1 = self._poly_mesh.edges[ne].vertices[0]
            ind_v2 = self._poly_mesh.edges[ne].vertices[1]
            p1 = np.array(self._poly_mesh.vertices[ind_v1].co)
            p2 = np.array(self._poly_mesh.vertices[ind_v2].co)
            dp = p2 - p1
            square_length = np.dot(dp,dp)
            length_increase = square_length / self._initial_lengths[ne]
            if square_length > pr_very_small:
                if self._qem_errors[ne]>max_qem_error:
                    max_qem_error = self._qem_errors[ne]

                if square_length > max_length_square:
                    max_length_square = square_length

            if length_increase > max_length_increase:
                max_length_increase = length_increase
                ind_max_length_increase = ne

        if max_length_square==0 or max_qem_error==0:
            print("!!!!!!!!! ne",ne,"max_length_square",max_length_square,"max_qem_error",max_qem_error, "!!!!!!!!!!!!!")

        # Sinci max_qem_error ad max_edge_lengths may have changed, update all criteria values.
        for ne in self._edges:
            qem_error = abs(self._qem_errors[ne]) / max_qem_error
            length_increase = square_length / self._initial_lengths[ne]
            coef_length_increase = max_length_increase / length_increase
            square_length = self._square_lengths[ne] / max_length_square
            cost = self._coef_qem * qem_error + (1-self._coef_qem)*square_length
            # cost = (self._coef_qem * qem_error + (1-self._coef_qem)*square_length)*coef_length_increase*coef_length_increase
            # cost = qem_error * square_length
            ind = self.getCriteriaIndex(ne)
            if ind == -1:
                self._criteria += [(cost, qem_error, square_length, self._collapse_positions[ne], ne)]
            else:
                self._criteria[ind] = (cost, qem_error, square_length, self._collapse_positions[ne], ne)

        self.removeCriteriaForCollapsedEdges()

    #-----------------

    def updatePolygonQEMs(self, poly_indices):
        '''
        poly_indices : a list of integers : polygon indices
        Return value : none
        This function re-calculates the QEMs of the polygons
        whose indices are in poly_indices. And it updates
        the _polygon_qems list.
        '''
        for npoly in poly_indices:
            self._polygon_qems[npoly] = QEM(self._poly_mesh, "POLYGON", npoly, self._topo)

    #-----------------

    def updateBorderPlaneQEMs(self, edge_indices):
        '''
        edge_indices : a list of integers : edge indices
        Return value : none
        For each edge, this function re-calculates the
        border plane QEMs.
        '''
        for ne in edge_indices:
            self._border_plane_qems[ne] = QEM(self._poly_mesh,"BORDER_PLANE", ne, self._topo)

    #-----------------

    def updateVertexQEMs(self, cluster_indices):
        '''
        cluster_indices : a list of integers : cluster indices
        Return value : none
        For each cluster, this function re-calculates the QEMs
        of one of the vertices and propagates it to the other
        vertices of the cluster. And it updates the _vertex_qems
        list.
        '''
        verts = []
        for ind_cl in cluster_indices:
            verts += self._vertex_clusters._clusters[ind_cl]

        for nv in verts:
            self._vertex_qems[nv] = QEM(self._poly_mesh, "VERTEX", nv, self._topo,
                                       polygon_qems=self._polygon_qems, border_plane_qems=self._border_plane_qems)

    #-----------------

    def updateEdgeQEMs(self, edge_indices):
        '''
        edge_indices : a list of integers : edge indices
        Return value : none
        This function re-calculates the QEMs of the edges
        whose indices are in edge_indices. And it updates
        the _edge_qems list.
        '''
        for ne in edge_indices:
            self._edge_qems[ne] = QEM(self._poly_mesh, "EDGE", ne, self._topo,
                                      polygon_qems=self._polygon_qems,
                                      vertex_qems=self._vertex_qems,
                                      border_plane_qems=self._border_plane_qems)
    #-----------------

    def selectNFirstEdgesInCriteria(self, N):
        '''
        '''
        bpy.ops.object.mode_set(mode="OBJECT")
        for n in range(N):
            ne = self._criteria[n][4]
            self._poly_mesh.edges[ne].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="EDGE")

    #-----------------

    def mergeAllZeroLengthEdges(self):
        '''
        parameters : None
        return value : None
        All edges whose length is smaller than pr_very_small are
        merged.
        '''
        bpy.ops.object.mode_set(mode="OBJECT")
        for nv in range(len(self._poly_mesh.vertices)):
            self._poly_mesh.vertices[nv].select = True
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode="OBJECT")

    #-----------------

    def decreaseAllExcentricities(self, ind_edge, threshold):
        '''
        ind_edge : an edge index
        threshold : a float
        return value : None
        This function calculates the excentricities at both ends of
        the edge. If the excentricities are greater than threshold,
        then the edges are replaced in order to derease the
        excentricities to the minimum value.
        '''
        nv_end1 = self._poly_mesh.edges[ind_edge].vertices[0]
        nv_end2 = self._poly_mesh.edges[ind_edge].vertices[1]
        (exc1, exc2) = self._vertex_clusters.edgeExcentricityValue(ind_edge)
        if abs(exc1) > threshold or abs(exc2) > threshold:
            nb_samples=11
            min_exc1 = min_exc2 = threshold+1
            min_alpha1 = min_alpha2 = -1
            # for i in range(nb_samples):
            for i in range(nb_samples-1):
                alpha1 = i/(nb_samples-1)
                alpha2 = 1 - alpha1
                (exc1,exc2) = self._vertex_clusters.edgeExcentricityValue(ind_edge, alpha1, alpha2)
                if abs(exc1) < min_exc1:
                    min_exc1 = abs(exc1)
                    min_alpha1 = alpha1
                if abs(exc2) < min_exc2:
                    min_exc2 = abs(exc2)
                    min_alpha2 = alpha2

            if min_alpha1 != -1 and min_alpha2 != -1 and min_alpha1 >= min_alpha2:
                min_alpha2 = min(1,min_alpha2 + 0.05)
                min_alpha1 = max(0,min_alpha1 - 0.05)

            p1 = np.array(self._poly_mesh.vertices[nv_end1].co)
            p2 = np.array(self._poly_mesh.vertices[nv_end2].co)
            if min_alpha1 > 0:
                p = (1-min_alpha1)*p1 + min_alpha1*p2
                self._poly_mesh.vertices[nv_end1].co = p
            if min_alpha2 > 0:
                p = (1-min_alpha2)*p1 + min_alpha2*p2
                self._poly_mesh.vertices[nv_end2].co = p

    #-----------------

    def reduce(self, aim_nb_edges=-1, collapse_rate=-1):
        '''
        aim_nb_edges : an integer
        collapse_rate : a float between 0 and 1
        if aim_nb_edges==-1, then this value is ignored
        if collapse_rate==-1, then this value is ignored
        One and only one of these values are to be taken into account.
        Algorithm :
        Repeat (nb_edges_to_reduce - aim_nb_edges) times :
              Take the first edge in the criteria list
              Superimpose both ends (and their clusters) at the right place
              and put the second end (and its co-clusters) into the first end's cluster
              update the qems of all adjoining faces, vertices and edges.
              update the critera list and
              affect a very high cost to all collapsed edges
              sort again.
        '''

        # file = open("dump_edges.txt","w")
        # file.write("[")

        assert aim_nb_edges==-1 and collapse_rate!=-1 or aim_nb_edges!=-1 and collapse_rate==-1
        if collapse_rate != -1:
            aim_nb_edges = int(len(self._edges) * (1-collapse_rate))

        # print("88888 ", len(self._edges))
        nb_collapses = len(self._edges)-aim_nb_edges
        for i in range(nb_collapses):
            unselectAllComponents(self._poly_mesh)
            collapse_edge = self._criteria[0][4]

            # print(i,"/",nb_collapses, "----------------- collapsed edge : ", collapse_edge, "-------------")

            # file.write(str(collapse_edge))
            # if i != nb_collapses-1:
            #      file.write(",")

            # print("---- collapse ---")
            collapse_position = self._criteria[0][3]
            self._vertex_clusters.mergeClusters(collapse_edge, collapse_position)
            self._collapsed_edges += [collapse_edge]

            nv_end1 = self._poly_mesh.edges[collapse_edge].vertices[0]
            nv_end2 = self._poly_mesh.edges[collapse_edge].vertices[1]

            '''
            # Decrease excentricities before calculating qems.
            adjacent_edges=[]
            adjacent_edges += self._vertex_clusters.adjacentEdges(nv_end1)
            adjacent_edges += self._vertex_clusters.adjacentEdges(nv_end2)
            adjacent_edges = [i for i in adjacent_edges if i in self._edges and i != collapse_edge]
            adjacent_edges = list(set(adjacent_edges))
            for ne in adjacent_edges:
                self.decreaseAllExcentricities(ne, self._threshold_excentricity)
            '''

            # print("---- update neighboring polygon qems ---")
            neighboring_polygons=[]
            min_area = pr_very_small
            neighboring_polygons += self._vertex_clusters.adjacentPolygons(nv_end1, min_area)
            neighboring_polygons += self._vertex_clusters.adjacentPolygons(nv_end2, min_area)
            neighboring_polygons = list(set(neighboring_polygons))
            self.updatePolygonQEMs(neighboring_polygons)

            # print("---- update neighboring border plane qems ---")
            vertex_polygons=[]
            vertex_polygons += self._topo._vertex_polygons[nv_end1]
            vertex_polygons += self._topo._vertex_polygons[nv_end2]
            neighboring_border_edges=[]
            for npoly in vertex_polygons:
                neighboring_border_edges += self._vertex_clusters.adjacentBorderEdges(npoly)
            self.updateBorderPlaneQEMs(neighboring_border_edges)

            # print("---- update neighboring vertex qems ---")
            neighboring_clusters=[]
            neighboring_clusters += self._vertex_clusters.neighboringClusters(nv_end1)
            neighboring_clusters += self._vertex_clusters.neighboringClusters(nv_end2)
            neighboring_clusters = list(set(neighboring_clusters))
            self.updateVertexQEMs(neighboring_clusters)

            # print("---- update neighboring edge qems ---")
            adjacent_edges=[]
            adjacent_edges += self._vertex_clusters.adjacentEdges(nv_end1)
            adjacent_edges += self._vertex_clusters.adjacentEdges(nv_end2)
            adjacent_edges = [i for i in adjacent_edges if i in self._edges and i != collapse_edge]
            adjacent_edges = list(set(adjacent_edges))
            self.updateEdgeQEMs(adjacent_edges)

            # print("---- update criteria edge qems ---")
            self.updateCriteria(adjacent_edges)
            self._criteria.sort()
            self.selectNFirstEdgesInCriteria(2)
            # pr.check("update--->>>", criteria=True, verbose=True, nedge=2493)

        # print("----=== End : merge vertex clusters ")
        self.mergeAllZeroLengthEdges()
        # file.write("]")
        # file.close()



'''
object_name = "Plane"
edges = [9,14,12,19,4]
'''
'''
object_name = "Cube"
edges = []

object = bpy.data.objects[object_name]
poly_mesh = object.data
topo = Topology(poly_mesh)

if edges==[]:
    edges = list(range(len(poly_mesh.edges)))

pr = PolyReduce(poly_mesh, edges_to_reduce=edges)
# pr = PolyReduce(poly_mesh, edges_to_reduce=edges, preserve_borders=False)
# pr.check("init", criteria=True, verbose=True)
pr.reduce(aim_nb_edges=15)
'''
