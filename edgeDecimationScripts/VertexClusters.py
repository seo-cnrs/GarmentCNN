'''
Tuesday January 16th 2024
Arash HABIBI
VertexClusters.py

Collapsing edges would reorder all vertices/edges/faces.
This is the reason why, during the course of polygon reduction,
we will not truly replace an edge by a vertex, but merely superimpose
these edges. This produces "vertex clusters" : several vertices placed
at the same point, and many degenerate edges and polygons.
When the polygon reduction is finished and satisfactory, all zero-length
edges will be actually collapsed. In other words, each vertex cluster
will be replaced by one vertex.

In this new version (2), we characterize each vertex cluster by a vector
called excentricity, which is the sum of the unit vectors of all edges
stemming from that vertex.
'''

import bpy
import sys
import os
import numpy as np

scene_path = os.getcwd()
project_path='/'
words=scene_path.split('/')
for i in range(1,len(words)-1):
    project_path += words[i]+'/'
sys.path.append(project_path+"scripts")

from Topology import *

#======================================================================

def objectFromMeshName(mesh_name):
    for obj in bpy.data.objects:
        if obj.type=="MESH":
            if obj.data.name == mesh_name:
                return obj

#======================================================================

class VertexClusters:

    def __init__(self,poly_mesh,topology):
        self._object = objectFromMeshName(poly_mesh.name)
        self._poly_mesh = poly_mesh
        self._topology = topology
        self._nb_vertices = len(poly_mesh.vertices)
        self._nb_clusters = self._nb_vertices
        self._cluster_indices = list(range(self._nb_vertices))
        # cluster_indices : for each vertex indicates the cluster to which it belongs
        self._clusters=[]
        for i in range(self._nb_vertices):
            self._clusters += [[i]]
        self._excentricities=[None]*self._nb_vertices

    #------------------

    def check(self, label=""):
        print("======= check clusters " + label + "============")
        print("object : " + self._object.name)
        print("poly_mesh : " + self._poly_mesh.name)
        print("cluster indices : ",self._cluster_indices)
        print("clusters")
        for i in range(len(self._clusters)):
            print(i,self._clusters[i])
        print("-----")

    #------------------

    def vertexIndicesInSameCluster(self, nv):
        '''
        nv is a vertex index.
        return value : an integer list representing the indices of
        all vertices that belong to the same cluster as vertex nv.
        '''
        cluster_index = self._cluster_indices[nv]
        return self._clusters[cluster_index]

    #------------------

    def adjacentPolygons(self, nv, min_area):
        '''
        nv : an integer : a vertex index
        epsilon : a floating point (low value)
        return value : a list of integers
        This list comprises the set of all the faces
        of the object which share vertex index nv
        or one of its co-clusters, and whose area
        is greater than min_area.
        '''
        polygons = []
        vertices_in_same_cluster = self.vertexIndicesInSameCluster(nv)
        for i in vertices_in_same_cluster:
            polys = self._topology._vertex_polygons[i]
            for np in polys:
                area = self._poly_mesh.polygons[np].area
                if area > min_area:
                    polygons += [np]
        return list(set(polygons))

    #------------------

    def neighboringClusters(self, nv):
        '''
        nv : an integer : a vertex index
        return value : a list of integers
        This list comprises the indices of
        all vertices that are linked to nv
        or one of its co-clusters AND who
        do not belong to the same cluster
        as vertex nv.
        '''
        clusters=[]
        cluster_index = self._cluster_indices[nv]
        vertices_in_same_cluster = self.vertexIndicesInSameCluster(nv)
        for i in vertices_in_same_cluster:
            neighbors = self._topology._vertex_neighbors[i]
            for nn in neighbors:
                if self._cluster_indices[nn] != cluster_index and self._cluster_indices[nn] not in clusters:
                    clusters += [self._cluster_indices[nn]]
        return clusters

    #------------------

    def adjacentEdges(self, nv):
        '''
        nv : an integer : a vertex index
        return value : a list of integers
        This list comprises the set of all the edges
        of the object which share vertex index nv
        or one of its co-clusters.
        '''
        edges=[]
        cluster_index = self._cluster_indices[nv]
        vertices_in_same_cluster = self.vertexIndicesInSameCluster(nv)
        for i in vertices_in_same_cluster:
            v_edges = self._topology._vertex_edges[i]
            for ne in v_edges:
                ind_v1 = self._poly_mesh.edges[ne].vertices[0]
                ind_v2 = self._poly_mesh.edges[ne].vertices[1]
                if self._cluster_indices[ind_v1] != cluster_index or self._cluster_indices[ind_v2] != cluster_index:
                    edges += [ne]
        return edges

    #------------------

    def adjacentBorderEdges(self, npoly):
        '''
        npoly : an integer : a polygon index
        return value : a list of integers
        The returned list represents the set of border edges
        around face npoly.
        For edge end1 and edge end2, we look in the cluster for border vertices
        For each border vertex of cluster1, we find the border edges
        and the other end of the border edge.
        If the other end of the border edge belongs to cluster2, then
        the border edge is appended to our list.
        '''
        res=[]
        border_vertices = self._topology.flattenList(self._topology._border_vertices)
        border_edges    = self._topology.flattenList(self._topology._border_edges)
        polygon_edges   = self._topology._face_edges[npoly]
        for ne in polygon_edges:
            ind_v1 = self._poly_mesh.edges[ne].vertices[0]
            ind_v2 = self._poly_mesh.edges[ne].vertices[1]
            ind_cluster1 = self._cluster_indices[ind_v1]
            ind_cluster2 = self._cluster_indices[ind_v2]
            if ind_cluster1 != ind_cluster2:
                verts_cluster1 = self._clusters[ind_cluster1]
                verts_cluster2 = self._clusters[ind_cluster2]
                for nv in verts_cluster1:
                    if nv in border_vertices:
                        this_vertex_edges = self._topology._vertex_edges[nv]
                        for nve in this_vertex_edges:
                            if nve in border_edges:
                                ind_v1 = self._poly_mesh.edges[nve].vertices[0]
                                ind_v2 = self._poly_mesh.edges[nve].vertices[1]
                                if ind_v1==nv:
                                    other_vertex = ind_v2
                                else:
                                    other_vertex = ind_v1
                                if other_vertex in verts_cluster2:
                                    res += [nve]
        return res

    #------------------

    def clusterExcentricityVector(self, ind_cl, p=()):
        '''
        ind_cl : a cluster index.
        return value : a vector which is the sum of the unit
        vectors of all edges stemming from vertex cluster ind_cl.
        When several edges link the same pair of vertex clusters,
        the unit vector must be counted only once.
        if p is None, then the position of the central point
        is the position of a vertex in ind_cl. Otherwise, p is
        taken as the position of the central vertex.
        '''
        # choice of the first (arbitrary) vertex of the cluster.
        excentricity=np.array((0,0,0))
        ind_v1 = self._clusters[ind_cl][0]
        if len(p)==0:
            p1 = np.array(self._poly_mesh.vertices[ind_v1].co)
        else:
            p1 = p
        neighbors = self.neighboringClusters(ind_v1)
        for nn in neighbors:
            assert nn != ind_cl
            ind_v2 = self._clusters[nn][0]
            p2 = np.array(self._poly_mesh.vertices[ind_v2].co)
            dp = p2 - p1
            l = np.linalg.norm(dp)
            if l>0:
                dp = (1/l) * dp
                excentricity = excentricity + dp
        return excentricity

    #------------------

    def edgeExcentricityValue(self, ind_edge, alpha1=0, alpha2=1):
        '''
        ind_edge : an edge index.
        return value : a pair of two floats.
        For each end vertex, this function operates
        a dot product between the excentricity of the
        vertex and edge's unit vector going from end1 to end2.
        If the dot product of end1 is positive, it means
        that that end is being pulled away from the edge and
        should be pushed back towards the edge.
        Conversely, if the dot product of end2 is negative, it
        means that that end must be pulled back towards the edge.
        In all other cases, the work will be done, if necessary,
        in other edges.
        The two return values are the dot products.
        - If alpha1 = 0 then the first excentricity is calculated
        exactly at the first vertex.
        - If alpha1 = 1 then the first excentricity is calculated
        exactly at the second vertex.
        For other values, the center of the excentricity can be
        anywhere along the edge. Same thing for alpha2 and the
        second excentricity.
        '''
        ind_v1 = self._poly_mesh.edges[ind_edge].vertices[0]
        ind_v2 = self._poly_mesh.edges[ind_edge].vertices[1]
        ind_cl1 = self._cluster_indices[ind_v1]
        ind_cl2 = self._cluster_indices[ind_v2]
        if ind_cl1 == ind_cl2:
            return (0,0)
        else:
            p_end1 = np.array(self._poly_mesh.vertices[ind_v1].co)
            p_end2 = np.array(self._poly_mesh.vertices[ind_v2].co)
            dp = p_end2 - p_end1
            l = np.linalg.norm(dp)
            if l==0:
                return (0,0)
            else:
                u_12 = (1/l) * dp
                p1 = (1-alpha1)*p_end1 + alpha1*p_end2
                p2 = (1-alpha2)*p_end1 + alpha2*p_end2
                exc_end1 = self.clusterExcentricityVector(ind_cl1, p1)
                exc_end2 = self.clusterExcentricityVector(ind_cl2, p2)
                return (np.dot(exc_end1,u_12), np.dot(exc_end2,-u_12))

    #------------------

    def mergeClusters(self, ne, position=-1):
        '''
        ne is an edge index
        position is a point in space (or -1)
        return value : None
        The vertices belonging to the same cluster as
        the edge's ends must also be merged.
        If nv1 belongs to cl1 and nv2 belongs to cl2, we arbitrarily
        choose to put all the vertices of cl2 in cl1. cl2 will be
        an empty cluster.
        All vertices belonging to cluster cl1 is placed at position.
        if position is -1 (if not specified) the position is set
        to the middle of the edge.
        '''
        nv1 = self._poly_mesh.edges[ne].vertices[0]
        nv2 = self._poly_mesh.edges[ne].vertices[1]
        ind_cl1 = self._cluster_indices[nv1]
        ind_cl2 = self._cluster_indices[nv2]
        cl1 = self._clusters[ind_cl1]
        cl2 = self._clusters[ind_cl2]
        for i in range(len(cl2)):
            cl = cl2[i]
            self._clusters[ind_cl1] += [cl]
            self._cluster_indices[cl] = ind_cl1
        self._clusters[ind_cl2] = []

        if position==-1:
            p1 = self._poly_mesh.vertices[nv1].co
            p2 = self._poly_mesh.vertices[nv2].co
            position=((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2)

        bpy.ops.object.mode_set(mode="OBJECT")
        for i in range(len(self._clusters[ind_cl1])):
            ind_v = self._clusters[ind_cl1][i]
            self._poly_mesh.vertices[ind_v].co = position
        bpy.ops.object.mode_set(mode="EDIT")

    #------------------

'''
object = bpy.data.objects["Plane"]
poly_mesh = object.data
topo = Topology(poly_mesh)
vc = VertexClusters(poly_mesh,topo)
vc.check("init")
vc.mergeClusters(3)
vc.mergeClusters(6)
vc.mergeClusters(10)

vc.mergeClusters(8)
vc.mergeClusters(2)

vc.mergeClusters(13)

vc.mergeClusters(1)
vc.mergeClusters(18)

vc.mergeClusters(0)
vc.mergeClusters(19)
vc.mergeClusters(22)

vc.check("apres")

print(vc.vertexIndicesInSameCluster(0))
print(vc.vertexIndicesInSameCluster(8))
print(vc.vertexIndicesInSameCluster(11))

print(vc.adjacentPolygons(10,0.0001))
print(vc.adjacentPolygons(11,0.0001))

print(vc.neighboringClusters(1))
print(vc.neighboringClusters(4))
print(vc.neighboringClusters(9))
print(vc.neighboringClusters(10))
print(vc.neighboringClusters(11))

print(vc.adjacentEdges(5))
print(vc.adjacentEdges(11))

print("----",vc.adjacentBorderEdges(2))
'''
