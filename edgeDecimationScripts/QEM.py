'''
Tuesday January 16th 2024
Arash HABIBI
QEM.py

The previous version is capable of evaluating, for each edge collapse
the distance with the initial shape. So we can sort the edges
in the increasing order of collapse error. Furthermore, it is
also capable of determining, for each edge collapse, the best
place to put the resulting vertex.

The current version implements a lazy QEM calculation.
If the aim is to split a limited set of edges, then we will only
calculate the qems associated with the connected polygons
and vertices.

The QEMs only take in account deformations in the direction of
face normals, which is OK as long as the points are not on the border.
A vertex on the border can be moved within the face's plane thus without
changing the QEM. But this deformation considerably change the overal
shape of the object. Thus in order to calclate a vertex's QEM, we take
in account the QEM of the adjacent polygons, but we also calculate the
QEM of the "border_planes" associated with the border edges.
A border_plane associate with a border edge is a plane which contains
the border edge, but also the normal of the adjacent polygon. In this
way, moving the vertex even in the plane of the adjacent faces does
modify the QEM.
'''

import bpy
import numpy as np
import sys
import os
import random

scene_path = os.getcwd()
project_path='/'
words=scene_path.split('/')
for i in range(1,len(words)-1):
    project_path += words[i]+'/'
sys.path.append(project_path+"scripts")
from Topology import *

q_is_degenerate=0
p_test = (-22.13484053044109, -2.8304320742933906, 1.2973550269909573, 1)

#======================================================================

def objectFromMeshName(mesh_name):
    for obj in bpy.data.objects:
        if obj.type=="MESH":
            if obj.data.name == mesh_name:
                return obj

#======================================================================

class QEM:
    '''
    Quadric Error Matrix
    A QEM is a 4x4 matrix
    Each polygon on on object can be characterized by a QEM matrix
    Each vertex is also characterized by a QEM matrix
    and so is each edge.
    With such a QEM and a position in space, one can calculate an error value.
    In order to calculate a vertex QEM, one must simply add the QEMs of the
        adjoining polygons
    In order to calculate an edge QEM, one must simply add the QEMs of the
        adjoining vertices
    In order to calculae a polygone QEM, ... well it's more complicated.
    For an edge QEM, one must also be able to calculate the position for which
    the error value is the smallest
    '''

    #-----------------

    def __init__(self, poly_mesh, type, index, topology, polygon_qems=None, vertex_qems=None, border_plane_qems=None):
        '''
        poly_mesh : a polygon mesh containing Nv vertices, Ne edges and Np polygons
        type : "POLYGON", "VERTEX" or "EDGE"
        topology : la topologie associée à poly_mesh
        polygon_qems : a list of Np qems, one for each polygon
        vertex_qems : a list of Nv qems, one for each vertex
        - if type is "POLYGON",
          polygon_qems  and vertex_qems are both ignored (should be omitted)
        - if type is "BORDER_PLANE",
          polygon_qems  and vertex_qems are both ignored (should be omitted)
        - if type is "VERTEX",
          vertex_qems is ignored (should be omitted) but not polygon_qems and border_plane_qems
        - if type is "EDGE",
          polygon_qems, vertex_qems and border_plane_qems are necessary (should not be omitted)

          polygon_qems and vertex_qems are necessary (should not be omitted)
        if type is "VERTEX then index must be a vertex index (between 0 and Nv-1)
        if type is "EDGE" then index must be an edge index (between 0 and Ne-1)
        if type is "POLYGON" then index must be a polygon index (between 0 and Np-1)
        if type is "BORDER_PLANE" then index must be an edge index (between 0 and Ne-1)
        Moreover, in this last case, edge index must be a border edge.

        '''
        self._object = objectFromMeshName(poly_mesh.name)
        self._poly_mesh = poly_mesh
        self._type = type
        self._index = index
        self._Q = np.matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

        self._polygon_qems = polygon_qems
        self._vertex_qems = vertex_qems
        self._border_plane_qems = border_plane_qems

        assert type == "POLYGON" or type == "VERTEX" or type == "EDGE" or type == "BORDER_PLANE"

        if type=="POLYGON":
            # assert polygon_qems==None and vertex_qems==None
            assert 0 <= index < len(poly_mesh.polygons)
            self.setPolygonQem(index)

        if type=="BORDER_PLANE":
            assert 0 <= index < len(poly_mesh.edges)
            assert index in topology.flattenList(topology._border_edges)
            self.setBorderPlaneQem(index, topology)

        elif type=="VERTEX":
            assert polygon_qems != None and border_plane_qems != None
            assert 0 <= index < len(poly_mesh.vertices)
            self.setVertexQem(index, topology)

        elif type=="EDGE":
            assert polygon_qems != None and vertex_qems != None and border_plane_qems != None
            assert 0 <= index < len(poly_mesh.edges)
            self.setEdgeQem(index, topology)

    #-----------------

    def check(self,label=''):
        print("====== Check QEM", label, "==========")
        print("object name :", self._object.name)
        print("mesh name :", self._poly_mesh.name)
        print(self._type.lower(),"index",self._index)
        print(self._Q)
        print("------------")

    #-----------------

    def setMatrix(self,array):
        assert self._Q.size == len(array) == 16
        count=0
        mtx=[]
        for i in range(4):
            row = []
            for j in range(4):
                row += [array[count]]
                count += 1
            mtx += [row]
            self._Q = np.matrix(mtx)

    #-----------------

    def add(self,q):
        self._Q = np.add(self._Q,q._Q)

    #-----------------

    def setPolygonQem(self, index):
        '''
        index : integer index of current polygon
        return value : None
        Side effect : the _Q matrix of the qem is set to
        (a*a, a*b, a*c, a*d)
        (b*a, b*b, b*c, b*d)
        (c*a, c*b, c*c, c*d)
        (d*a, d*b, d*c, d*d)
        where a, b, c and d are the plane coefficients of
        the polygon with a*a + b*b + c*c = 1
        '''
        poly_vertex_indices = self._poly_mesh.polygons[index].vertices
        nb_vertices=len(poly_vertex_indices)
        assert nb_vertices > 2
        found_non_degenerate=False
        nv=0
        while nv < nb_vertices and not found_non_degenerate:
            # print(">>>", nv, nb_vertices, (nv+1)%nb_vertices, (nv+2)%nb_vertices)
            p1 = np.array(self._poly_mesh.vertices[poly_vertex_indices[nv]].co)
            p2 = np.array(self._poly_mesh.vertices[poly_vertex_indices[(nv+1)%nb_vertices]].co)
            p3 = np.array(self._poly_mesh.vertices[poly_vertex_indices[(nv+2)%nb_vertices]].co)
            if not np.array_equal(p1,p2) and not np.array_equal(p2,p3) and not np.array_equal(p1,p3):
                found_non_degenerate=True
            else:
                nv += 1
        #assert found_non_degenerate==True
        # if not found_non_degenerate:
        #    print("polygon" , index , "unable to find two non-degenerate edges")
        #else:
        normal = np.cross((p2-p1),(p3-p2))
        len_normal = np.linalg.norm(normal)
        assert len_normal != 0
        normal = (1/len_normal) * normal
        plane_coefs = [normal[0], normal[1], normal[2], -np.dot(p1,normal)]
        matrix_coefs=[]
        for i in range(4):
            for j in range(4):
                matrix_coefs += [plane_coefs[i]*plane_coefs[j]]
        self.setMatrix(matrix_coefs)

    #-----------------

    def setBorderPlaneQem(self, index, topology):
        '''
        index : integer index of a border edge, thus with
        one adjacent polygon of index np.
        return value : None
        Side effect : the _Q matrix of the qem is set to
        (a*a, a*b, a*c, a*d)
        (b*a, b*b, b*c, b*d)
        (c*a, c*b, c*c, c*d)
        (d*a, d*b, d*c, d*d)
        where a, b, c and d are the plane coefficients of
        a plane containing the border_edge index and the normal
        vector of polygon np.
        '''
        # get the ends of the edge
        ind_v1 = self._poly_mesh.edges[index].vertices[0]
        ind_v2 = self._poly_mesh.edges[index].vertices[1]
        p1 = np.array(self._poly_mesh.vertices[ind_v1].co)
        p2 = np.array(self._poly_mesh.vertices[ind_v2].co)

        # get the normal vector of the adjacent face.
        npoly = topology._edge_faces[index][0]
        face_normal = np.array(self._poly_mesh.polygons[npoly].normal)

        # normalize this normal vector
        border_plane_normal = np.cross(p2-p1,face_normal)
        l = np.linalg.norm(border_plane_normal)
        assert l != 0
        border_plane_normal = (1/l) * border_plane_normal

        # find the fourth coefficient d.
        plane_coefs = [border_plane_normal[0], border_plane_normal[1], border_plane_normal[2], -np.dot(p1,border_plane_normal)]

        matrix_coefs=[]
        for i in range(4):
            for j in range(4):
                matrix_coefs += [plane_coefs[i]*plane_coefs[j]]
        self.setMatrix(matrix_coefs)

    #-----------------

    def setVertexQem(self, index, topology):
        '''
        index : integer index of current polygon
        return value : None
        Side effect : the _Q matrix of the qem is set
        '''

        debug=False

        adjoining_faces = topology._vertex_polygons[index]
        self.setMatrix([0]*16)
        for nf in adjoining_faces:
            if self._polygon_qems[nf]==None:
                self._polygon_qems[nf]=QEM(self._poly_mesh, "POLYGON", nf, topology)
            self.add(self._polygon_qems[nf])
            if debug:
                print("----------->>> setVertexQem : poly",nf,self._polygon_qems[nf]*p_test)

        # if the vertex is a border vertex, we have to add the border plane qems
        if index in topology.flattenList(topology._border_vertices):
            border_edges = topology._vertex_border_edges[index]
            assert hasattr(type(border_edges),'__len__')==True and len(border_edges)==2
            (ne1,ne2) = border_edges
            for ne in [ne1,ne2]:
                if self._border_plane_qems[ne]==None:
                    self._border_plane_qems[ne]=QEM(self._poly_mesh,"BORDER_PLANE", ne, topology)
                self.add(self._border_plane_qems[ne])


    #-----------------

    def setEdgeQem(self, index, topology):
        '''
        index : integer index of current polygon
        polygon_qems : list of the qems of each vertex
        return value : None
        Side effect : the _Q matrix of the qem is set
        '''
        debug=False

        ind_v1 = self._poly_mesh.edges[index].vertices[0]
        ind_v2 = self._poly_mesh.edges[index].vertices[1]
        self.setMatrix([0]*16)

        if self._vertex_qems[ind_v1]==None:
            self._vertex_qems[ind_v1]=QEM(self._poly_mesh, "VERTEX", ind_v1, topology, polygon_qems=self._polygon_qems, border_plane_qems=self._border_plane_qems)
        if self._vertex_qems[ind_v2]==None:
            self._vertex_qems[ind_v2]=QEM(self._poly_mesh, "VERTEX", ind_v2, topology, polygon_qems=self._polygon_qems, border_plane_qems=self._border_plane_qems)

        self.add(self._vertex_qems[ind_v1])
        self.add(self._vertex_qems[ind_v2])
        if debug:
            print("----------->>> setEdgeQem : vrtx",ind_v1,self._vertex_qems[ind_v1]*p_test)
            print("----------->>> setEdgeQem : vrtx",ind_v2,self._vertex_qems[ind_v2]*p_test)

    #-----------------

    def __mul__(self,row_vect):
        '''
        vect should be a simple row vector
        return value : a float
        This is a probably dangerous shortcut.
        The returned float is the result of
        vect * Q * vect.transpose
        '''
        if len(row_vect)==3:
            row_vect = tuple(row_vect) + (1,)
        col_vect = np.atleast_2d(row_vect).T
        product1 = self._Q * col_vect
        product2 = row_vect * product1
        return product2.item(0)

    #-----------------

    def bestCollapsePosition(self):
        '''
        The current qem type should be EDGE.
        The best collapse position is sought along the edge
        (even if the very best position can possibly be elsewhere)
        We seek for alpha such that p_collapse = (1-alpha)*p1 + alpha*p2
        where p1 and p2 are the edges end vertices.
        4 values of alpha are found. We choose the one who produces the
        smallest error value.
        return value : a tuple composed of the best collapse position
        and the value of the error.
        Calculations based on page 3 (footnote) of
        https://www.cs.cmu.edu/~./garland/Papers/quadrics.pdf
        '''
        assert self._type == "EDGE"

        ind_v1 = self._poly_mesh.edges[self._index].vertices[0]
        ind_v2 = self._poly_mesh.edges[self._index].vertices[1]
        v1 = np.array(tuple(self._poly_mesh.vertices[ind_v1].co) + (1,))
        v2 = np.array(tuple(self._poly_mesh.vertices[ind_v2].co) + (1,))
        Delta = v2-v1
        A = np.zeros(4)
        B = np.zeros(4)
        Qrow = np.zeros(4)
        for i in range(4):
            for j in range(4):
                Qrow[j] = self._Q.item(i,j)
            A[i] = np.dot(Qrow,v1)
            B[i] = np.dot(Qrow,Delta)
        numerator = -(np.dot(A,Delta) + np.dot(B,v1))
        denominator = 2 * np.dot(B,Delta)

        if denominator != 0:
            alpha = numerator / denominator
            alpha = max(0,min(1,alpha))
            v = (1-alpha)*v1 + alpha*v2
            v = tuple(v)[:3]
            error = self*v

        else: # Just in case : sample the v1,v2 segment and choose the point with the lowest error
            print("edge ", self._index, " : zero denom")
            N=11
            positions=[]
            errors=[]
            for i in range(N):
                alpha = i/(N-1)
                v = ((1-alpha)*v1[0] + alpha*v2[0],
                     (1-alpha)*v1[1] + alpha*v2[1],
                     (1-alpha)*v1[2] + alpha*v2[2])
                positions += [v]
                errors += [self*v]
            #-- still have to choose the lowest error
            ind_min=-1
            val_min=1e20
            for i in range(len(errors)):
                if errors[i]<val_min:
                    val_min = errors[i]
                    ind_min = i
            v = positions[ind_min]
            error = val_min
        return (v, error)

    #-----------------

"""
# Cube object
object = bpy.data.objects["Cube"]
poly_mesh = object.data
topo = Topology(poly_mesh)

polygon_qems=[]
for npol in range(len(poly_mesh.polygons)):
    polygon_qems += [QEM(poly_mesh, npol, topo)]
    # polygon_qems[npol].check(str(npol))

'''
# visual test of polygon qems
qem = polygon_qems[2]
# face 0 points towards -x
# face 2 points towards +x
for i in range(10):
    p = (3.5,random.uniform(-5,5),random.uniform(-5,5))
    print(p, qem * p)
'''
vertex_qems=[]
for nv in range(len(poly_mesh.vertices)):
    vertex_qems += [QEM(poly_mesh, nv, topo, polygon_qems=polygon_qems)]
    # vertex_qems[nv].check(str(nv))

'''
# visual test of vertex qems
# vertex 8 is in the middle of the cube's edge.
# It is not a corner.
qem = vertex_qems[8]
x = random.uniform(-5,5)
y = random.uniform(-5,5)
for i in range(10):
    z = random.uniform(-5,5)
    p = (x,y,z)
    print(p, qem * p)
'''

edge_qems=[]
for ne in range(len(poly_mesh.edges)):
    edge_qems += [QEM(poly_mesh, ne, topo, vertex_qems=vertex_qems)]
    # edge_qems[ne].check(str(ne))
    (p,e) = edge_qems[ne].bestCollapsePosition()
    print(ne,":",p,e)
"""
