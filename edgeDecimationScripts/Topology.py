'''
Thursday January 11th 2024
Arash HABIBI
Topology.py

In the previous version of Topology, there was no notion of border vertices or border edges.
'''
import bpy

class Topology:

    #-------------------------

    def __init__(self,poly_mesh):

        self._poly_mesh = poly_mesh

        self._nb_vertices = len(poly_mesh.vertices)
        self._nb_edges    = len(poly_mesh.edges)
        self._nb_polygons = len(poly_mesh.polygons)

        self._vertex_edges = self.vertexEdges()
        self._vertex_neighbors = self.vertexNeighbors()
        self._vertex_polygons = self.vertexPolygons()

        self._face_edges = self.faceEdges(self._vertex_edges)
        self._edge_faces = self.edgeFaces(self._face_edges)
        all_border_edges = self.borderEdges(self._edge_faces)
        (self._border_edges,self._border_vertices, self._vertex_border_edges) = self.sortBorderEdges(all_border_edges)
        self._edges_touching_border = self.edgesTouchingBorder(self._vertex_border_edges)

        self.sort()

    #-------------------------

    def vertexEdges(self):
        '''
        return value : list of _nb_vertices integer lists
        Each vertex v is associated with one integer list which represents
        the indices of the edges that meet at vertex v.
        '''
        vertex_edges=[]
        for i in range(self._nb_vertices):
            vertex_edges += [[]]
        # vertex_edges is a list of nb_vertices empty lists

        for e in self._poly_mesh.edges:
            vertex_edges[e.vertices[0]] += [e.index]
            vertex_edges[e.vertices[1]] += [e.index]
        return vertex_edges


    #-------------------------

    def vertexNeighbors(self):
        '''
        return value : list of _nb_vertices integer lists
        Each vertex v is associated with one integer list which represents
        the indices of the vertices linked to v by an edge.
        '''
        neighbors=[]
        for i in range(self._nb_vertices):
            neighbors += [[]]
        # neighbors is a list of nb_vertices empty lists

        for e in self._poly_mesh.edges:
            neighbors[e.vertices[0]] += [e.vertices[1]]
            neighbors[e.vertices[1]] += [e.vertices[0]]

        return neighbors


    #-------------------------

    def vertexPolygons(self):
        '''
        return value : list of _nb_vertices integer lists
        Each vertex v is associated with one integer list which represents
        the indices of the polygons that share vertex v.
        '''
        vertex_polygons=[]
        for i in range(self._nb_vertices):
            vertex_polygons += [[]]
        # vertex_polygons is a list of nb_vertices empty lists

        for p in self._poly_mesh.polygons:
            for nv in p.vertices:
                vertex_polygons[nv] += [p.index]
        return vertex_polygons

    #-------------------------

    def check(self,label=""):
        print("============== check topology : " + label + " =============")
        print("vertex edges : " + str(self._vertex_edges))
        print("vertex neighbors : " + str(self._vertex_neighbors))
        print("vertex polygons : " + str(self._vertex_polygons))
        print("face edges : " + str(self._face_edges))
        print("edge_faces : " + str(self._edge_faces))
        print("border_edges : " + str(self._border_edges))
        print("border_vertices : " + str(self._border_vertices))
        print("vertex_border_edges : " + str(self._vertex_border_edges))
        print("edges touching the border : " + str(self._edges_touching_border))

    #-------------------------

    def getCircularNeighbors(self, lst, e):
        '''
        lst is a list of at least three elements with no duplicates.
        e is an element of lst
        return value : a tuple of integers (i,j,k)
        where : j is the index of e in lst
                i is the index of the previous element
                k is the index of the next element.
        For example for lst=[10,2,13] and e = 2
        the returned value would be (0,1,2)
        But for lst = [10,2,13] and e = 13
        the returned value would be (1,2,0)
        '''
        if len(lst)>=3:
            if e in lst:
                n = len(lst)
                j = lst.index(e)
                i = (j-1)%n
                k = (j+1)%n
                return (i,j,k)
            else:
                print("getCircularNeighbors : e (" + str(e) + ") must be in lst (" + str(lst) + ").")
                return None
        else:
            print("getCircularNeighbors : lst (" + str(lst) + ") must have at least 3 elements")
            return None

    #-------------------------

    def polygonsContaningVertex(self, nv, lst_polygons):
        '''
        nv represents a vertex index of poly_mesh
        lst_polygons is a list of integers representing polygons of poly_mesh
        This function returns a list of integers representing the subset of
        lst_polygons which contains vertex number nv.
        '''
        sublst_poly = []
        for np in lst_polygons:
            if nv in self._poly_mesh.polygons[np].vertices:
                sublst_poly += [np]
        return sublst_poly

    #-------------------------

    def edgeBetweenVertices(self, nv1, nv2, lst_edges):
        '''
        nv1 and nv2 represent vertex indices of poly_mesh. precondition : nv1 != nv2
        lst_edges is a list of integers representing edges of poly_mesh
        This function returns an integer representing the index of an edge
        linking nv1 and nv2. If there is no edge linking both vertices, the
        returned value is -1
        '''
        found_ne=-1
        for ne in lst_edges:
            nv_end1 = self._poly_mesh.edges[ne].vertices[0]
            nv_end2 = self._poly_mesh.edges[ne].vertices[1]
            if nv_end1 == nv1 and nv_end2 == nv2:
                found_ne = ne
            elif nv_end1 == nv2 and nv_end2 == nv1:
                found_ne = ne
        return found_ne


    #-------------------------

    def faceEdges(self, vertex_edges):
        '''
        vertex_edges an list of nv int lists (nv : number of vertices in the mesh)
        which, for each vertex, indicates the edges leading to this vertex.
        For each face, find the edge indices around that face.
        return value : list of np int lists
        (np : number of polygons in the mesh)
        '''
        face_edges = []
        for nf in range(len(self._poly_mesh.polygons)):
            this_face_edges = []
            vertices = self._poly_mesh.polygons[nf].vertices
            for i in range(len(vertices)):
                if i < len(vertices)-1:
                    nv1 = vertices[i]
                    nv2 = vertices[i+1]
                else:
                    nv1 = vertices[i]
                    nv2 = vertices[0]
                edges = vertex_edges[nv1]
                for ne in edges:
                    found=False
                    ind_v1 = self._poly_mesh.edges[ne].vertices[0]
                    ind_v2 = self._poly_mesh.edges[ne].vertices[1]
                    if ind_v1==nv1:
                        ind_other = ind_v2
                    elif ind_v2==nv1:
                        ind_other = ind_v1
                    if ind_other==nv2:
                        found=True
                        this_face_edges += [ne]
            face_edges += [this_face_edges]
        return face_edges

    #-------------------------

    def edgeFaces(self, face_edges):
        '''
        for each edge, find the face indices that are around this edge.
        return value : list of ne int lists
        (np : number of edges in the mesh)
        '''
        edge_faces = []
        for ne in range(self._nb_edges):
            edge_faces += [[]]
        for nf in range(len(self._poly_mesh.polygons)):
            f_edges = face_edges[nf]
            for ne in f_edges:
                edge_faces[ne] += [nf]
        return edge_faces

    #-------------------------

    def borderEdges(self, edge_faces):
        border_edges = []
        for ne in range(self._nb_edges):
            if len(edge_faces[ne])==1:
                border_edges += [ne]
        return border_edges

    #-------------------------

    def sortBorderEdges(self, all_border_edges):
        '''
        all_border_edges : the list of edge indices which are on a border (have one adjacent face)
        But these edges are not necessarily ordered.
        return value : three lists :
        -  sorted_border_edges : a list of nb integer lists
              nb is the the number of borders of the object : a cube has no borders, a plane has one border, a plane with a hole has two borders
              In each list, two successive edges are adjacent. When we loop through the edges of this list, we turn around the border in a
              counter-clockwise manner.
        - sorted_border_vertices : list of nb integer lists
              Same thing for the vertices on each border
        - vertex_border_edges) : list of nv lists of 2 integers or Nones
              Element index i in this list is either None (if vertex index i is not on the border) or it is a list of 2 integers
              which represent the indices of the two border edges adjacent to this vertex
        '''
        sorted_border_vertices = []
        sorted_border_edges = []
        vertex_border_edges = []

        for nv in range(len(self._poly_mesh.vertices)):
            vertex_border_edges += [[None,None]]

        while len(all_border_edges)>0:
            border_vertex_loop = []
            border_edge_loop = []
            n_current_edge = min(all_border_edges)
            loop_end = False
            while not loop_end:
                all_border_edges.remove(n_current_edge)
                border_edge_loop += [n_current_edge]

                ind_nv1 = self._poly_mesh.edges[n_current_edge].vertices[0]
                ind_nv2 = self._poly_mesh.edges[n_current_edge].vertices[1]
                npoly = self._edge_faces[n_current_edge][0]
                face_vertices = list(self._poly_mesh.polygons[npoly].vertices)
                assert ind_nv1 in face_vertices and ind_nv2 in face_vertices
                ind = face_vertices.index(ind_nv1)
                ind_plus_1 = (ind + 1) % len(face_vertices)
                ind_moins_1 = (ind - 1) % len(face_vertices)
                if face_vertices[ind_plus_1] == ind_nv2:
                    first_vertex_on_edge = ind_nv1
                    next_vertex_on_edge = ind_nv2
                else:
                    first_vertex_on_edge = ind_nv2
                    next_vertex_on_edge = ind_nv1

                border_vertex_loop += [ind_nv1, ind_nv2]

                vertex_border_edges[first_vertex_on_edge][1]=n_current_edge
                vertex_border_edges[next_vertex_on_edge][0]=n_current_edge

                found=False
                for ne in all_border_edges:
                    ind_nv1 = self._poly_mesh.edges[ne].vertices[0]
                    ind_nv2 = self._poly_mesh.edges[ne].vertices[1]
                    if ind_nv1==next_vertex_on_edge or ind_nv2==next_vertex_on_edge:
                        found=True
                        n_current_edge = ne
                if not found:
                    loop_end=True
            sorted_border_edges += [border_edge_loop]
            sorted_border_vertices += [list(set(border_vertex_loop))]
        return (sorted_border_edges, sorted_border_vertices, vertex_border_edges)

    #-------------------------

    def flattenList(self, lst):
        '''
        lst is a list possibly containing sublists, subsublists etc.
        return value : a list without sublists
        The returned list's elements are the union of the elements of the sublists of lst.
        '''
        res = []
        for l in lst:
            if hasattr(type(l),'__len__'):   # if l is a sequence
                res += self.flattenList(l)
            else:
                res += [l]
        return res

    #-------------------------

    def edgesTouchingBorder(self, vertex_border_edges):
        '''
        vertex border edges :
        '''
        edges_touching_border = []
        for nv in range(len(self._poly_mesh.vertices)):
            if vertex_border_edges[nv]!=[None,None]:
                edges_touching_border += self._vertex_edges[nv]

        return list(set(edges_touching_border))

    #-------------------------

    def isABorderVertex(self, nv):
        return self._vertex_border_edges[nv] != [None,None]

    #-------------------------

    def getFirstNeighbors(self, nv, lst_polygons, lst_neighbors, lst_edges, first_polygon=-1):
        '''
        nv is an integer which represents the index of a polygon vertex in poly_mesh
        lst_polygons is the list of the polygon indices containing vertex number nv
        lst_neighbors is the list of vertex indices linked to vertex nv by an edge
        lst_edges is the list of edges which meet at edge nv
        return value : (np, nv1, nv2, ne1, ne2)
        where :
        - np is the index of one of the polygons containing nv
        - nv1 and nv2 represent two neighboring vertices of nv in polygon np
            ordered : nv2-nv-nv1
        - ne1 and ne2 are the indices of the edges that link nv-nv1 and nv-nv2
        '''
        if first_polygon!=-1:  np = first_polygon
        else:                  np = lst_polygons[0]

        if np not in lst_polygons:
            print("getFirst : nv=", nv, " first_polygon=",first_polygon, "lpoly_nv=",lst_polygons)

        assert np in lst_polygons
        np_vertices = list(self._poly_mesh.polygons[np].vertices)
        (i,j,k) = self.getCircularNeighbors(np_vertices,nv)
        nv1 = np_vertices[k]
        nv2 = np_vertices[i]
        assert np_vertices[j]==nv
        ne1 = self.edgeBetweenVertices(nv, nv1, lst_edges)
        ne2 = self.edgeBetweenVertices(nv, nv2, lst_edges)
        return (np, nv1, nv2, ne1, ne2)

    #-------------------------

    def getNextNeighbors(self, np, nv, nv1, lst_polygons, lst_neighbors, lst_edges):
        '''
        np is an integer representing a polygon in poly_mesh containing vertices nv and nv1 (nv1 comes before nv)
        nv and nv1 are integers which represent the index of two polygon vertices in poly_mesh
        lst_polygons is the list of the polygon indices containing vertex number nv
        lst_neighbors is the list of vertex indices linked to vertex nv by an edge
        lst_edges is the list of edges which meet at edge nv
        return value : (np2, nv2, ne2)
        where :
        - np2 is the index of the polygon (in lst_polygons) containing nv and nv1 and where nv1 comes after nv
        - nv2 is the index of the vertex coming after nv on polygon np
        - ne2 is the edge corresponding to nv-nv2.
        Precondition : if nv is a border vertex, nv1 should not be the last neighbor (counter-clockwise order) of nv
        '''
        lst_nv1 = self.polygonsContaningVertex(nv1, lst_polygons)
        assert np in lst_nv1
        if len(lst_nv1)!=2:
            print("nv=",nv, "nv1=",nv1, "np=",np)

        # assert len(lst_nv1)==2  # This may not be true for border vertices
        lst_nv1.remove(np)
        np2 = lst_nv1[0]
        np_vertices = list(self._poly_mesh.polygons[np2].vertices)
        (i,j,k) = self.getCircularNeighbors(np_vertices,nv)
        assert np_vertices[k]==nv1
        assert np_vertices[j]==nv
        nv2 = np_vertices[i]
        ne2 = self.edgeBetweenVertices(nv, nv2, lst_edges)
        return (np2,nv2,ne2)

    #-------------------------

    def sort(self):
        '''
        lst_polygons : a list of n integer lists representing, for each vertex, the list of poylgons containing that vertex
        lst_neighbors : a list of n integer lists representing, for each vertex, the list of neighboring vertices of that vertex
        lst_edges : a list of n integer lists representing, for each vertex, the list of edges meeting at that vertex.
        return value : (lst_polygons, lst_neighbors, lst_edges)
        where the elements of all lists have been sorted in the counter-clockwise order.
        '''
        for nv in range(len(self._poly_mesh.vertices)):

            lpoly_nv = self._vertex_polygons[nv]
            lneighb_nv = self._vertex_neighbors[nv]
            ledge_nv = self._vertex_edges[nv]

            lpoly_sorted = []
            lneighb_sorted = []
            ledge_sorted = []

            # if nv is a border vertex, we have to choose the first polygon (counter-clockwise)
            # otherwise, we can choose any polygon
            is_a_border_vertex = self.isABorderVertex(nv)
            if is_a_border_vertex:
                ne_first_border_edge = self._vertex_border_edges[nv][1]
                ind_v1 = self._poly_mesh.edges[ne_first_border_edge].vertices[0]
                ind_v2 = self._poly_mesh.edges[ne_first_border_edge].vertices[1]
                if ind_v1==nv:    nv_first_neighbor =ind_v2
                else:             nv_first_neighbor =ind_v1

                ne_last_border_edge = self._vertex_border_edges[nv][0]
                ind_v1 = self._poly_mesh.edges[ne_last_border_edge].vertices[0]
                ind_v2 = self._poly_mesh.edges[ne_last_border_edge].vertices[1]
                if ind_v1==nv:    nv_last_neighbor =ind_v2
                else:             nv_last_neighbor =ind_v1

                np = self._edge_faces[ne_first_border_edge][0]
                (np1, nv0, nv1, ne0, ne1) = self.getFirstNeighbors(nv, lpoly_nv, lneighb_nv, ledge_nv, np)
            else:
                (np1, nv0, nv1, ne0, ne1) = self.getFirstNeighbors(nv, lpoly_nv, lneighb_nv, ledge_nv)

            lpoly_sorted += [np1]
            lneighb_sorted += [nv0]
            lneighb_sorted += [nv1]
            ledge_sorted += [ne0]
            ledge_sorted += [ne1]

            nv2 = nv1
            while not (not is_a_border_vertex and nv2==nv0 or is_a_border_vertex and nv2 == nv_last_neighbor):
                (np2,nv2,ne2) = self.getNextNeighbors(np1, nv, nv1, lpoly_nv, lneighb_nv, ledge_nv)
                lpoly_sorted+=[np2]

                lneighb_sorted+=[nv2]
                ledge_sorted+=[ne2]
                np1=np2
                nv1=nv2

                self._vertex_polygons[nv]=lpoly_sorted
                self._vertex_neighbors[nv]=lneighb_sorted
                self._vertex_edges[nv]=ledge_sorted


    #-------------------------

'''
objet=bpy.data.objects["Plane"]
topo_plane = Topology(objet.data)
topo_plane.check("topo_plane")

lst = [1,2,3,[4,5,[6,7]],8,9,[10,11]]
print(topo_plane.flattenList(lst))
'''
