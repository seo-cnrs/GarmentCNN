# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:54:36 2023

@author: Boyang YU
"""

import bpy
import bmesh
from collections import OrderedDict
import numpy as np
import os
import glob
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] 

path=argv[0]
target=argv[1]

# create the folders required for MeshCNN
try:
    os.makedirs(os.path.join(target, 'train'))
    os.makedirs(os.path.join(target, 'vseg'))
    os.makedirs(os.path.join(target, 'edges'))
    os.makedirs(os.path.join(target, 'seg'))
    os.makedirs(os.path.join(target, 'sseg'))
except FileExistsError:
    # directory already exists
    pass

#TODO: change to your own path
root="/home/recherche/Documents/Mouad/MouadDataSet/MouadCodes/MouadHARROUZcodes/CreateGroundTruth/" 

# needs to be absolute for Blende to find them
path=root+path[2:]
target=root+target[2:]
print(path)
print(target)

# remove everything
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete(use_global=False)

for item in bpy.data.meshes:
    bpy.data.meshes.remove(item)


for filename in glob.glob(os.path.join(path, 'train/*.obj')):
    print("name: ",filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    print(basename)
    
    
    # load SEG file
    seg_path = os.path.join(path, basename + '_segmentation.txt')
    print(seg_path)
    #label_name = os.path.join(path,'tee_0ADQ2LDQJA_sim_segmentation.txt')   # Replace with the path to your text file
    #print(label_name)
    
    # Read the segmentaion file 
    with open(seg_path, 'r') as file:
        lines = file.readlines()
        # Removing newline characters from each line and storing in a list
        vertex_seg_before_decimation = [line.strip() for line in lines]
        
    order=list(OrderedDict.fromkeys(vertex_seg_before_decimation)) #TODO: should this be set once before everything
    ints=[order.index(e) for e in vertex_seg_before_decimation]
        
    searched_value = "stitch"
    vertex_indices_to_preserve = [index for index, value in enumerate(vertex_seg_before_decimation) if value == searched_value]
    
    print(len(vertex_indices_to_preserve))
    print("high reso ",len(ints))
    
    # Load OBJ file
    #obj_path = os.path.join(path,'tee_0ADQ2LDQJA_sim_3D.obj') 
    
    bpy.ops.import_scene.obj(filepath=filename, split_mode='OFF')
    obj = bpy.context.selected_objects[0] ####<--Fix
    print('Imported name: ', obj.name)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(state=True)
    
    # Get the active object
    print("hello")
    obj = bpy.context.active_object
    mesh = obj.data
    print(obj.name)
    print("***********************************")
    
    print("======== Get Segmentation Label =======")
    layer = mesh.vertex_layers_int.new(name="att")
    for i, label in enumerate(ints):
        layer.data[i].value = label
    
    print("======== Decimate with boundary preservation =======")
    # Create a new vertex group and add vertices to it
    vertex_group_name = "topreserve"
    vertex_group = obj.vertex_groups.get(vertex_group_name)
    if vertex_group is None:
        vertex_group = obj.vertex_groups.new(name=vertex_group_name)
    
    # Add all vertex indices to the vertex group
    vertex_group.add(vertex_indices_to_preserve, 1.0, 'REPLACE')
    
    # Add Decimation modifier
    decimate_modifier = obj.modifiers.new(name="Decimation", type='DECIMATE')
    decimate_modifier.ratio=0.2                 #TODO: this up to your choice 
    #decimate_modifier.use_collapse_triangulate=True
    decimate_modifier.vertex_group = vertex_group_name
    decimate_modifier.invert_vertex_group=True
    
    # Apply the Decimation modifier
    bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)
    
    
    print("===== Update Segmentation Label =====")
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh) 
    layer = bm.verts.layers.int["att"]
    bm.verts.ensure_lookup_table()
    
    vsegs=list()  # new per vetex --TODO check if this is right
    for i in range(len(bm.verts)):
        new_seg_label=bm.verts[i][layer]
        vsegs.append(new_seg_label)
    
    print("Low resolution ",len(vsegs))  
    
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    target_path = target+'/train/'+ basename+".obj"
    bpy.ops.export_scene.obj(filepath=target_path, check_existing=True, axis_forward='-Z', axis_up='Y', filter_glob="*.obj", use_selection=False, use_animation=False, use_mesh_modifiers=True, use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True, use_uvs=True, use_materials=True, use_triangles=False, use_nurbs=False, use_vertex_groups=False, use_blen_objects=True, group_by_object=False, group_by_material=False, keep_vertex_order=True, global_scale=1, path_mode='AUTO')
    
    #vseg_path='./{}.vseg'
    vseg_path = target+'/vseg/{}.vseg'
    np.savetxt(vseg_path.format(bpy.context.active_object.name), vsegs, delimiter=',',fmt='%s')
     
    print("alright here")
    
    #bpy.ops.object.select_all(action='SELECT')
    #bpy.ops.object.delete()
    
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete(use_global=False)
    
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    
print("Done")






