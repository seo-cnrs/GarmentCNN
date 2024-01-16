# GarmentCNN

Steps for creating Dataset of MeshCNN: 

1- Copy and paste mesh files in obj and segumentation files in txt(per vertex label) from Maria's dataset using extract_objseg.py by command follows: 
	- python extract_objseg.py InputFolder OutputFolder NumberOfInstancesToGet
	- example : python extract_objseg.py ./tee_test ./tee_samples 2
	 where Input: ./tee_test is organised as any garment category in Maria's dataset(to be replaced by user), 
	Output ./tee_samples has train folder(cleaned original mesh files), and segmentation files. 

2-(Optional) Depends on your choice to do simplification of mesh: 
	- Using Blender as backend, it's enough to execute command follows which specifies the target directory, note that this operation decimates the
	meshes and thus new per vertex lable files are generated and saved in the vseg subfolder, and we need to add the _simplified as suffix
	for create segs later based on the decimated new meshes:
		- blender --background --python decimation.py ./tee_samples ./tee_samples_simplified

3- Generate seg files using command follwows:
	- python create_seg.py Folder 
	- example: python create_seg.py ./tee_samples
	This fills subfolders seg(per edge label), and edges. 
	NB: .edges file is a side product to save oredered edges, which are expressed with the two vertices it connects. But not required for MeshCNN.

4- Generate Sseg files using command follwows: 
	- python create_sseg.py Folder 
	- example: python create_sseg.py ./tee_samples
	This fills sseg subfolder (weights for edges)
