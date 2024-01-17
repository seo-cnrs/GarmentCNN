# GarmentCNN

# Dataset Creation

This guide outlines the steps for preparing a dataset for GarmentCNN.

## Step 1: Copy and Clean up Mesh

Copy and paste mesh files in obj and segmentation files in '.txt'(per vertex label) from Maria's dataset(https://zenodo.org/records/5267549).

```bash
python extract_objseg.py InputFolder OutputFolder NumberOfInstancesToGet
# Example:
python extract_objseg.py ./tee_test ./tee_samples 2
```

## Step 2: (Optional)Decimate Mesh 
 
Use Blender as the backend, it's enough to execute the command follows which specifies the target directory, note that this operation decimates the meshes and thus new per vertex lable files are generated and saved in the 'vseg' subfolder, and we need to add the "_simplified" as suffix to the target folder.

```bash
blender --background --python decimation.py -- ./tee_samples ./tee_samples_final
```

## Step 3: Generate 'seg' files
This fills subfolders 'seg'(per edge label), and edges. 
NB: .edges file is a side product to save oredered edges, which are expressed with the two vertices it connects. But not required for GarmentCNN.

```bash
python create_seg.py Folder 
# Example:
python create_seg.py ./tee_samples
```
## Step 4: Generate 'sseg' file
This fills 'sseg' subfolder (weights for edges).

```bash
python create_sseg.py Folder 
# Example:
python create_sseg.py ./tee_samples
```

