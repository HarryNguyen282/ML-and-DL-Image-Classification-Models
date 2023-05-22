*** This file has been adapted from the original CUB_200_2011 dataset for the sole purpose of the CVIA assignment. Do not distribute. ***

===========================================
The Caltech-UCSD Birds-200-2011 Dataset
===========================================

For more information about the dataset, visit the project website:

  http://www.vision.caltech.edu/visipedia

If you use the dataset in a publication, please cite the dataset in
the style described on the dataset website (see url above).

Directory Information
---------------------

- images/
    The images organized in subdirectories based on species. See 
    IMAGES AND CLASS LABELS section below for more info.



=========================
IMAGES AND CLASS LABELS:
=========================
Images are contained in the directory images/, with 200 subdirectories (one for each bird species)

------- List of image files (images.txt) ------
The list of image file names is contained in the file images.txt, with each line corresponding to one image:

<image_id> <image_name>
------------------------------------------


------- List of class names (classes.txt) ------
The list of class names (bird species) is contained in the file classes.txt, with each line corresponding to one class:

<class_id> <class_name>
--------------------------------------------


------- Image class labels (image_class_labels.txt) ------
The ground truth class labels (bird species labels) for each image are contained in the file image_class_labels.txt, with each line corresponding to one image:

<image_id> <class_id>

where <image_id> and <class_id> correspond to the IDs in images.txt and classes.txt, respectively.
---------------------------------------------------------





=========================
BOUNDING BOXES:
=========================

Each image contains a single bounding box label.  Bounding box labels are contained in the file bounding_boxes.txt, with each line corresponding to one image:

<image_id> <x> <y> <width> <height>

where <image_id> corresponds to the ID in images.txt, and <x>, <y>, <width>, and <height> are all measured in pixels
