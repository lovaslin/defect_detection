###################################################################################################
## Usage example for the basic features of the defect_detection API.                             ##
## In order to run it, you need download the example data using the get_source_images.sh script. ##
###################################################################################################


import defect_detection as dd
import numpy as np
import os


## Dataset generation

# Path to the directory where the source images are stored
spath = "source/"

# The image files to be used for the dataset generation must be stored in a text file
# It is possible to different list of images to generate a dataset.
# Path to the file containing the source images to be used.
flist = ["example_list1.txt", "example_list2.txt"]

# Optionally, a fixed initial cropping can be applied on every source images before data augmentation.
# If there are more than one list of images, one initial cropping per list is needed.
crop = [
    [900, -1000, 1400, -1280],  # for source 1
    [1600, -1750, 1400, -1280],  # for source 2
]

# Path to the ouput directory where the dataset will be stored
opath = "dataset/"
if not os.path.exists(opath):
    os.mkdir(opath, 0o755)

# Call the generate_dataset function
dd.generate_dataset(
    name="example_data",
    flist=flist,
    spath=spath,
    opath=opath,
    crop=crop,
    Naug=2,
    offset=25,
    da=[-0.2, 0.2],
    db=[-0.2, 0.2],
    dw=15,
    Nseg=8,
    size=512,
    opref="img",
    Ncore=4,
    seed=42,
    shuf=True,
)

# The following will be generated under the output path (opath):
#    - A folder named "example_data" containint the generated images
#    - A file named "example_data.txt" containing all the arguments used to generate the dataset


## New model training

# The name given to the new model, together with the path where is should be saved
model = "example_AE"
model_dir = "model/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir, 0o755)

# The path to the configuration file defining the structure of the model
# (number of layers, kernel size, number of features, ...)
# The detail of the format is specified in the documentation
config = "example_conf.txt"

# Name of the dataset to be used for the training, together with the path where it can be found
# Here we use the dataset generated previously
data = "example_data"
data_dir = "dataset/"

# Optionnaly, it is possible to define custom noise patterns for the training of the denoising Auto-Encoder model
patch_dir = "example_patterns/"

# Call the deepAE_train function create and train the new model
loss_train, loss_val = dd.deepAE_train(
    model,
    config,
    data,
    data_dir=data_dir,
    model_dir=model_dir,
    epochs=2,
    ntrain=5,
    npatch=2,
    patch_dir=patch_dir,
    patch_size=[10, 100],
    seed=42,
)

# A folder named "example_AE" will be generated under the model path (model_dir)
# This folder will contain the model configuration as well as the trained parameter


## Apply model

# Load a trained model from disk
# Here we use the model trained previously
ae = dd.deepAE_load(model_dir + model + "/save/")

# Don't forget to put it in evaluation mode
ae.eval()

# Load a batch of images to apply the model on.
# This is done using a list of file path/names to be loaded
# In this example, we take the 8 first images of the dataset that was generated previously
apply_list = [
    "dataset/example_data/img_1_x1y1.jpg",
    "dataset/example_data/img_1_x1y2.jpg",
    "dataset/example_data/img_1_x1y3.jpg",
    "dataset/example_data/img_1_x1y4.jpg",
    "dataset/example_data/img_1_x1y5.jpg",
    "dataset/example_data/img_1_x1y6.jpg",
    "dataset/example_data/img_1_x1y7.jpg",
    "dataset/example_data/img_1_x1y8.jpg",
]
print("List of files for application :")
print(apply_list)
batch = dd.load_batch(apply_list)

# Get the loss values and model outputs from the input batch
reco, loss = ae.batch_apply(batch)
print("Loss shape :", loss.shape)

# The input batch which is a torch.Tensor type can be reconverted to a numpy array
batch = dd.get_array(batch)

# The per pixel anomaly score (based on reconstruction error) can be computed from the obtained output
# This example is for a single emap (using only first input/output pair)
emap = dd.emap(batch[0], reco[0])

# Selection threshold to be applied on the anomaly score map (emap)
# For this example, we take a value proportionnal the 99th percentile of the flattened emap
sel_th = np.percentile(emap.flatten(), 99) * 1.2

# Parameter to be given to the DBSCAN algorithm used to cluster the selected pixels (given in a dict)
# Isolated pixels that don't belong to a cluster will be dropped
dbs_param = {"eps": 4, "min_samples": 7}

# Minimum number of pixel per cluster
# Smaller clusters will be dropped
pix_th = 15

# Call the get_pixel function to cluster anmomalous pixels and select relevant defect candidates
pix = dd.get_pixels(emap, sel_th, dbs_param, pix_th)
print("Number of pixelsin selexcted clusters :", pix.shape[0])

# The final list of selected pixels (pix) can be used to identify potential anomalies (defects) in the input image
