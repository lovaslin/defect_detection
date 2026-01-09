########################################################################################
## Define a generic preprocessing function for input images.                          ##
## Includes data augmentation and image spliting.                                     ##
## This function can be used to generate a training dataset for a new model.          ##
########################################################################################


import numpy as np
import cv2 as cv
import os
from pathlib import Path
import concurrent.futures as thd


def generate_dataset(
    name,
    flist,
    spath,
    opath,
    crop=None,
    Naug=5,
    offset=25,
    da=0.2,
    db=0.2,
    dw=15,
    Nseg=8,
    size=512,
    opref="img",
    Ncore=4,
    seed=666,
    shuf=False,
):
    """
    Generates a dataset from a list of source images.
    The images ondergo data augmentation and spliting into square tiles.
    A configuration file with the preprocessing arguments is also generated.

    Arguments:
        name : (str)
            Name of the dataset.
            Can be used ad ID when working with different dataset.

        flist : (Path or str)
            File containing the list of source image files to be processed.

        spath : (Path or str)
            Path containing the source images.

        opath : (Path or str)
            Path where the dataset is generated

        crop : (iterable or None)
            Defines the initial cropping information to be applied to the images.
            In the case there are more than one source directory, cropping mus be provided for all of them.
            The expected format for N source image directories is the following :
                crop = [[xmin_1, xmax_1, ymin_1, ymax_1], ..., [xmin_N, xmax_N, ymin_N, ymax_N]]
            If None, no initial cropping is applied
            Default to None

        Naug : (int)
            Number of source image duplicate generated during augmentation.
            Default to 5

        offset : (int)
            Maximal random cropping offset applied during augmentation (in pixel).
            Default to 25

        da : (float or iterable)
            Amplitude of the contrast variation applied during augmentation.
            If a float value is given, the amplitude will be [-da, +da].
            If a length 2 iterable is give, the amplitude will be [da[0], da[1]].
            Defalt to 0.2


        db : (float or iterable)
            Amplitude of the Luminosity variation applied during augmentation.
            If a float value is given, the amplitude will be [-da, +da].
            If a length 2 iterable is give, the amplitude will be [da[0], da[1]].
            Default to 0.2

        dw : (int)
            Random crop scale variation applied during augmentation (in pixel).
            Default to 15

        Nseg : (int)
            Number of vertical and horizontal split applied for image splitting.
            Each augmented image will then give Nseg*Nseg tiles.
            Default to 8

        size : (int)
            Size in pixel of the generated image tiles.
            The source image is resized to fit the required size before splitting.
            Default to 512

        opref : (str)
            Prefix used for naming the generated images.
            Default to "img"


        Ncore : (int)
            Number of CPU core to use for parallelizing the image generation.
            Default to 4

        seed : (int)
            The seed to be used for numpy random generator (ensure reproducibility).
            Default to 666
    """

    # Initialize seed
    np.random.seed(seed)

    # Check if da and db are given as floats
    if type(da) is float:
        da = [-da, da]
    if type(db) is float:
        db = [-db, db]

    # Check if flist is given as a string
    if type(flist) is str:
        flist = [flist]

    # Check also the shape of crop (if given)
    if crop:
        if np.array(crop).ndim == 1:
            crop = [crop]

    # Save preprocessing option in a file
    param = {
        "name": name,
        "flist": flist,
        "spath": spath,
        "opath": opath,
        "crop": crop,
        "Naug": Naug,
        "offset": offset,
        "da": da,
        "db": db,
        "dw": dw,
        "Nseg": Nseg,
        "size": size,
        "opref": opref,
        "Ncore": Ncore,
        "seed": seed,
        "shuf": shuf,
    }
    with Path.open(Path(param["opath"] + param["name"] + ".txt"), "w") as f:
        print(param, file=f)

    print("Preprocessing parameters :")
    for k, i in param.items():
        print(f"\t{k} : {i}")

    # Check if output path directory exists
    opath = param["opath"] + param["name"] + "/"
    if not os.path.exists(opath):
        os.mkdir(opath, 0o755)

    # Read source file names from the list(s)
    fl = []
    list_idx = []
    list_len = []
    for i, fname in enumerate(param["flist"]):
        # Open and read file i
        with Path.open(Path(fname)) as f:
            count = 0
            for li in f.readlines():
                if li[0] == "#":
                    continue
                else:
                    fl.append(li.strip())
                    list_idx.append(i)
                    count = count + 1
            list_len.append(count)
    print(f"\n{len(fl)} images to process from {len(list_len)} source(s)")

    # Load temporarly the first image(s) of the list(s) to get shapes
    sxi, syi = [], []
    list_len = np.array(list_len)
    for i in range(len(list_len)):
        # Get index of the first imge in list i
        if i == 0:
            idx = 0
        else:
            idx = np.sum(list_len[:i])

        img = cv.imread(param["spath"] + fl[idx])
        sxi.append(img.shape[0])
        syi.append(img.shape[1])

    # Apply shuffling if required
    if param["shuf"]:
        # Convert to numpy array (for easier shuffling)
        fl = np.array(fl)
        list_idx = np.array(list_idx)

        # Get randomly shuffled indices
        shuf_id = np.arange(0, fl.size, dtype=int)
        shuf_id = np.random.shuffle(shuf_id)

        # Apply shufling using shuffled indices
        fl = fl[shuf_id][0]
        list_idx = list_idx[shuf_id][0]
        del shuf_id

    # Comput the x/y size of the augmented images
    xmin, xmax, ymin, ymax = [], [], [], []
    for i in range(len(list_len)):
        if param["crop"]:
            # Compute margin accounting for croping and offset
            xmin.append(max(param["crop"][i][0], param["offset"] + param["dw"]))
            xmax.append(
                min(
                    sxi[i] + param["crop"][i][1], sxi[i] - param["offset"] - param["dw"]
                )
            )
            ymin.append(max(param["crop"][i][2], param["offset"] + param["dw"]))
            ymax.append(
                min(
                    syi[i] + param["crop"][i][3], syi[i] - param["offset"] - param["dw"]
                )
            )
        else:
            # Compute margin accounting for offset only
            xmin.append(param["offset"] + param["dw"])
            xmax.append(sxi[i] - param["offset"] - param["dw"])
            ymin.append(param["offset"] + param["dw"])
            ymax.append(syi[i] - param["offset"] - param["dw"])
        print(
            f"Image shape before segmentation for source {i + 1} : ({xmax[i] - xmin[i]}, {ymax[i] - ymin[i]}) (+-{2*param['dw']})"
        )

    # Function to process one image (augmentation+segmentation)
    # To be parallelized (So must be global).
    global process_img

    def process_img(i):
        np.random.seed(seed * i)

        # Compute segment edges
        sx = np.linspace(0, param["size"] * param["Nseg"], param["Nseg"] + 1, dtype=int)
        sy = np.linspace(0, param["size"] * param["Nseg"], param["Nseg"] + 1, dtype=int)

        # Augmentation loop
        for a in range(param["Naug"]):
            # Set random offset for position, brightness and contrast
            offset = np.random.randint(-param["offset"], param["offset"])
            da = np.random.uniform(param["da"][0], param["da"][1])
            db = np.random.uniform(param["db"][0], param["db"][1])
            dw = np.random.randint(0, param["dw"])

            # Read image in the croped/translated range
            img = cv.imread(param["spath"] + fl[i])[
                xmin[list_idx[i]] + offset - dw : xmax[list_idx[i]] + offset + dw,
                ymin[list_idx[i]] + offset - dw : ymax[list_idx[i]] + offset + dw,
            ]

            # Resize image so that it fits with segmentation
            img = cv.resize(
                img, (param["size"] * param["Nseg"], param["size"] * param["Nseg"])
            )

            # Apply brightness transformation
            img = cv.convertScaleAbs(img, 1 + da, 1 + db)

            # Segmentation loop
            iid = param["Naug"] * i + a
            for si in range(sx.size - 1):
                for sj in range(sy.size - 1):
                    cv.imwrite(
                        opath + param["opref"] + f"_{iid + 1}_x{si+1}y{sj+1}.jpg",
                        img[sx[si] : sx[si + 1], sy[sj] : sy[sj + 1]],
                    )

        # Free memmory
        del img
        del sx
        del sy
        print(f"\timage{i + 1} OK")

        return

    # Inintialize process pool for parallelization
    print("\n####PROCESSING IMAGES####")
    with thd.ProcessPoolExecutor(max_workers=param["Ncore"]) as exe:
        # Image loop
        for i in range(len(fl)):
            # Submit image process to the pool
            exe.submit(process_img, i)

    print("DONE")

    return
