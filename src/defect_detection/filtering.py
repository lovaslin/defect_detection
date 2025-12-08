import numpy as np
from sklearn.cluster import DBSCAN


# Function that takes in an input image and configuration and returns the selected pixels
def get_pixels(emap, th, dbs_param, pix_th):
    """
    Perform a selection of anomalous pixels based on the error map associated to the an image.
    The basic selection is done based on the given threshold value.
    The obtained list is then denoised using the DBSCAN clustering algorithm.

    Arguments :
        emap : the error map associated to the image
        th : the initial selection threshold
        dbs_param : dict of additional arguments to be passed to the DBSCAN instance
        pix_th : The minimum number of pixels required per clusters (None means no requirement)

    Returns :
        pix : clean list of selected pixels
    """

    # Get raw selection (set non selected to 0)
    emap[emap <= th] = 0

    # Get pixel coordinates
    pix = np.argwhere(emap)
    del emap

    # Check for empty selection
    if pix.shape[0] == 0:
        return pix

    # Initialise DBSCAN with parameters and run clustering
    dbs = DBSCAN(**dbs_param)
    fil = dbs.fit_predict(pix)

    # Add the pix_th criteria
    if pix_th:
        # Remove small cluster and remove noisy labels
        lab, count = np.unique(fil, return_counts=True)
        lab = lab[1:]
        count = count[1:]
        pix = pix[np.isin(fil, lab[count >= pix_th])]
        del lab
        del count
    else:
        # Only filter out noisy labels
        pix = pix[fil > -1]

    # Return the filtered pixel list
    return pix
