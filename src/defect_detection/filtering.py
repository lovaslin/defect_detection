import numpy as np
from sklearn.cluster import DBSCAN


# Function that takes in an input image and configuration and returns the selected pixels
def get_pixels(emap, th, dbs_param, pix_th):
    """
    Perform a selection of anomalous pixels based on the error map associated to the an image.
    The basic selection is done based on the given threshold value.
    The obtained list is then denoised using the DBSCAN clustering algorithm.

    Arguments :
        emap : (numpy.array)
            The error map associated to the image.
            It should have the shape (img_x, img_y).

        th : (float)
            The initial selection threshold to be applied on per pixel anomaly score.
            Only the pixels which score is higher will be selected.

        dbs_param : (dict)
            Additional arguments to be passed to the DBSCAN instance given as a dict.
            It should be of the from {'arg1': val1, ..., 'argN': valN}.

        pix_th : (int or None)
            The minimum number of pixels required per clusters.
            Anomalous pixel clusters with fewer pixels will be ignored.
            If None, no minimal cluster size requirement is applied.

    Returns :
        pix : (numpy.array)
            The list of selected pixels after filtering has been applied.
            It is geven with the format [[pix_x1, pix_y1], ..., [pix_xN, pix_yN]].
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
