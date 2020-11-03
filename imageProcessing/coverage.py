import numpy as np
import cv2


def _calc_sat(r, g, b):
    """Formula for calculating saturation"""
    # np.seterr('raise')
    try:
        if (r < g < b):
            return 1 - (3 * r) / (r + b + g)
        elif (g < b):
            return 1 - (3 * g) / (r + b + g)
        elif (b < g):
            return 1 - (3 * b) / (r + b + g)
    except:
        return 1e9


# Vectorize the functions above so that we can use numpy to easily apply the functions
# to all pixels
v_sat = np.vectorize(_calc_sat)


def cloud_recognition(img, SAT_THRESHOLD=0.08):
    """Converts all pixel RGBs in an image to 0 (not cloud) or 255 (cloud)"""
    # OpenCV opens images as GBR. We need to change it to RGB, convert it to a numpy array
    # and then normalize all the values
    img = np.asarray(img).astype(np.double)
    img /= 255

    # Use the vectorized functions above and apply to every element of the matrix
    sat = v_sat(img[:, :, 2], img[:, :, 1], img[:, :, 0])

    # If pixel saturation is greater than the threshold set opacity to 0, else 1
    opacity = np.where(sat > SAT_THRESHOLD, 0, 1)

    # Add opacity to fourth "column" of img numpy object
    output = np.dstack((img, opacity))

    output *= 255

    # Return the image in the same format, in which it was inputted
    return output
