import numpy as np
import cv2

# CONSTANTS
ARROW_STEP = 16
# parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next
# layer is twice smaller than the previous one.
pyr_scale = 0.5
# number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
levels = 3
# averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection,
# but yield more blurred motion field.
winsize = 15
# number of iterations the algorithm does at each pyramid level.
iterations = 3
# size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated
# with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
poly_n = 5
# tandard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can
# set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
poly_sigma = 1.2
flags = 0

# Flags
DRAW_COLORS = False
DRAW_ARROWS = True


def draw_colors(optflow):
    # Prepare HSV
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # Populate HSV image with optical flow values
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Return the optical flow in BGR, as it was inputted
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(frame, flow, step=ARROW_STEP):
    """Draw vector field based on image and displacement vectors"""
    h, w = frame.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    nonzero_lines = []
    x_f = []
    x_i = []

    for (x1, y1), (x2, y2) in lines:
        if (x1 != x2) and (y1 != y2):
            #nonzero_lines.append(np.array( [[x1, y1], [x2, y2]] ))
            cv2.arrowedLine(frame, (x1, y1), (x2, y2),
                            (0, 255, 0, 255), thickness=1, tipLength=0.3)
            x_i.append((x1, y1))
            x_f.append((x2, y2))

    #cv2.polylines(frame, nonzero_lines, False, (0, 255, 0, 255))
    cv2.imwrite('cloud_optical_flow.png', frame)
    return frame, x_i, x_f


def calculate_opt_dense(frame1, frame2):
    """Calculates displacement vectors for each pixel in two frames"""
    # Convert the images to Grayscale
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("greyed out", prev)
    cv2.waitKey(0)
    # Calculate the optical flow
    return cv2.calcOpticalFlowFarneback(prev, next, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
