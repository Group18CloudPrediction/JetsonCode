import os

import cv2
import numpy as np
import tzlocal
from _datetime import datetime
from matplotlib import pyplot as plt
from pysolar.solar import get_altitude, get_azimuth
from scipy import signal


# also ignore leap second warnings
# the lat, long will be of the camera
def get_sun(lat, long, date):
    """Retrieves sun azimuth and altitude"""
    altitude = get_altitude(lat, long, date)
    azimuth = get_azimuth(lat, long, date)
    return azimuth, altitude


def draw_sun_circle(sun_radius, sun_center, img):
    """Draws circle on img, centered on sun_center (x, y), with radius sun_radius"""
    # draw red circle around the center point
    cv2.circle(img, sun_center, sun_radius, (0, 0, 255), -1, 8, 0)
    # identify all red pixels and extract their coordinates
    points = np.where((img == [0, 0, 255]).all(axis=2))

    sun_pixels = list(zip(points[1][:], points[0][:]))

    return sun_pixels


def mask_sun_pixel(img, sun_radius, SUN_THRESHOLD=2.9375, FILTER_SIZE=25):
    # Locate the brightest pixel in the image (i.e. the sun) and cover it up so it doesn't
    # mess with our coverage code
    img = np.asarray(img).astype(np.double)
    img /= 255

    intensity = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]

    # To eliminate noise and find the center of the sun, we're going to do a mean convolution
    mean_matrix = np.full(shape=(FILTER_SIZE, FILTER_SIZE),
                          fill_value=1/FILTER_SIZE**2)
    convolved_intensity = signal.convolve2d(
        intensity, mean_matrix, mode='full', boundary='fill', fillvalue=0)

    # locate the brightest pixel in the image (aka the pixel with the highest intensity value)
    max_intensity = np.amax(convolved_intensity)

    # If the sun is in the image, the max_intensity should be greater than this threshold
    # If it's not, the that means that the sun is probably covered by a cloud (or not in frame)
    if max_intensity >= SUN_THRESHOLD:
        brightest = np.where(convolved_intensity == max_intensity)
        l = int(len(brightest[0]) / 2)
        sun_center = (brightest[1][l], brightest[0][l])
        sun_pixels = draw_sun_circle(sun_radius, sun_center, img)

        return sun_center, sun_pixels

    return (None, None), None


def mask_sun_pysolar(lat, long, sun_radius):
    """Returns point at center of sun and pixel mask for sun."""
    # date = tzlocal.get_localzone().localize(datetime.now())
    date = datetime(2019, 10, 19, 15, 42, 00, tzinfo=tzlocal.get_localzone())
    # frame = cv2.resize(frame1, (640, 480))

    # 2.) Find sun using (azimuth, altitude)
    azimuth, altitude = get_sun(lat, long, date)
    # print(azimuth, altitude)

    # *************** Creating the polar grid ****************
    # To convert degrees to radian for plotting
    rad = np.pi / 180
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='polar')
    ax1.grid(True)
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.grid(linewidth=1)
    ax1.set_ylim(0, 90)
    ax1.set_yticks(np.arange(0, 90, 10))
    yLabel = ['90', '', '', '60', '', '', '30', '', '']
    ax1.set_yticklabels(yLabel)

    # *************** PLOTTING DIRECTLY ONTO POLAR GRID WITH AZIMUTH, ALTITUDE input ***************
    # original: single center point
    plt.polar((azimuth)*rad, 90-altitude, 'ro', markersize=1)
    plt.savefig('sunPos.png')

    # load sunPos image to create mask
    sun_image = cv2.imread('sunPos.png')
    # mask = np.zeros_like(sun_image)
    # mask[np.where((sun_image == [0, 0, 255]).all(axis=2))] = [0, 0, 255]

    point = np.where((sun_image == [0, 0, 255]).all(axis=2))

    sun_center = (point[1][0], point[0][0])

    # Mask the sun in the image, store the masked area as 'sunPixels'
    sun_pixels = draw_sun_circle()

    # TODO: check if the sun is not out then:
    # return None, None

    # return 'sun_center' to motion estimation and coverage
    return sun_center, sun_pixels
