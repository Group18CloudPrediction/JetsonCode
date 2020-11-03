import cv2
import numpy as np


def create_fisheye_mask(prev, next, mask_radius=3):
    """Creates a mask for the fisheye lens"""
    h, w, ch = prev.shape
    center = (int(w / 2), int(h / 2))
    radius = int(w / mask_radius)

    mask = np.zeros(shape=prev.shape, dtype="uint8")
    cv2.circle(mask, center, radius, (255, 255, 255), -1, 8, 0)

    maskedPrev = np.zeros_like(prev)
    maskedPrev[mask > 0] = prev[mask > 0]

    maskedNext = np.zeros_like(next)
    maskedNext[mask > 0] = next[mask > 0]

    return maskedPrev, maskedNext


def image_crop(img, mask_radius_ratio=3.5):
    """Crops the inputted image according to mask radius ratio"""
    w = img.shape[0]
    h = img.shape[1]
    s = w / mask_radius_ratio

    top_edge = int(h/2-s)
    bottom_edge = int(h/2 + s)

    left_edge = int(w/2-s)
    right_edge = int(w/2 + s)

    croppedImg = img[left_edge:right_edge,  top_edge:bottom_edge, :]

    return croppedImg
