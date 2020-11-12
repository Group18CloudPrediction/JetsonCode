import base64
import subprocess as sp
import sys
import time

import cv2
import numpy as np
import pymongo
import socketio

from config import cloud_tracking_config as ct_cfg
from config import creds
from config import substation_info as substation_cfg
from imageProcessing import fisheye_mask as fisheye
from imageProcessing.coverage import cloud_recognition
from imageProcessing.sunPos import mask_sun_pixel, mask_sun_pysolar
from opticalFlow import opticalDense


def current_milli_time(): return int(round(time.time() * 1000))


sock = None
db = None


def initialize_socketio(url):
    """Initialize socket io"""
    sio = socketio.Client()

    try:
        @sio.event
        def connect():
            print("Connected to Application Server")
        sio.connect(url)

    except socketio.exceptions.ConnectionError as e:
        print(e)
        sio = None
    return sio


def send_coverage_to_db(coverage):
    """Sends percent cloud coverage to database"""
    cloud = np.count_nonzero(coverage[:, :, 3] > 0)
    not_cloud = np.count_nonzero(coverage[:, :, 3] == 0)

    coverage = np.round((cloud / (cloud + not_cloud)) * 100, 2)

    post = {
        "author": "cloud_tracking.py",
        "cloud_coverage": coverage,
        "system_num": substation_cfg.id
    }

    posts = db.cloudCoverageData
    post_id = posts.insert_one(post).inserted_id
    print("DB sent -> cover_post_id: " + str(post_id))


def send_image_to_db(frame):
    """Send inputted image as png byte array to database"""
    success, im_buffer = cv2.imencode('.png', frame)
    cv2.imwrite('cloudImage.png', frame)
    if success is False:
        print("couldnt encode png image")
        return

    byte_image = im_buffer.tobytes()
    post = {
        "author": "cloud_tracking.py",
        "camera_frame": byte_image,
        "system_num": substation_cfg.id
    }

    posts = db.cloudImage
    post_id = posts.insert_one(post).inserted_id
    print("DB sent -> img_post_id: " + str(post_id))


def send_image_socket(image, event_name):
    """Emits an image through socketIO connection"""
    success, im_buffer = cv2.imencode('.png', image)

    if success is False:
        print("couldnt encode png image")
        return

    frame = im_buffer.tobytes()
    sock.emit(event_name, [frame, substation_cfg.id])
    print("sock -> emit: ", event_name)


def send_cloud_socket(frame):
    """Sends cloud image to website via socketIO"""
    send_image_socket(frame, 'coverage')


def send_shadow_socket(coverage):
    """Sends shadow image to website via socketIO"""
    shadow = coverage.copy()

    shadow[(shadow[:, :, 3] > 0)] = (0, 0, 0, 127)
    send_image_socket(shadow, 'shadow')


def byteRead_to_npArray(rawimg):
    """Transform byte read into a numpy array"""
    img = np.frombuffer(rawimg, dtype='uint8')
    img = img.reshape((768, 1024, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.fliplr(img)
    return img


def experiment_step(prev, next):
    """Perform image processing + optical flow on the two inputted frames"""
    clouds = None
    sun_center = None
    sun_pixels = None

    # Apply fisheye mask
    if ct_cfg.do_mask is True:
        prev, next = fisheye.create_fisheye_mask(prev, next)

    # crop image to square to eliminate curved lens interference
    if ct_cfg.do_crop is True:
        prev = fisheye.image_crop(prev)
        next = fisheye.image_crop(next)

    # Locate center of sun + pixels that are considered "sun"
    if ct_cfg.livestream_online is True:
        sun_center, sun_pixels = mask_sun_pysolar(
            substation_cfg.LAT, substation_cfg.LONG, ct_cfg.SUN_RADIUS)
    else:
        # If locally stored video is being used for footage, sun must be located by pixel intensity, as time and long_lat coordinates aren't available to use pysolar
        sun_center, sun_pixels = mask_sun_pixel(next, ct_cfg.SUN_RADIUS)

    cv2.circle(prev, sun_center, ct_cfg.SUN_RADIUS, (255, 0, 0), -1)
    cv2.circle(next, sun_center, ct_cfg.SUN_RADIUS, (255, 0, 0), -1)

    # Convert pixel opacity values to not cloud (0) or cloud (1) -> based on pixel saturation
    clouds = cloud_recognition(next)

    # Find the flow vectors for the prev and next images
    flow_vectors = opticalDense.calculate_opt_dense(prev, next)

    # Draw vector field based on cloud/not cloud image and displacement vectors
    flow, __, __ = opticalDense.draw_arrows(clouds.copy(), flow_vectors, 10)

    # Return experiment step
    return clouds, flow


def create_ffmpeg_pipe():
    if ct_cfg.livestream_online:
        command = ['ffmpeg',
                   '-loglevel', 'panic',
                   '-nostats',
                   '-rtsp_transport', 'tcp',
                   '-i', 'rtsp://192.168.0.10:8554/CH001.sdp',
                   '-s', '1024x768',
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vf', 'fps=fps=1/8',
                   '-vcodec', 'rawvideo', '-']
    else:
        command = ['ffmpeg',
                   '-loglevel', 'panic',
                   '-nostats',
                   '-i', ct_cfg.VIDEO_PATH,
                   '-s', '1024x768',
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    return pipe


def experiment_ffmpeg_pipe(pipe):
    while True:
        try:
            prev_rawimg = pipe.stdout.read(1024*768*3)
            # transform the byte read into a numpy array
            prev = byteRead_to_npArray(prev_rawimg)

            # throw away the data in the pipe's buffer.
            pipe.stdout.flush()

            next_rawimg = pipe.stdout.read(1024*768*3)
            # transform the byte read into a np array
            next = byteRead_to_npArray(next_rawimg)

            # throw away the data in the pipe's buffer.
            pipe.stdout.flush()

            cloudPNG, flow = experiment_step(prev, next)

            # Send cloud image, and shadow image via socketIO to website
            if ct_cfg.socket_on is True:
                send_cloud_socket(flow)
                send_shadow_socket(cloudPNG)

            # Send cloud coverage data to MongoDB
            if ct_cfg.send_to_db is True:
                if ct_cfg.send_img_to_db is True:
                    send_image_to_db(prev)
                send_coverage_to_db(cloudPNG)

        except Exception as inst:
            print(inst)
            break
    return


def main():
    global sock
    global db

    # creds will need to be created on each system
    client = pymongo.MongoClient("mongodb+srv://" + creds.username + ":" +
                                 creds.password + "@cluster0.lgezy.mongodb.net/<dbname>?retryWrites=true&w=majority")
    db = client.cloudTrackingData

    # initialize ffmpeg pipe
    pipe = create_ffmpeg_pipe()

    # initialize socket
    if ct_cfg.socket_on is True:
        sock = initialize_socketio(ct_cfg.URL_APP_SERVER)
    else:
        sock = None

    experiment_ffmpeg_pipe(pipe)

    if sock is not None:
        sock.disconnect()


main()
