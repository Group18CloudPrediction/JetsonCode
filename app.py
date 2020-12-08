import base64
import subprocess as sp
import sys
import time

import cv2
import numpy as np
import pymongo
import socketio

from config import cloud_tracking_config as ct_cfg, creds, substation_info as substation_cfg
<<<<<<< HEAD
from config import cloud_tracking_config 
=======
from config import cloud_tracking_config
>>>>>>> 2d5767fd2ba8bd0a14815c4257f7117c2ba07b69
from imageProcessing import fisheye_mask as fisheye, coverage
from opticalFlow import opticalDense


def current_milli_time(): return int(round(time.time() * 1000))


# Constants
# URL_APP_SERVER          = 'http://localhost:3001/'
<<<<<<< HEAD
URL_APP_SERVER = cloud_tracking_config.URL_APP_SERVER #'https://cloudtracking-v2.herokuapp.com/'
=======
URL_APP_SERVER = cloud_tracking_config.URL_APP_SERVER 
>>>>>>> 2d5767fd2ba8bd0a14815c4257f7117c2ba07b69
DISPLAY_SIZE = (512, 384)
MASK_RADIUS_RATIO = 3.5
SECONDS_PER_FRAME = 1

sock = None
db = None

# FLAGS -- used to test different functionalities
send_images = True
do_coverage = True
do_mask = True
do_crop = True


def initialize_socketio(url):
    """Initialize socket io"""
    sio = socketio.Client()

    @sio.event
    def connect():
        print("Connected to Application Server")

    sio.connect(url)
    return sio


def send_coverage(coverage):
    """Sends percent cloud coverage to database"""
    if sock is None:
        return

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


def send_image(image, event_name):
    """Emits an image through socketIO connection"""
    if send_images is False or sock is None:
        return
    success, im_buffer = cv2.imencode('.png', image)

    if success is False:
        print("couldnt encode png image")
        return

    byte_image = im_buffer.tobytes()
    sock.emit(event_name + substation_cfg.id, byte_image)


def send_cloud(frame):
    """Sends cloud image to website via socketIO"""
    send_image(frame, 'coverage')


def send_shadow(coverage):
    """Sends shadow image to website via socketIO"""
    shadow = coverage.copy()
    shadow[(shadow[:, :, 3] > 0)] = (0, 0, 0, 127)
    send_image(shadow, 'shadow')


def black2transparent(bgr):
    """Make all black pixels transparent"""
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[(bgra[:, :, 0:3] == [0, 0, 0]).all(2)] = (0, 0, 0, 0)
    return bgra


def experiment_step(prev, next):
    """Perform image processing + optical flow on the two inputted frames"""
    before = current_milli_time()
    clouds = None

    # Apply fisheye mask
    if do_mask is True:
        prev, next = fisheye.create_fisheye_mask(prev, next)

    # crop image to square to eliminate curved lens interference
    if do_crop is True:
        prev = fisheye.image_crop(prev)
        next = fisheye.image_crop(next)

    # Find the flow vectors for the prev and next images
    flow_vectors = opticalDense.calculate_opt_dense(prev, next)

    # Convert pixel opacity values to not cloud (0) or cloud (1) -> based on pixel saturation
    if do_coverage is True:
        clouds = coverage.cloud_recognition(next)

    # Draw vector field based on cloud/not cloud image and displacement vectors
    flow, _, __ = opticalDense.draw_arrows(clouds.copy(), flow_vectors)

    after = current_milli_time()
    elapsed = (after - before)
    print('Experiment step took: %s ms' % elapsed)

    # Return experiment step
    return (prev, next, flow, clouds)


def create_ffmpeg_pipe():
    """Creates + executes command ffmpeg for video stream, and returns pipe"""
    if ct_cfg.livestream_online is True:
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
    before = current_milli_time()

    # BRONZE SOLUTION
    First = True
    BLOCK = False

    while True:
        try:
            prev_rawimg = pipe.stdout.read(1024*768*3)
            # transform the byte read into a numpy array
            prev = np.fromstring(prev_rawimg, dtype='uint8')
            prev = prev.reshape((768, 1024, 3))
            prev = cv2.cvtColor(prev, cv2.COLOR_RGB2BGR)
            prev = np.fliplr(prev)

            # throw away the data in the pipe's buffer.
            pipe.stdout.flush()

            next_rawimg = pipe.stdout.read(1024*768*3)
            # transform the byte read into a np array
            next = np.fromstring(next_rawimg, dtype='uint8')
            next = next.reshape((768, 1024, 3))
            next = cv2.cvtColor(next, cv2.COLOR_RGB2BGR)
            next = np.fliplr(next)

            # throw away the data in the pipe's buffer.
            pipe.stdout.flush()

            (prev, next, flow, coverage) = experiment_step(prev, next)

            after = current_milli_time()

            send_cloud(flow)
            send_shadow(coverage)
            send_coverage(coverage)
        except Exception as inst:
            print(inst)
            break
    return


def main():
    global sock
    sock = initialize_socketio(URL_APP_SERVER)

    global db
<<<<<<< HEAD
    client = pymongo.MongoClient(creds.base_url + creds.username + creds.seperator +
                                 creds.password + creds.cluster_url)
    db = client.cloudTrackingData
=======
    client = pymongo.MongoClient(creds.base_url + creds.username + creds.separator +
                                 creds.password + creds.cluster_url)
    db = client.CloudTrackingData
>>>>>>> 2d5767fd2ba8bd0a14815c4257f7117c2ba07b69

    pipe = create_ffmpeg_pipe()

    experiment_ffmpeg_pipe(pipe)
    if sock is not None:
        sock.disconnect()


main()
