import subprocess as sp
import base64
import time
import sys
import cv2
import numpy as np
import socketio
from multiprocessing import Process, Queue

from config import cloud_tracking_config as ct_cfg, substation_info as substation_cfg, creds
from imageProcessing import fisheye_mask as fisheye
from imageProcessing.coverage import cloud_recognition
from imageProcessing.sunPos import mask_sun_pixel, mask_sun_pysolar
from opticalFlow import opticalDense


def current_milli_time(): return int(round(time.time() * 1000))


MASK_RADIUS_RATIO = 3.5
SECONDS_PER_FRAME = 1
SECONDS_PER_PREDICTION = 30

# FLAGS -- used to test different functionalities
display_images = True
send_images = True
do_coverage = True
sock = None

# Initialize socket io


def initialize_socketio(url):
    sio = socketio.Client()

    @sio.event
    def connect():
        print("Connected to Application Server")

    sio.connect(url)
    return sio


def send_coverage(coverage):
    if sock is None:
        return

    cloud = np.count_nonzero(coverage[:, :, 3] > 0)
    not_cloud = np.count_nonzero(coverage[:, :, 3] == 0)

    coverage = np.round((cloud / (cloud + not_cloud)) * 100, 2)

    print(coverage)

    sock.emit('coverage_data', {"cloud_coverage": coverage})


def send_image(image, event_name):
    if send_images is False or sock is None:
        return
    success, im_buffer = cv2.imencode('.png', image)

    if success is False:
        print("couldnt encode png image")
        return

    byte_image = im_buffer.tobytes()
    sock.emit(event_name, byte_image)

# send coverage image


def send_cloud(frame):
    send_image(frame, 'coverage')


def send_shadow(coverage):
    shadow = coverage.copy()
    shadow[(shadow[:, :, 3] > 0)] = (0, 0, 0, 127)
    send_image(shadow, 'shadow')


def black2transparent(bgr):
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[(bgra[:, :, 0:3] == [0, 0, 0]).all(2)] = (0, 0, 0, 0)
    return bgra


def experiment_step(prev, next):
    before = current_milli_time()
    clouds = None

    if ct_cfg.do_mask is True:
        mask = fisheye.create_fisheye_mask(prev, next, MASK_RADIUS_RATIO)

    if ct_cfg.do_crop is True:
        w = prev.shape[0]
        h = prev.shape[1]
        s = w / MASK_RADIUS_RATIO

        top_edge = int(h/2-s)
        bottom_edge = int(h/2 + s)

        left_edge = int(w/2-s)
        right_edge = int(w/2 + s)
        prev = prev[left_edge:right_edge,  top_edge:bottom_edge, :]
        next = next[left_edge:right_edge,  top_edge:bottom_edge, :]

    # Find the flow vectors for the prev and next images
    flow_vectors = opticalDense.calculate_opt_dense(prev, next)

    # Locate center of sun + pixels that are considered "sun"
    if ct_cfg.livestream_online is True:
        sun_center, sun_pixels = mask_sun_pysolar(
            substation_cfg.LAT, substation_cfg.LONG, ct_cfg.SUN_RADIUS)
    else:
        # If locally stored video is being used for footage, sun must be located by pixel intensity, as time and long_lat coordinates aren't available to use pysolar
        sun_center, sun_pixels = mask_sun_pixel(next, ct_cfg.SUN_RADIUS)

    cv2.circle(prev, sun_center, ct_cfg.SUN_RADIUS, (255, 0, 0), -1)
    cv2.circle(next, sun_center, ct_cfg.SUN_RADIUS, (255, 0, 0), -1)

    if do_coverage is True:
        clouds = cloud_recognition(next)

    flow, _, __ = opticalDense.draw_arrows(clouds.copy(), flow_vectors)

    after = current_milli_time()
    elapsed = (after - before)
    print('Experiment step took: %s ms' % elapsed)

    # Return experiment step
    return (prev, next, flow, clouds)


def experiment_display(prev, next, flow, coverage):
    if display_images is False:
        return
    # Resize the images for visibility
    flow_show = cv2.resize(flow, ct_cfg.DISPLAY_SIZE)
    prev_show = cv2.resize(prev, ct_cfg.DISPLAY_SIZE)
    next_show = cv2.resize(next, ct_cfg.DISPLAY_SIZE)

    # Show the images
    cv2.imshow('flow?', flow_show)
    cv2.imshow('previous', prev_show)
    cv2.imshow('next', next_show)

    # Wait 30s for ESC and return false if pressed
    k = cv2.waitKey(30) & 0xff
    if (k == 27):
        return False
    return True


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

            send_cloud(flow)
            send_shadow(coverage)
            send_coverage(coverage)

            # Break if ESC key was pressed
            if (experiment_display(prev, next, flow, coverage) == False):
                break
        except Exception as inst:
            print(inst)
            break
    return


def main():
    global sock
    sock = initialize_socketio(ct_cfg.URL_APP_SERVER)
    pipe = create_ffmpeg_pipe()

    experiment_ffmpeg_pipe(pipe)
    if sock is not None:
        sock.disconnect()


main()
