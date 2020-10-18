import subprocess as sp
import time
from multiprocessing import Process, Queue
from threading import Event, Thread
import socketio
import cv2
import numpy as np
import pymongo

from credentials import creds
from imageProcessing import fisheye_mask as fisheye
from imageProcessing.coverage import cloud_recognition
from imageProcessing.sunPos import mask_sun_pixel, mask_sun_pysolar
from opticalFlow import opticalDense


def current_milli_time(): return int(round(time.time() * 1000))


# VALUES
VIDEO_PATH = 'opticalFlow/test1sec.mp4'
# URL_APP_SERVER = 'http://localhost:3001/'
URL_APP_SERVER = 'https://cloudtracking-v2.herokuapp.com/'

# CONSTANTS
DISPLAY_SIZE = (512, 384)
SUN_RADIUS = 50
LAT = 28.601722
LONG = -81.198545

# FLAGS
livestream_online = False
send_to_db = True
socket_on = True
do_mask = True
do_crop = True


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


def create_ffmpeg_pipe():
    """Creates + executes command ffmpeg for video stream, and returns pipe"""
    if livestream_online is True:
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
                   '-i', VIDEO_PATH,
                   '-s', '1024x768',
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    return pipe


def experiment_step(prev, next):
    """Perform image processing + optical flow on the two inputted frames"""
    clouds = None
    sun_center = None
    sun_pixels = None

    # Apply fisheye mask
    if do_mask is True:
        prev, next = fisheye.create_fisheye_mask(prev, next)

    # crop image to square to eliminate curved lens interference
    if do_crop is True:
        prev = fisheye.image_crop(prev)
        next = fisheye.image_crop(next)

    # Locate center of sun + pixels that are considered "sun"
    if livestream_online is True:
        sun_center, sun_pixels = mask_sun_pysolar(LAT, LONG, SUN_RADIUS)
    else:
        # If locally stored video is being used for footage, sun must be located by pixel intensity, as time and long_lat coordinates aren't available to use pysolar
        sun_center, sun_pixels = mask_sun_pixel(next, SUN_RADIUS)

    cv2.circle(prev, sun_center, SUN_RADIUS, (255, 0, 0), -1)
    cv2.circle(next, sun_center, SUN_RADIUS, (255, 0, 0), -1)

    # Convert pixel opacity values to not cloud (0) or cloud (1) -> based on pixel saturation
    clouds = cloud_recognition(next)

    # Find the flow vectors for the prev and next images
    flow_vectors = opticalDense.calculate_opt_dense(prev, next)

    # Draw vector field based on cloud/not cloud image and displacement vectors
    flow, __, __ = opticalDense.draw_arrows(clouds.copy(), flow_vectors, 10)

    # Return experiment step
    return clouds, flow


def byteRead_to_npArray(rawimg):
    """Transform byte read into a numpy array"""
    img = np.frombuffer(rawimg, dtype='uint8')
    img = img.reshape((768, 1024, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.fliplr(img)
    return img


class CloudTrackingRunner(Thread):
    def __init__(self):
        Thread.__init__(self)

        # creds will need to be created on each system
        self.client = pymongo.MongoClient("mongodb+srv://" + creds.username + ":" +
                                          creds.password + "@cluster0.lgezy.mongodb.net/<dbname>?retryWrites=true&w=majority")
        self.db = self.client.cloudTrackingData

        # initialize ffmpeg pipe
        self.pipe = create_ffmpeg_pipe()

        # initialize socket
        if socket_on is True:
            self.sock = initialize_socketio(URL_APP_SERVER)
        else:
            self.sock = None

    def run(self):
        while True:
            try:
                startTime = current_milli_time()

                # grab frame from pipe
                prev_rawimg = self.pipe.stdout.read(1024*768*3)
                # throw away the data in the pipe's buffer.
                self.pipe.stdout.flush()

                # ensure there's at least 30 seconds between frames
                time.sleep(30)

                # grab next frame from pipe
                next_rawimg = self.pipe.stdout.read(1024*768*3)
                # throw away the data in the pipe's buffer.
                self.pipe.stdout.flush()

                prev = byteRead_to_npArray(prev_rawimg)
                next = byteRead_to_npArray(next_rawimg)

                cloudPNG, flow = experiment_step(prev, next)

                # Send cloud image, and shadow image via socketIO to website
                if socket_on is True:
                    self.send_cloud_socket(flow)
                    self.send_shadow_socket(cloudPNG)

                # Send cloud coverage data to MongoDB
                if send_to_db is True:
                    self.send_image_to_db(prev)
                    self.send_coverage_to_db(cloudPNG)

                finishTime = current_milli_time()

                time.sleep(60 - (finishTime - startTime) / 1000)
            except Exception as inst:
                print(inst)
                break

    def send_image_to_db(self, frame):
        """Send inputted image as png byte array to database"""
        success, im_buffer = cv2.imencode('.png', frame)
        cv2.imwrite('cloudImage.png', frame)
        if success is False:
            print("couldnt encode png image")
            return

        byte_image = im_buffer.tobytes()
        post = {
            "author": "cloud_tracking.py",
            "camera_frame": byte_image
        }

        posts = self.db.cloudImage
        post_id = posts.insert_one(post).inserted_id
        print("img_post_id: " + str(post_id))

    def send_coverage_to_db(self, coverage):
        """Sends percent cloud coverage to database"""
        cloud = np.count_nonzero(coverage[:, :, 3] > 0)
        not_cloud = np.count_nonzero(coverage[:, :, 3] == 0)

        coverage = np.round((cloud / (cloud + not_cloud)) * 100, 2)

        post = {
            "author": "cloud_tracking.py",
            "cloud_coverage": coverage
        }

        posts = self.db.cloudCoverageData
        post_id = posts.insert_one(post).inserted_id
        print("cover_post_id: " + str(post_id))

    def send_image_socket(self, image, event_name):
        """Emits an image through socketIO connection"""
        success, im_buffer = cv2.imencode('.png', image)

        if success is False:
            print("couldnt encode png image")
            return

        byte_image = im_buffer.tobytes()
        sock.emit(event_name, byte_image)

    def send_cloud_socket(self, frame):
        """Sends cloud image to website via socketIO"""
        self.send_image_socket(frame, 'coverage')

    def send_shadow_socket(self, coverage):
        """Sends shadow image to website via socketIO"""
        shadow = coverage.copy()

        # TURN SHADOW TO BLACK AND WHITE
        cv2.cvtColor(shadow, cv2.COLOR_BGR2GRAY)
        shadow[(shadow[:, :, 3] > 0)] = (0, 0, 0, 127)
        send_image_socket(shadow, 'shadow')


def main():
    cloudtracking_runner = CloudTrackingRunner()
    cloudtracking_runner.start()


if __name__ == "__main__":
    main()
