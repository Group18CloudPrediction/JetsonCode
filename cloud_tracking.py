import subprocess as sp
import time
from multiprocessing import Process, Queue
from threading import Event, Thread

import cv2
import numpy as np
import pymongo

import forecast
from credentials import creds
from imageProcessing import fisheye_mask as fisheye
from imageProcessing.coverage import cloud_recognition
from imageProcessing.sunPos import mask_sun_pixel, mask_sun_pysolar
from opticalFlow import opticalDense


def current_milli_time(): return int(round(time.time() * 1000))


# VALUES
VIDEO_PATH = 'opticalFlow/test1sec.mp4'

# CONSTANTS
DISPLAY_SIZE = (512, 384)
SECONDS_PER_FRAME = 1
SECONDS_PER_PREDICTION = 30
LAT = 28.601722
LONG = -81.198545

# FLAGS
livestream_online = False
display_images = True
send_images = True
do_coverage = True
do_mask = True
do_crop = True


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
    before = current_milli_time()
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

    # Locate center of sun + pixels that are the sun if livestream is on
    if livestream_online is True:
        sun_center, sun_pixels = mask_sun_pysolar(LAT, LONG)
    else:
        # If livestream isn't on, sun must be located by pixel, as time and long_lat coordinates aren't available to use pysolar
        sun_center = mask_sun_pixel(next)

    # Convert pixel RGB values to not sun (0) or sun (255)
    clouds = cloud_recognition(next, sun_center)

    # Find the flow vectors for the prev and next images
    flow_vectors = opticalDense.calculate_opt_dense(prev, next)

    # Draw vector field based on cloud/not cloud image and displacement vectors
    flow, x_i, x_f = opticalDense.draw_arrows(clouds.copy(), flow_vectors, 10)

    after = current_milli_time()
    print('Experiment step took: %s ms' % (after - before))

    # Return experiment step
    return (clouds, flow, x_i, x_f, sun_pixels)


def experiment_display(prev, next, flow, coverage):
    """Display results of experiment step with cv.imshow"""
    if display_images is False:
        return
    # Resize the images for visibility
    flow_show = cv2.resize(flow, DISPLAY_SIZE)
    prev_show = cv2.resize(prev, DISPLAY_SIZE)
    next_show = cv2.resize(next, DISPLAY_SIZE)

    # Show the images
    cv2.imshow('flow?', flow_show)
    cv2.imshow('previous', prev_show)
    cv2.imshow('next', next_show)

    # Wait 30s for ESC and return false if pressed
    k = cv2.waitKey(30) & 0xff
    if (k == 27):
        return False
    return True


def forecast_(queue, x_i, x_f, sun_pixels):
    times = forecast.get_time(
        x_i, x_f, sun_pixels, 18, 1/SECONDS_PER_FRAME)

    prediction_frequencies = np.array(
        np.unique(np.round(times), return_counts=True)).T

    queue.put(prediction_frequencies)


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

        self.pipe = create_ffmpeg_pipe()

        self.prediction_queue = Queue()

        self.sleep_time = 60

    def run(self):
        while True:
            try:
                before = current_milli_time()

                # grab frame from pipe
                prev_rawimg = self.pipe.stdout.read(1024*768*3)
                prev = byteRead_to_npArray(prev_rawimg)
                # throw away the data in the pipe's buffer.
                self.pipe.stdout.flush()

                time.sleep(10)

                # grab next frame from pipe
                next_rawimg = self.pipe.stdout.read(1024*768*3)
                next = byteRead_to_npArray(next_rawimg)
                # throw away the data in the pipe's buffer.
                self.pipe.stdout.flush()

                (cloudPNG, flow, x_i, x_f,
                 sun_pixels) = experiment_step(prev, next)

                after = current_milli_time()

                if (after - before > (1000 * SECONDS_PER_PREDICTION)):
                    # p = Process(target=forecast_, args=(queue, prev, next))
                    # p.start()
                    forecast_(self.prediction_queue, x_i, x_f, sun_pixels)

                # if prediction queue isn't empty, send predictions to MongoDB
                if(self.prediction_queue.empty() != True):
                    prediction_frequencies = self.prediction_queue.get()
                    print("Sending predictions", np.shape(
                        prediction_frequencies))
                    self.send_predictions(prediction_frequencies)

                # Send cloud coverage data, cloud image, and shadow image to MongoDB
                self.send_cloud(flow)
                self.send_shadow(cloudPNG)
                self.send_coverage(cloudPNG)

            except Exception as inst:
                print(inst)
                break

            time.sleep(self.sleep_time -
                       ((time.time() - before) % self.sleep_time))

    def send_predictions(self, data):
        """Send cloud prediction output to database"""
        post = {
            "author": "cloud_tracking.py",
            "cloudPrediction": {a: b for a, b in data}
        }

        posts = self.db.cloudPredictionData
        post_id = posts.insert_one(post).inserted_id
        print("post_id: " + str(post_id))

    def send_coverage(self, coverage):
        """Sends percent cloud coverage to database"""
        cloud = np.count_nonzero(coverage[:, :, 3] > 0)
        not_cloud = np.count_nonzero(coverage[:, :, 3] == 0)

        coverage = np.round((cloud / (cloud + not_cloud)) * 100, 2)

        print(coverage)

        post = {
            "author": "cloud_tracking.py",
            "cloud_coverage": coverage
        }

        posts = self.db.cloudCoverageData
        post_id = posts.insert_one(post).inserted_id
        print("post_id: " + str(post_id))

    def send_cloud(self, frame):
        """Sends cloud image to database"""
        success, im_buffer = cv2.imencode('.png', frame)

        if success is False:
            print("couldnt encode png image")
            return

        byte_image = im_buffer.tobytes()
        post = {
            "author": "cloud_tracking.py",
            "coverage": byte_image
        }

        posts = self.db.cloudImage
        post_id = posts.insert_one(post).inserted_id
        print("post_id: " + str(post_id))

    def send_shadow(self, coverage):
        """Sends shadow image to database"""
        shadow = coverage.copy()
        shadow[(shadow[:, :, 3] > 0)] = (0, 0, 0, 127)
        success, im_buffer = cv2.imencode('.png', shadow)

        if success is False:
            print("couldnt encode png image")
            return

        byte_image = im_buffer.tobytes()
        post = {
            "author": "cloud_tracking.py",
            "shadow": byte_image
        }

        posts = self.db.shadowImage
        post_id = posts.insert_one(post).inserted_id
        print("post_id: " + str(post_id))


def main():
    cloudtracking_runner = CloudTrackingRunner()
    cloudtracking_runner.start()


if __name__ == "__main__":
    main()
