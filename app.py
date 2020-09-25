import base64
import socketio

import cloud_tracking

# Constants
URL_APP_SERVER          = 'http://localhost:3000/'
# URL_APP_SERVER          = 'https://cloudtracking-v2.herokuapp.com/'

# FLAGS -- used to test different functionalities
sock = None

# Initialize socket io
def initialize_socketio(url):
    sio = socketio.Client()

    @sio.event
    def connect():
        print("Connected to Application Server")

    sio.connect(url)
    return sio

def main():
    global sock
    sock = initialize_socketio(URL_APP_SERVER)
    pipe = cloud_tracking.create_ffmpeg_pipe('opticalFlow/20191121-134744.mp4')

    cloud_tracking.experiment_ffmpeg_pipe(pipe)
    if sock is not None:
        sock.disconnect()

main()
