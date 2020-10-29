import subprocess

import power_verification

import cloud_tracking
from config import cloud_tracking_config as ct_cfg, substation_info as substation_cfg


def create_livestream():
    if ct_cfg.livestream_online is True:
        command = ['ffmpeg',
                   '-rtsp_transport', 'tcp',
                   '-i', 'rtsp://192.168.0.10:8554/CH001.sdp',
                   '-f', 'mpegts',
                   '-s', '1280x1200',
                   '-codec:v', 'mpeg1video',
                   '-b:v', '3000k',
                   '-framerate', '30',
                   '-r', '30',
                   '-bf', '0',
                   ct_cfg.URL_APP_SERVER+'cloudtrackinglivestream/'+substation_cfg.id]
    else:
        command = ['ffmpeg',
                   '-rtsp_transport', 'tcp',
                   '-i', ct_cfg.VIDEO_PATH,
                   '-f', 'mpegts',
                   '-s', '1280x1200',
                   '-codec:v', 'mpeg1video',
                   '-b:v', '3000k',
                   '-framerate', '30',
                   '-r', '30',
                   '-bf', '0',
                   ct_cfg.URL_APP_SERVER+'cloudtrackinglivestream/'+substation_cfg.id]
    subprocess.call(command)


def main():
    create_livestream()
    cloud_tracking.main()
    power_verification.main()


main()
