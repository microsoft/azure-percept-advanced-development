"""
Adapted from:  https://github.com/NVIDIA-AI-IOT/jetcam/blob/master/jetcam/usb_camera.py
"""

from .camera import Camera
import cv2
import atexit
import numpy as np
import threading
import traitlets


class CaptureCamera(Camera):
    
    capture_fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=640)
    capture_height = traitlets.Integer(default_value=480)   
    capture_device = traitlets.Unicode(default_value="/dev/video0")
    
    def __init__(self, *args, **kwargs):
        super(CaptureCamera, self).__init__(*args, **kwargs)
        try:            
            self.cap = cv2.VideoCapture(self.capture_device)

            re, image = self.cap.read()
            
            if not re:
                raise RuntimeError('Could not read image from camera.')
            
        except:
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

        atexit.register(self.cap.release)

    def release(self):
        """Release the video resource"""
        self.cap.release()
    
    def _read(self):
        re, image = self.cap.read()
        if re:
            image_resized = cv2.resize(image, (int(self.width),int(self.height)))
            return image_resized
        else:
            raise RuntimeError('Could not read image from camera')