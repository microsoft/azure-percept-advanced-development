"""
To capture images from the Percept DK Vision SoM Camera
Adapted from:  https://github.com/NVIDIA-AI-IOT/jetcam/blob/master/jetcam/usb_camera.py
"""
import traitlets
import threading
import numpy as np
import cv2
import atexit


class Camera(traitlets.HasTraits):

    value = traitlets.Any()
    width = traitlets.Integer(default_value=640)
    height = traitlets.Integer(default_value=480)
    cap_format = traitlets.Unicode(default_value='bgr8')
    running = traitlets.Bool(default_value=False)    
    capture_device = traitlets.Unicode(default_value="/dev/video0")
    
    def __init__(self, *args, **kwargs):
        super(Camera, self).__init__(*args, **kwargs)
        if self.cap_format == 'bgr8':
            self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._running = False
        try:            
            self.cap = cv2.VideoCapture(self.capture_device)

            re, image = self.cap.read()
            
            if not re:
                raise RuntimeError('Could not read image from camera.')
        except:
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')
        atexit.register(self.cap.release)

    def _read(self):
        re, image = self.cap.read()
        if re:
            image_resized = cv2.resize(image, (int(self.width),int(self.height)))
            return image_resized
        else:
            raise RuntimeError('Could not read image from camera')

    def _capture_frames(self):
        while True:
            if not self._running:
                break
            self.value = self._read()
            
    def read(self):
        if self._running:
            raise RuntimeError('Cannot read directly while camera is running')
        self.value = self._read()
        return self.value
            
    def release(self):
        """Release the video resource"""
        self.cap.release()
            
    @traitlets.observe('running')
    def _on_running(self, change):
        if change['new'] and not change['old']:
            # transition from not running -> running
            self._running = True
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()
        elif change['old'] and not change['new']:
            # transition from running -> not running
            self._running = False
            self.thread.join()