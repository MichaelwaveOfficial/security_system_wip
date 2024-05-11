
import cv2, numpy as np, time, io, picamera2

from picamera2 import Picamera2

 
class Camera(object):

    '''
    The Camera class handles functionality associated with accessing the devices onboard camera, processing the input taken. 
    '''

    def __init__(self) -> None:

        # Dictionary to store, access and retrieve the cameras settings. 
        self.settings = {
            # Camera on/off
            'camera_toggle' : True,
            # Detection alerts toggle.
            'vision_toggle' : False,
            # Time to re-initialise motion detection functionality.
            'sleep' : 5, 
            # Threshold setting for motion detection.
            'threshold' : 35, 
            # Sensitivity setting for motion detection.
            'sensitivity' :800,
            # Stream framerate setting. 
            'fps' : 30,
            # Stream resolution setting. 
            'resolution' : 720,
        }

        # Access the onboard camera using OpenCV, 0 represents camera, 1 for video input. 
        #self.video_stream = cv2.VideoCapture(0)]

        self.camera = picamera2.Picamera2()

        self.camera.configure(self.camera.create_video_configuration(
            main={
                'size' : (800, 600)
            }
        ))


    ''' Functions concerned with the cameras functionality. '''


    def encode_frame(self, frame : np.ndarray) -> bytes:

        '''
        Takes frame, converts it to bytes for HTTP upload.

        :param frame: Frame to be converted.
        :return bytes: Frame encoded as bytes.
        '''

        # Encode the frames and check the operation has been succesful with return bool.
        ret, jpeg = cv2.imencode('.jpg', frame)

        # If nothing returned after operation, inform user. 
        if not ret:
            raise RuntimeError('Frames could not be converted!')
        
        # Return encoded frame.
        return jpeg.tobytes()
    

    def retrieve_frames_PIC(self):

        while True:

            frame = self.camera.get_frame()

            if not frame: 
                raise IOError('Could not read frames from the camera!')

            return frame
            

    def enforce_frame_rate(self, elapsed_time : int) -> None:

        '''
        Strictly enforce the specified framerate for the stream. Can be decreased to reduce computational load on the device. 

        :param: elapsed_time - Time passed from the inital start.
        '''

        # Calcuate the time required to retrieve next frame.
        timeout = (1 / self.settings['fps']) - elapsed_time

        # If calculated timeout greater than nothing. Pause time taken to fetch next frame.
        if timeout > 0:
            time.sleep(timeout)