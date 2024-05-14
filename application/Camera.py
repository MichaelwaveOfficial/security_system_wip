import cv2, numpy as np, time
 
class Camera(object):

    '''
    The Camera class handles functionality associated with accessing the devices onboard camera, processing the input taken. 
    '''

    def __init__(self) -> None:

        # Dictionary to store, access and retrieve the cameras settings. 
        self.settings = {
            # Camera on/off
            'camera_toggle' : True,
            # Time to re-initialise motion detection functionality.
            'sleep' : 5, 
            # Threshold setting for motion detection.
            'threshold' : 1500, 
            # Sensitivity setting for motion detection.
            'sensitivity' : 1800,
            #
            'range' : 100,
            # Stream framerate setting. 
            'fps' : 60,
        }

        # Access the onboard camera using OpenCV, 0 represents camera, 1 for video input. 
        self.video_stream = cv2.VideoCapture(0)


    ''' Functions concerned with the cameras functionality. '''


    def draw_clock(self, frame : np.ndarray, time : str) -> np.ndarray:

        '''
        dsa

        :param: frame - Frame to be copied and drawn upon.
        :param: time - Current time passed in as a string argument. 
        :return: clock_overlay - New frame copied from one parameterised, applies clock to the top left corner. Mitigates interference with computer vision.
        '''

        # Get dimensions of the parameterised frame for the copy. 
        width, height, _ = frame.shape

        # Create overlay copy using supplied dimensions.
        clock_overlay = np.zeros((width, height, 3), dtype=np.uint8)

        # Background for the text.
        cv2.rectangle(
            img=clock_overlay, 
            pt1=(0, 0),
            pt2=(225, 50),
            color=(50, 50, 50),
            thickness = -1,
            lineType=cv2.LINE_8,
            shift=0
        )
        # Date text for the frame.
        cv2.putText(
            img=clock_overlay,
            text=str(time),
            org=(15, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1,
            color=(5, 100, 5), 
            thickness= 2
        )

        # Return frame copy. 
        return clock_overlay
        

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
    

    def retrieve_frame_CV2(self) -> np.ndarray:

        '''
        Read the frames from the OpenCv videoCapture object, return the frame.

        :return frame: Return the frame read.
        '''

        # Check video stream can be accessed properly. 
        if not self.video_stream.isOpened():
            raise RuntimeError('Failed to access onboard camera!')

        # Grab frame from the onboard cameras video stream. 
        ret, frame = self.video_stream.read()

        # Check whether frame has been returned or not before progressing further. 
        if not ret:
            raise IOError('Could not read frames from the camera!')
        
        # Return the native frame and its encoded counterpart. 
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