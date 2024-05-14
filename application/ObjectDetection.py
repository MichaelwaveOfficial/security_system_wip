from ObjectTracking import ObjectTracking
from FileHandling import FileHandling
from Camera import Camera


import numpy as np, cv2, os, time


class ObjectDetection(object):

    '''
    
    '''

    def __init__(self, KERNEL_SIZE = (3,3)) -> None:

        self.camera = Camera()
        
        self.KERNEL = KERNEL_SIZE

        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

        self.object_tracking = ObjectTracking()

        self.file_handling = FileHandling()

        # Dictionary to correlate threat levels with OpenCV BGR colours.
        self.threat_levels : dict[int, tuple[int, int, int]] = {
            # Level 1 = Green (Okay)
            1 : (0, 255, 0),
            # Level 2 = Orange (Warning)
            2 : (0, 165, 255),
            # Level 3 = Red (Danger)
            3 : (0, 0, 255),
        }

    
    def capture_frame(self, frame, directory):

        # Create directory if it does not exist. 
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Join the desired directory and the filename to save capture.
        filename = f'{directory}{str(time.strftime(self.file_handling.FORMATTED_FILENAME_DATE))}'

        # Write capture to directory with native filename.
        cv2.imwrite(f'{filename}.jpg', frame)

        # Once new capture is written, check if file limit has been exceeded and remove older captures to avoid resource exhaustion.
        self.file_handling.check_file_exhaustion(directory, self.file_handling.MAXIMUM_FILES_STORED)


    def downsample_frame(self, frame, sample_scale):

        '''
        '''

        sample_width = int(frame.shape[1] * sample_scale / 100)
        sample_height = int(frame.shape[0] * sample_scale / 100)

        sampled_dimensions = (sample_width, sample_height)

        downsampled_frame = cv2.resize(
            frame,
            sampled_dimensions,
            cv2.INTER_AREA
        )

        return downsampled_frame


    def process_frames(self, frame):

        scale = 20

        self.downsample_frame(frame, scale)

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        morphological_operation = cv2.GaussianBlur(grayscale_frame, self.KERNEL, 0)

        foreground_mask = self.background_subtractor.apply(morphological_operation)

        return foreground_mask
    

    def draw_bounding_boxes(self, frame : np.ndarray, detections) -> np.ndarray:

        '''
        Draws bounding boxes for supplied detection data, this includes the ID value, the detections center points and colours them based on their current threat level. Visualising 
        the danger the detection poses.

        :param: frame - Frame for bounding boxes and other supporting data to be drawn upon.
        :param: detections - List of detections data to supply this functions features. [x, y, w, h, ID, threat_level] 
        :return: frame - Frame with data appended and visualised.
        :return: threat_level - Current threat level for the detection for external class logic.
        '''

        # Initialise threat level variable in case none available from the start. 
        threat_level = 1
        threat_text = 'N/A'

        # Iterate over each detection in the list provided.
        for detection in detections:

            # Accumulate detection data.
            x, y, w, h, threat_level = detection

            x_pred, y_pred = self.object_tracking.kf_predict(x, y)

            # Associate visualiation colour with the detections threat level.
            if threat_level == 1:
                detection_colour = self.threat_levels[threat_level]
                threat_text = 'Low'
            elif threat_level == 2:
                detection_colour = self.threat_levels[threat_level]
                threat_text = 'Medium'
            elif threat_level == 3:
                detection_colour = self.threat_levels[threat_level]
                threat_text = 'High'
            else: 
                # default error colour. 
                detection_colour = (0, 0, 0)

            # Use OpenCv to draw onto the frame. 
            cv2.putText(
                img=frame,
                text=f'Threat Level: {threat_text}', 
                org=(w // 2, y // 2 - 12),
                fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=2, 
                color=detection_colour,
                thickness=2
            )

            # Predicted kalman bbox
            cv2.rectangle(
                img=frame,
                pt1=(x_pred, y_pred), 
                pt2=(x_pred + w, y_pred + h), 
                color=detection_colour,
                thickness=2
            )   

        # Return processed frame and detections threat_level.
        return frame, threat_level
    
    
    def motion_detection(self, prev_frame, curr_frame, camera : Camera):

        '''
        '''

        motion_detected = False

        # Check frames passed are not None Type. Raise exception if they are. 
        if curr_frame is None or prev_frame is None:
            return ValueError('Provided frames were returned as None!')

        prev = self.process_frames(prev_frame)
        curr = self.process_frames(curr_frame)

        frame_differencing = cv2.absdiff(prev, curr)

        _, thresholded_frame_pixels = cv2.threshold(
            frame_differencing,
            254, 
            255,
            cv2.THRESH_BINARY,
        )

        # IF the sum of thresholded_frame_pixels is greater than the thresholded value. (Very large value divided for closer approximation).
        if (np.sum(thresholded_frame_pixels) / 100) > camera.settings['sensitivity']:

            # Set motion_detected boolean value to true.
            motion_detected = True 

        return motion_detected


    def register_detections(self, frame, camera : Camera, threshold = 1500, range = 100):

        detections = []

        # Check frames passed are not None Type. Raise exception if they are. 
        if frame is None:
            return ValueError('Provided frames were returned as None!')

        processed_frame = self.process_frames(frame)

        _, masked_frame = cv2.threshold(
            processed_frame, 
            camera.settings['range'],
            255,
            cv2.THRESH_BINARY
        )

        highlighted_contours, _ = cv2.findContours(
            masked_frame,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in highlighted_contours:

            contour_area = cv2.contourArea(contour)

            if contour_area > camera.settings['threshold']:

                # Compute the bounding box data for that contour.
                x, y, w, h = cv2.boundingRect(contour)

                # Append the data including size and coordinates to the list. 
                detections.append( [x, y, w, h] )

        return frame, detections
