from ObjectTracking import ObjectTracking
from FileHandling import FileHandling


import numpy as np, cv2, os


class ObjectDetection(object):

    '''
    
    '''

    def __init__(self, KERNEL_SIZE = (3,3)) -> None:
        
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
        filename = f'{directory}{self.file_handling.FORMATTED_FILENAME_DATE}'

        # Write capture to directory with native filename.
        cv2.imwrite(f'{filename}.jpg', frame)

        # Once new capture is written, check if file limit has been exceeded and remove older captures to avoid resource exhaustion.
        self.file_handling.check_file_exhaustion(directory, self.file_handling.MAXIMUM_FILES_STORED)


    def process_frames(self, frame):

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

        # Iterate over each detection in the list provided.
        for detection in detections:

            # Accumulate detection data.
            x, y, w, h, ID, threat_level = detection

            # Associate visualiation colour with the detections threat level.
            if threat_level == 1:
                detection_colour = self.threat_levels[threat_level]
            elif threat_level == 2:
                detection_colour = self.threat_levels[threat_level]
            elif threat_level == 3:
                detection_colour = self.threat_levels[threat_level]
            else: 
                # default error colour. 
                detection_colour = (0, 0, 0)

            # Calculate a detections center points. 
            center_point_x = ((x * 2) + w) // 2
            center_point_y = ((y * 2) + h) // 2

            # Use OpenCv to draw onto the frame. 
            cv2.putText(
                img=frame,
                text=f'ID: {ID}', 
                org=(center_point_x - 12, center_point_y - 12),
                fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=3, 
                color=detection_colour,
                thickness=3
            )
            cv2.rectangle(
                img=frame,
                pt1=(x, y), 
                pt2=(x + w, y + h), 
                color=detection_colour,
                thickness=2
            )
            cv2.circle(
                img=frame, 
                center=(center_point_x, center_point_y),
                radius=4, 
                color=detection_colour, 
                thickness=-1
            )      

        # Return processed frame and detections threat_level.
        return frame, threat_level
    
    
    def motion_detection(self, prev_frame, curr_frame):

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
        if (np.sum(thresholded_frame_pixels) / 10000) > 1800:

            # Set motion_detected boolean value to true.
            motion_detected = True 

        return motion_detected


    def register_detections(self, frame, threshold = 3400):

        detections = []

        # Check frames passed are not None Type. Raise exception if they are. 
        if frame is None:
            return ValueError('Provided frames were returned as None!')

        processed_frame = self.process_frames(frame)

        _, masked_frame = cv2.threshold(
            processed_frame, 
            254,
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

            if contour_area > threshold:

                # Compute the bounding box data for that contour.
                x, y, w, h = cv2.boundingRect(contour)

                # Append the data including size and coordinates to the list. 
                detections.append( [x, y, w, h] )

        return frame, detections
