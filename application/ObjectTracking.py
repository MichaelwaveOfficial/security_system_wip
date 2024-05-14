from typing import List, Tuple, Dict
import math, time, cv2, numpy as np


class ObjectTracking(object):

    '''
    Class to seperate and handle logic for identifying and keeping track of objects. 
    '''

    def __init__(self, EUCLIDEAN_DISTANCE_THRESHOLD : int = 225, MAXIMUM_THREAT_LEVEL : int = 3, DEREGISTRATION_TIME : int = 10, ESCALATION_TIME : int = 10) -> None:
        
        # Dictionary to hold detections data which can be used for IDs, bounding boxes and center points. 
        self.detection_center_points : Dict[int, Tuple[int, int]] = {}

        # Dictionary to store the detections ID and the time it was last seen.
        self.last_detected = {}

        # Assign unique ID values to each detection.
        self.ID_increment_counter : int = 0

        # Time taken for a detection to be deregistered.
        self.DEREGISTRATION_TIME = DEREGISTRATION_TIME

        # Minimum number of pixels between each center point before they are classed as new detections. 
        self.EUCLIDEAN_DISTANCE_THRESHOLD = EUCLIDEAN_DISTANCE_THRESHOLD

        # Dictionary to store detection ID accompanied by its threat level. 
        self.detection_threat_level : Dict[int, int] = {}
        
        # Dictionary storing detection ID and last time its threat level was escalated.
        self.last_increments : Dict[int, float] = {}

        # Maximum threat level allowed.
        self.MAXIMUM_THREAT_LEVEL = MAXIMUM_THREAT_LEVEL

        # Time taken to escalate a detections threat level. 
        self.ESCALATION_TIME = ESCALATION_TIME

        self.kf_filter = cv2.KalmanFilter(4, 2)  # State vector size is now 8, Measurement vector size is 4

        self.kf_filter.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)

        self.kf_filter.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)

        self.kf_filter.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32) * 0.05


    '''
    Functions to handle the registering, deregistering and tracking of object detections. 
    '''
    
    
    def update_detections_V3(self, detections : List[Tuple[int, int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:

        '''
        Accepts a list of data concerned with the detections and their bounding box data. This will be used to calculate the Euclidean Distance (straight line distance) between
        one detections center point and another detections center point to determine whether or not the difference between these points from separate frames can be 
        classed as the same object/detection. If the distance is greater than the supplied threshold, it can be classed as a separate object. 

        :param: detections - List of detections data (x, y, w, h)
        :return: bounding_boxes - Updated list of detections data (x, y, w, h)
        '''

        intial_time : float = time.time()

        # Initialise list storing bounding box data. 
        bounding_boxes : List[Tuple[int, int, int, int]] = []

        # Iterate over detections parameterised. 
        for detection in detections:
            
            # Initialise boolean variable to determine whether or not a detection has already been accounted for. 
            already_detected : bool = False 

            # Grab bounding box variables for calculations.
            x, y, w, h = detection

            # Calculate a detections center points. 
            center_point_x : float = (x * 2 + w) // 2
            center_point_y : float = (x * 2 + h) // 2

            # Iterate over items stored within the detections dictionary. 
            for detection_ID, center_point in self.detection_center_points.items():
                
                # Calculate the straight line distance between two points. 
                euclidean_distance : float = math.hypot(center_point_x - center_point[0], center_point_y - center_point[1])

                # If that distance falls within the set threshold.
                if euclidean_distance < self.EUCLIDEAN_DISTANCE_THRESHOLD:
                    
                    # Determine whether that detection exists already or not.
                    self.detection_center_points[detection_ID] = (center_point_x, center_point_y)

                    # Append last time detection was seen to the dictionary with its ID.
                    self.last_detected[detection_ID] = intial_time, x, y, w, h

                    # Check detections current time exceeds the threat escalation timer before its level can be raised.
                    if intial_time - self.last_increments.get(detection_ID, 0) > self.ESCALATION_TIME:

                        # Append detection ID and its current threat level to the dictionary. 
                        self.detection_threat_level[detection_ID] = min(self.detection_threat_level.get(detection_ID, 0) + 1, self.MAXIMUM_THREAT_LEVEL)

                        # Set time detections threat level was last escalated.
                        self.last_increments[detection_ID] = intial_time

                    # Update the bounding_box list with current data. 
                    bounding_boxes.append([x, y, w, h, self.detection_threat_level[detection_ID]])

                    # Detection has been handled, set its status as already_detected to True. 
                    already_detected = True

                    # Break out of the loop.
                    break
            
            # If a detections status as already_detected is False. 
            if not already_detected:

                # Assign ID and center point values to that detection.
                self.detection_center_points[self.ID_increment_counter] = (center_point_x, center_point_y)

                # Attribute detections last seen time.
                self.last_detected[self.ID_increment_counter] = intial_time, x, y, w, h

                # Initalise the threat level for that detection.
                self.detection_threat_level[self.ID_increment_counter] = 1

                self.last_increments[self.ID_increment_counter] = intial_time

                # Update the bounding_box list with current data.
                bounding_boxes.append([x, y, w, h, 1])

                # Increment the detections counter. 
                self.ID_increment_counter += 1
        
        # Initialise list to store deregistrations.
        deregistered_detections : List[int] = []
        
        # Iterate over detections and their attributes from the dictionary. 
        for detection_ID, last_seen in self.last_detected.items():

            last_timed, _, _, _, _ = last_seen

            # If the detection exceeds deregistration time.
            if intial_time - last_timed > self.DEREGISTRATION_TIME:
                
                # Append to deregistrations list. 
                deregistered_detections.append(detection_ID)

            else:
                # Otherwise decrease the threat level for active detections.
                self.detection_threat_level[detection_ID] = max(self.detection_threat_level.get(detection_ID, 1), -1, 1)

        # Iterate over each deregistration stored. 
        for deregistration_ID in deregistered_detections:
            
            # Remove the deregistrations from the dictionaries. 
            del self.detection_center_points[deregistration_ID]
            del self.last_detected[deregistration_ID]
            del self.detection_threat_level[deregistration_ID]
            del self.last_increments[deregistration_ID]
        
        # Return bounding_boxes list for later access. 
        return bounding_boxes
    

    def kf_predict(self, x, y):

        '''

        Estimate object position for smoother tracking.

        '''

        calculation = np.array([[ np.float32(x) ], [ np.float32(y)]])

        self.kf_filter.correct(calculation)

        prediction = self.kf_filter.predict()

        x, y = int(prediction[0]), int(prediction[1])

        return x, y
    