from flask import Flask, Response, render_template, request, redirect, url_for
from typing import List, Dict, Generator, Tuple

from ObjectTracking import ObjectTracking
from ObjectDetection import ObjectDetection
from FileHandling import FileHandling
from Camera import Camera 

import os, time, threading, cv2
from datetime import datetime 


class App(object):

    ''' 
    Application class setup to handle all logic concerned with the applications operation. This includes the routes and the associated functionality within those pages. 
    '''
    
    def __init__(self, THREAT_LEVEL : int = 3) -> None:

        # Initalise Flask application object. 
        self.app : object = Flask(__name__)

        # Initialise Camera object for its methods and attributes.
        self.camera : object = Camera()

        self.object_detection = ObjectDetection()

        self.object_tracking = ObjectTracking()

        # Initialise FileHandling module to access functions to manage files in local storage
        self.file_handling : object = FileHandling()

        # Dictionary storing key value pairs representing applications current information.
        self.app_info : Dict[str, str] = {
            # Total sum of captures within the devices local storage.
            'total_captures' : 0,
            # Date of the most recent capture.
            'capture_date' : 'N/A',
            # Time of the most recent capture.
            'capture_time' : 'N/A',
            # Current status of the stream.
            'stream_status' : 'Inactive',
        }

        # Threat level required to take action.
        self.threat_level = THREAT_LEVEL

        '''
        Page routes, Functions to handle page logic.
        '''


        @self.app.route('/')
        def index() -> str:

            '''
            Application landing page where usesr can view the stream with basic application metrics.
            
            :return: Render template returns the homepage with the html template, title and application info dictionary appended.
            '''

            # Update index with the current length of stored images.
            self.app_info['no_of_captures'] = str(len(self.file_handling.stored_images))
            # Update index with last sorted capture, returning the first result. 
            self.app_info['capture_date'] =  self.file_handling.sort_files(self.file_handling.stored_images)[0]['capture_date'] 
            self.app_info['capture_time'] =  self.file_handling.sort_files(self.file_handling.stored_images)[0]['capture_time'] 
            # Update index with the current status of the camera. 
            self.app_info['device_status'] = 'Active' if self.camera.settings['camera_toggle'] == True else 'Inactive'

            # Call render template function.
            return render_template(
                'index.html', 
                # app_info dictionary for template to access data stored as key value pairs. 
                app_info = self.app_info
            )
        

        @self.app.route('/video_stream')
        def video_stream() -> Response:

            '''
            Real-time video streaming achieved by a multipart response, providing multiple frames with one HTTP response.
            :return: Stream of frames.
            '''

            # Call response object, accessing the camera and its settings the content type.
            return Response(
                # Call stream_frames function.
                App.stream_frames(
                    self,
                    self.camera,
                    self.object_detection,
                    self.object_tracking
                ),
                # Set content type argument. 
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )


        @self.app.route('/captures', methods = ['GET', 'POST'])
        def captures() -> str:

            '''
            Page where all stored captures can be viewed and accessed. 
            
            :return: render captures stored within image_stored onto the page. 
            '''

            # If form submission is made containing the sort_oder for files.
            if request.form.get('sort_order'):
                
                # Negate current file order boolean value.              
                self.file_handling.file_order = not self.file_handling.file_order

                # Redirect users back to settings page with changes appended. 
                return redirect(url_for('captures'))

            # Sort images stored list using the order provided. 
            sorted_images = self.file_handling.sort_files(self.file_handling.stored_images, reverse_order = self.file_handling.file_order)

            # Call function to manage the captures displayed content. Pass 12 as the maximum number of images argument. 
            current_images, total_pages, page_number = self.file_handling.manage_images_displayed(12, sorted_images)
      
            # Call render template function.
            return render_template(
                'captures.html',
                title = 'Captures' if len(self.file_handling.stored_images) > 1 else 'No Captures Yet :(',
                image = current_images,
                total_pages = total_pages,
                current_page = page_number,
                order = self.file_handling.file_order,
            )
        

        @self.app.route('/captures/delete/<filename>', methods = ['POST'])
        def delete_capture(filename) -> str:

            '''
            Application route to handle the deletion of captures, giving users the freedom to control what is stored on their device. 

            :param filename: Specified file to be deleted. 
            '''

            # Get the current captures directory. 
            directory = self.file_handling.CAPTURES_DIRECTORY

            # Construct filepath from parameterised filename. 
            capture = f'{directory}{filename}.jpg'

            # Check path exists before moving forward.
            if os.path.exists(capture):

                # Use os library to remove file from the devices local storage. 
                os.remove(capture)

                # Redirect users back to captures page. 
                return redirect(url_for('captures'))
            else:
                # If file not found, notify user. 
                return 'Resource not found!', 404


        @self.app.route('/settings')
        def settings() -> str:

            '''
            Users can fine tune the computer vision algorithms functionality,
            making it more appropriate for their specific use cases.
            
            :return: Render the settings page with the current settings stored. 
            '''

            # Call render template function.
            return render_template(
                'settings.html',
                title = 'Settings',
                settings = self.camera.settings,
            )
        

        @self.app.route('/settings/update', methods=['POST'])
        def update_device_settings() -> Response:

            '''
            Process form submissions recieved from the settings page, whether a button, slider or drop down. Apply updated values to the device.

            :return: Redirect the user back to settings page, should make experience seamless.
            '''

            # If the value returned is a button.
            if 'toggle' in request.form:

                # Grab the buttons name.
                btn_name = request.form.get('toggle')
                # Construct button name by appending _toggle designator.
                target_btn = f'{btn_name}_toggle'

                # Retrieve current status, negate current values.
                self.camera.settings[target_btn] = not self.camera.settings[target_btn]

            # If the value returned is a slider.
            elif 'slider' in request.form:
                
                # Grab the sliders current value.
                value = request.form.get('slider')

                # Grab the sliders name.
                name = request.form.get('slider_name')
                
                # Apply updated value to the settings dictionary. 
                self.camera.settings[name] = int(value)

            # If the value returned is a drop down.
            elif 'drop' in request.form:
                
                # Get the drop down name.
                drop_name = request.form.get('drop_name')

                # Get the option selected by the user. 
                value = request.form.get('drop')

                # Apply the value to settings dictionary. 
                self.camera.settings[drop_name] = int(value)

            # Redirect the user back to the settings page to make experience seamless.
            return redirect(url_for('settings'))

    
    '''
    Functions separating page logic from the application routes for increased maintainability.
    '''

    
    def stream_frames(self, camera : Camera, object_detection : ObjectDetection, object_tracking : ObjectTracking):
        
        '''
        Retrieve frames from the devices onboard camera, encodes them into response chunks to upload for streaming.
        '''
        # Toggle for the camera on/off.
        camera_toggle = camera.settings['camera_toggle']

        # Initialise previous frame variable, store first frame when loading to avoid errors.
        previous_frame = camera.retrieve_frame_CV2()

        while camera_toggle: 

            # Initialise timer used to enforce stream framerate. 
            fps_timer_start = time.time()
            
            # Retrive the current, untampered frame from the devices onboard camera.
            raw_frame = camera.retrieve_frame_CV2()

            motion_detected = object_detection.motion_detection(previous_frame, raw_frame, camera)

            frame, detections = object_detection.register_detections(raw_frame, camera)

            updated_detections = object_tracking.update_detections_V3(detections)

            detection_frame, threat_level = object_detection.draw_bounding_boxes(frame, updated_detections)

            # Access current time, formatted to display on the video stream.
            current_time = datetime.now().strftime('%I:%M:%S%p')
            # Append that time by drawing the clock onto the frame.
            time_layer = camera.draw_clock(detection_frame, current_time)
            # Layer clock frame over the current frame, mitigates interference for bounding boxes.
            appended_frame = cv2.addWeighted(detection_frame, 0.5, time_layer, 0.5, 0)

            # Encode the frame into bytes for streaming.
            encoded_frame = camera.encode_frame(appended_frame)

            if motion_detected == True and threat_level == self.threat_level:
            
                # Capture that specific frame where motion has been detected.
                object_detection.capture_frame(
                    detection_frame, 
                    './static/captures/', 
                )   

            # Yielded sequence of the encoded frames as response chunks for the stream.
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n'
            )

            # Update the previous frame with the raw onboard camera frame.
            previous_frame = raw_frame

            # Reset the elapsed time for the fps timer.
            elapsed_time = time.time() - fps_timer_start

            # Enforce stream framerate. 
            camera.enforce_frame_rate(elapsed_time)

    
    def run_app(self) -> None:

        '''
        Function called to start the application and notify users it is loading.

        :return: N/A
        '''
        
        # Inform users the application is currently starting up.
        print('Application is loading!')

        # Initialise the flask application and run.
        self.app.run(
            host='0.0.0.0'
        )


if __name__ == '__main__':

    '''
    Main method. Start application and its threads.
    '''
    
    # Instantiate the application object to access its methods. 
    application = App()

    # Access files in devices local storage whilst application loads using target directory. 
    application.file_handling.access_stored_captures(application.file_handling.CAPTURES_DIRECTORY)

    # Create a thread to run the cameras streaming functionality in concurrency with the rest of the application.
    camera_stream_thread = threading.Thread(
        target=application.stream_frames,
        args=(
            application.camera,
            application.object_detection,
            application.object_tracking,
        )
    )
    camera_stream_thread.isDaemon = True
    camera_stream_thread.start()

    # Run the application. 
    application.run_app()