## trying scripting approach since classes muck about with accessing cap obj. 

from flask import Flask, Response, render_template, request, redirect, url_for
import cv2 

app : object = Flask(__name__)

capture = cv2.VideoCapture(0)


@app.route('/')
def index() -> str:

    '''
    Application landing page where usesr can view the stream with basic application metrics.
    
    :return: Render template returns the homepage with the html template, title and application info dictionary appended.
    '''

    # Call render template function.
    return render_template(
        'index.html', 
        title = 'Welcome to the app!', 
    )


@app.route('/video_stream')
def video_stream() -> Response:

    '''
    Real-time video streaming achieved by a multipart response, providing multiple frames with one HTTP response.
    :return: Stream of frames.
    '''

    # Call response object, accessing the camera and its settings the content type.
    return Response(
        # Call stream_frames function.
        read_frames_CV2(),
        # Set content type argument. 
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


def read_frames_CV2():

    while True:

        ret, frame = capture.read()

        if not ret:
            # Check video stream can be accessed properly. 
            raise RuntimeError('Failed to access onboard camera!')
        
        # Convert frame channels for linux OS. 
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

         # Encode the frames and check the operation has been succesful with return bool.
        ret, jpeg = cv2.imencode('.jpg', frame)

        # If nothing returned after operation, inform user. 
        if not ret:
            raise RuntimeError('Frames could not be converted!')
        
    
        # Yielded sequence of the encoded frames as response chunks for the stream.
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
        )


if __name__ == '__main__' : 

    app.run(debug=True)


