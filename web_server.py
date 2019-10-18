from flask import Flask , render_template
import cv2
from captionbot import CaptionBot

app = Flask(__name__)
frame = None

@app.route('')
def index():
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(1)
    global frame
    while True:
        ret,frame = cap.read()
        if ret ==True:
            flag , encodedImage = cv2.imencode(".jpg" , frame)
            if not flag:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        else:
            continue
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
