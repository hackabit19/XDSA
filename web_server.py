from flask import Flask, render_template, Response , json
import cv2
from captionbot import CaptionBot
import numpy as np

app = Flask(__name__)
frame = None

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

def gen():
    cap = cv2.VideoCapture(1)
    global frame
    while True:
        ret ,frame =  cap.read()
        if ret == True:
            flag, encodedImage = cv2.imencode(".jpg" , frame)
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

@app.route('/generate_caption')
def generate_caption():
    c = CaptionBot()
    print("Generating Caption...")
    global frame
    cv2.imwrite('image.jpg' , frame)
    # caption = c.file_caption('/home/aditya/Hack-a-bit2019/' + 'image.jpg')
    caption = c.file_caption("C:/Users/Bharat/Desktop/Hack-A-Bit 2019/image.jpg")
    print(caption)
    res = {'caption':caption}
    return json.dumps(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
