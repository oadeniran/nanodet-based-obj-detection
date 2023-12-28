from flask import Flask, render_template, request, Response, redirect, jsonify
from first import make_inference_img, make_inference_frame
import os
import cv2
import json
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = "img." + file.filename.rsplit('.', 1)[1].lower()
        file.save(filename)
        make_inference_img(filename)
        return render_template("show-img.html", image_path = "inf-img.jpg")
    else:
        return 'Invalid file format!'

@app.route('/frame-collect', methods=['POST'])
def upload_frame():
    frame_j = request.form['frame']
    frame_i = np.array(json.loads(frame_j))
    return jsonify(make_inference_frame(frame_i))

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug=True, port = 7000)
