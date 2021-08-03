from flask import Flask, render_template, request
import torch
from glob import glob
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():

    # Get previous prediction and loaded image paths
    previous_prediction_path = glob('static/*')
    previous_image_path = glob('images/*')

    print(f'Previous path = {previous_image_path}')
    print(f'Previous prediciton path = {previous_prediction_path}')

    # clear directory of old predictions and images to save storage
    if len(previous_image_path) != 0:
        if os.path.isfile(previous_image_path[0]):
            os.remove(previous_image_path[0])

    if len(previous_prediction_path) != 0:
        if os.path.isfile(previous_prediction_path[0]):
            os.remove(previous_prediction_path[0])

    # save image from input into images directory
    image_file = request.files['imagefile']
    image_path = "./images/" + image_file.filename
    image_file.save(image_path)

    # Load Model and make prediction
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    results = model(image_path)
    results.save() # or .show(), .save(), .crop(), .pandas(), etc.

    path = glob('runs/detect/exp/*')[0]
    renamed_path = 'static/' + path.split('/')[-1]
    print(renamed_path)
    os.rename(path, renamed_path)

    os.rmdir('runs/detect/exp')

    return render_template('index.html', prediction_path=renamed_path)

if __name__ == '__main__':
    app.run(port = 3000, debug = True)