from copyreg import pickle
import os
import io
from pathlib import Path

from flask import Flask, render_template, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
from commons import transform_image

from inference import get_prediction
from commons import transform_image, format_class_name
import tensorflow as tf
import pickle
import cv2
import numpy as np
import pandas as pd

# load Base Dir for easy absolute pathfinding
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = f'{BASE_DIR}/interface/uploads/images'

# load model
MODEL = pickle.load(open(f'{BASE_DIR}/model.pkl', 'rb'))

# allowed types for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

##############################################################################
########################### FLASK APP ########################################
##############################################################################

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/uploads/images/<path:path>")
def static_dir(path):
    return send_from_directory("uploads/images/", path)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # remove unwanted files from the upload folder
    # deletes files from uploads when viewer reloads the page
    # DO NOT PUSH THIS IN PRODUCTION, this is for test purposes only
    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))

    if request.method == 'POST':
        # check if the post request has the file part
        if 'files' not in request.files:
            flash('No file part')
            return redirect(request.url)

        
        files = request.files.getlist('files')
        # if user does not select file, browser also
        # submit an empty part without filename

        df = {'filename':[],'class_name': [], 'class_id': []}
        for file in files:

            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
        
            if file and allowed_file(file.filename):
                # load files and save them on upload folder to be viewed on results page
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                image_path = UPLOAD_FOLDER + '/' + filename
                image = transform_image(image_path)

                # Get the prediction
                class_name ,class_id = get_prediction(image_bytes=image, model=MODEL)

                # append the prediction to the dataframe for visualization
                df['filename'].append(filename)
                df['class_name'].append(class_name)
                df['class_id'].append(class_id)

        df = pd.DataFrame(df)
        return render_template('result.html', df=df)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
