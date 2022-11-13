import os

import numpy as np
import tensorflow as tf
import cv2
import keras
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

from wtforms.validators import InputRequired

application = Flask(__name__)
app = application
application.config['SECRET_KEY'] = 'supersecretkey'
application.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@application.route('/', methods=['GET',"POST"])
@application.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        img_filename = secure_filename(file.filename)

        #Image resizing
        file.save(img_filename)
        img = cv2.imread(img_filename)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(img_filename, img)

        #Giving to tf for processing
        data = tf.keras.utils.img_to_array(img)

        # data = tf.keras.utils.image_dataset_from_directory('data', image_size=(224, 224))
        # print("Processing:", data)

        #Rescaling
        resc = lambda x: (x / 255)
        f = np.vectorize(resc)
        data = f(data)
        data = np.expand_dims(data, axis = 0)

        # data = data.map(lambda x, y: (x / 255, y))

        print("Scaling: ", data)
        print(data.shape)

        #predicting
        m = tf.keras.models.load_model('skin')
        res = m.predict(data)
        print("Result: ", res)


        #filepath = os.path.join(img_filename)
        # Getting uploaded file name
        # filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),application.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        # file.save(filepath)
        # file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),application.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        # filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        # file.save(filepath)
        # Storing uploaded file path in flask session
        # uploaded_img_path = os.path.join(application.config['UPLOAD_FOLDER'], img_filename)


        # return render_template("uploaded_successfully.html",user_image = img_filename)
        return render_template("index.html",form = form, result=res)
    return render_template('index.html', form=form)

if __name__ == '__main__':
    application.run(debug=True)
