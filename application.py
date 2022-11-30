import numpy as np
import tensorflow as tf
import cv2
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
    dis = {0: "Nail Infection", 1: "Viral Infection", 2: "Melanoma"}
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        img_filename = secure_filename(file.filename)

        #Image resizing
        file.save(img_filename)
        img = cv2.imread(img_filename)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(img_filename, img)
        # cv2.imwrite("static/files/"+img_filename, img)

        #Giving to tf for processing
        data = tf.keras.utils.img_to_array(img)

        #Rescaling
        resc = lambda x: (x / 255)
        f = np.vectorize(resc)
        data = f(data)
        data = np.expand_dims(data, axis = 0)

        print("Scaling: ", data)
        print(data.shape)

        #predicting
        m = tf.keras.models.load_model('model_10.h5')
        res = m.predict(data)
        ind = np.argmax(res[0])
        ires = dis[ind]
        print("Index of the maximum value: ", ires)

        d1 = "Melanoma: " + str(round(res[0][0] * 100)) +"%"
        d2 = "Nail Fungus: " + str(round(res[0][1] * 100)) +"%"
        d3 = "Viral Infection: " + str(round(res[0][2] * 100)) +"%"



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
        return render_template("upload_result.html",form = form, result=ires, d1 = d1, d2 = d2, d3 = d3)
    return render_template("index.html", form=form)

if __name__ == '__main__':
    application.run(debug=True)
