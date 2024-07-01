import os
import uuid
import flask
import urllib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import keras

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the custom object (FixedDropout) class
class FixedDropout(keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = keras.backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

# Register the custom object scope to handle 'FixedDropout' during model loading
with keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
    model = load_model(os.path.join(BASE_DIR, 'Model_Tanaman_Herbal_B7_V10.hdf5'))

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])
classes = ['Daun Jambu Biji', 'Daun Kari', 'Daun Kemangi', 'Daun Kunyit', 'Daun Mint', 'Daun Pepaya',
           'Daun Sirih', 'Daun Sirsak', 'Lidah Buaya', 'Teh Hijau']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def predict(filename, model):
    try:
        img = load_img(filename, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape(1, 224, 224, 3)
        img = img.astype('float32') / 255.0

        result = model.predict(img)

        # Get top 3 classes and their probabilities
        top_indices = np.argsort(result[0])[::-1][:3]
        class_result = [classes[idx] for idx in top_indices]
        prob_result = [result[0][idx] * 100 for idx in top_indices]

        predictions = {
            "class1": class_result[0],
            "prob1": round(prob_result[0], 2),
            "class2": class_result[1],
            "prob2": round(prob_result[1], 2),
            "class3": class_result[2],
            "prob3": round(prob_result[2], 2)
        }

        return predictions, None

    except Exception as e:
        error_msg = str(e)
        return None, error_msg


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/success', methods=['POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')

    if request.method == 'POST':
        if request.files:
            file = request.files['file']

            if file and allowed_file(file.filename):
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                file.save(img_path)

                predictions, error_msg = predict(img_path, model)

                if error_msg:
                    error = "Error processing the image."
                else:
                    return render_template('success.html', img=filename, predictions=predictions)

            else:
                error = "Please upload images of jpg, jpeg, and png extension only."

    return render_template('index.html', error=error)


if __name__ == "__main__":
    app.run(debug=True)
