from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = f'static/uploads/{file.filename}'
        file.save(file_path)

        img = image.load_img(file_path, target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        classes = {0: 'Bacterial leaf blight', 1: 'Brown spot', 2: 'Leaf smut'}
        predicted_class = classes[class_index]

        return render_template('index.html', file_path=file_path, predicted_class=predicted_class)

    return render_template('index.html', file_path=None, predicted_class=None)

if __name__ == '__main__':
    app.run(debug=True)
