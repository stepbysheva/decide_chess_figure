import os

from flask import Flask, render_template, request
import tensorflow as tf

from classifier import classify

app = Flask(__name__)
cnn_model = tf.keras.models.load_model("static/save_at_20.h5")

@app.route('/')
def hello_world():# put application's code here
    return render_template("index.html")


@app.post('/')
def return_probability():
    file = request.files["image"]
    file_path = os.path.join("static/uploads/" + file.filename)
    file.save(file_path)
    prob = classify(cnn_model, file_path)

    probability = round((prob * 100), 2)
    return {
        "prob": probability
    }


if __name__ == '__main__':
    app.run()
