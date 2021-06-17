from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = tf.keras.models.load_model('classification-model')


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def upload_file():
    text = (str(request.args.get('text', '')))
    predictions = model.predict(np.array([text]))
    acc = predictions[0][0].item()
    return jsonify(result=True if acc > 0 else False, acc=acc, text=text)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
