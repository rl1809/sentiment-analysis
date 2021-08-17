from flask import Flask, request
from flask.json import jsonify

from preprocess import normalize_text
from model import get_model

app = Flask(__name__)
model = get_model()


@app.route('/sentiment/predict', methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    text = normalize_text(text)
    result = model.predict(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
