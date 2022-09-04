from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from waitress import serve 
import model_predict as mdl_predict
import model_train as mdl_train
import globals


app = Flask(__name__)


@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    lang = data['lang']
    text = data['txt']
    prediction = mdl_predict.predict(text, lang, globals.model_xlmr_banking, globals.classifier_smallTalk,
                             globals.tokenizer_xlmr_banking, globals.device, globals.pipe_xlmr_banking)
    # output = prediction
    return jsonify(
        score = prediction[1],
        intent = prediction[0],
        ranked_intents = prediction[2]
    )


@app.route('/train',methods=['POST'])
def train():
    mdl_train.train_classifier()
    return jsonify(
        status = "200"
    )


if __name__ == '__main__':
    globals.init()
    print("init completed..")
    print("server started ..")
    serve(app, host="0.0.0.0", port=5000)
#    app.run(host="0.0.0.0", port=8080)
