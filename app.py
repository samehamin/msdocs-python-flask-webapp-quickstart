from flask import Flask, request, jsonify
from waitress import serve 
import model_predict as mdl_predict
import model_train as mdl_train
import globals_nlp


app = Flask(__name__)


@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    lang = data['lang']
    text = data['txt']

    # predict intents
    intents = mdl_predict.predict(text, lang, globals_nlp.model_xlmr_banking, globals_nlp.classifier_smallTalk,
                             globals_nlp.tokenizer_xlmr_banking, globals_nlp.device, globals_nlp.pipe_xlmr_banking)

    # predict entities 
    ners = mdl_predict.predictNER(text)
    
    # output = prediction
    return jsonify(
        score = intents[1],
        intent = intents[0],
        ranked_intents = intents[2],
        entities = ners
    )


@app.route('/predictNER',methods=['POST'])
def predictNER():
    data = request.get_json(force=True)
    # lang = data['lang']
    text = data['txt']
    prediction = mdl_predict.predictNER(text)
    # output = prediction
    return jsonify(
        prediction
    )


@app.route('/train',methods=['POST'])
def train():
    mdl_train.train_classifier()
    return jsonify(
        status = "200"
    )


if __name__ == '__main__':
    print("initializing ..")
    globals_nlp.init()
    print("server started ..")

    # run for production 
    # serve(app, host="0.0.0.0", port=8080)
    # run for dev 
    app.run(host="0.0.0.0", port=8080)


