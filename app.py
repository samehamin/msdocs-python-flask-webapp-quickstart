from flask import Flask, request, jsonify
import model_predict_embedding as mdl_predict
import model_train_embedding as mdl_train
import globals_nlp


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    lang = data['lang']
    text = data['txt']

    # predict intents
    intent, score, ranked_intents = mdl_predict.predict(text)

    # predict entities 
    ners = mdl_predict.predictNER(text)
    
    # output = prediction
    return jsonify(
        intent = intent,
        score = str(score),
        ranked_intents = ranked_intents.to_json(orient='index'),
        # entities = ners
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
