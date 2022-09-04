from datetime import datetime
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import model as mdl
import json
import numpy as np
from waitress import serve 
from transformers import AutoModel, AutoTokenizer 
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
import torch


app = Flask(__name__)
# MODEL_PATH = "/src/model_files"

#load the banking model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_xlmr_banking = AutoTokenizer.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')
model_xlmr_banking = AutoModel.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')
model_xlmr_banking = model_xlmr_banking.to(device)

model_xlmr_banking_seq = AutoModelForSequenceClassification.from_pretrained('nickprock/xlm-roberta-base-banking77-classification')
model_xlmr_banking_seq = model_xlmr_banking_seq.to(device)
pipe_xlmr_banking = pipeline("text-classification", model_xlmr_banking_seq, tokenizer=tokenizer_xlmr_banking)

#load the smalltalk model
classifier_smallTalk = pipeline("zero-shot-classification",
                    model="joeddav/xlm-roberta-large-xnli")



@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    lang = data['lang']
    text = data['txt']
    prediction = mdl.predict(text, lang, model_xlmr_banking, classifier_smallTalk,
                             tokenizer_xlmr_banking, device, pipe_xlmr_banking)
    # output = prediction
    return jsonify(
        score = prediction[1],
        intent = prediction[0],
        ranked_intents = prediction[2]
    )


@app.route('/')
def index():
   print('Request for index page received')
   return "Hello NLP Demo"
#    return render_template('index.html')
    

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
    print("server is up and running!")
    serve(app, host="0.0.0.0", port=8080)
#    app.run(host="0.0.0.0", port=8080)
