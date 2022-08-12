from datetime import datetime
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import model as mdl
import json

app = Flask(__name__)


@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['txt']
    prediction = mdl.predict(text)
    output = prediction
    return jsonify(output)


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
   app.run(host="0.0.0.0", port=8080)
