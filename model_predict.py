from inspect import getclasstree, getmodule
from posixpath import basename
from transformers import pipeline
import torch
import pickle
import pandas as pd
import numpy as np
import operator
import mysqlconnector as mysqlconn


intents_df = None
conf_df = None


def load_intents():
    global intents_df
    try:
        query = "SELECT * FROM intents;"
        intents_df = mysqlconn.read_df_sqlalchemy(query)
        print(intents_df.head())
        return intents_df
    except mysqlconn.Error as error:
        print("database error occured: {}".format(error))


def load_config():
    global conf_df
    try:
        query = "SELECT * FROM configs;"
        conf_df = mysqlconn.read_df_sqlalchemy(query)
        print(conf_df.head())
        return conf_df
    except mysqlconn.Error as error:
        print("database error occured: {}".format(error))


def getConfigValue(confKey):
    confValue = conf_df.loc[conf_df['config_key'] == confKey, 'config_value'].values[0]
    return confValue


def getIntent(label):
    intent = intents_df[intents_df.label == label].intent.item()
    return intent


    # load the model from disk
def getClassifier(lang="all"):
    model_file = "./Data/UDModel_banks.sav"
    loaded_model = pickle.load(open(model_file, 'rb'))
    return loaded_model


def predict_user(pred_text, tokenizer_xlmr_banking, model_xlmr_banking, device):
    # Tokenize the input
    pred_inputs = tokenizer_xlmr_banking(pred_text, return_tensors="pt")
    pred_inputs = {k:v.to(device) for k,v in pred_inputs.items()}

    with torch.no_grad():
        pred_hidden_state = model_xlmr_banking(**pred_inputs).last_hidden_state[:,0].cpu().numpy()

    # get the classifier and predict
    clfr = getClassifier()
    pred = clfr.predict(pred_hidden_state), clfr.predict_proba(pred_hidden_state)
    intent = getIntent(pred[0][0])
    score = np.max(pred[1])
    return intent, score


def predict_dsl(pipeline, predtext):
    pipepred = pipeline(predtext)
    intent = pipepred[0]['label']
    score = pipepred[0]['score']
    return intent, score


def predict_smallTalk(predtext, classifier):
    candidate_labels = ["greeting", "general inquiry"]
    pred = classifier(predtext, candidate_labels)
    score = np.max(pred["scores"])
    idx = pred["scores"].index(score)
    intent = pred["labels"][idx]
    return intent, score


def predict(pred_text, lang, model_banking, classifier_smallTalk, 
            tokenizer_banking, device, pipe_xlmr_banking):

    load_intents()
    load_config()
    
    preds = {}
    threshold = float(getConfigValue("threshold"))
    threshold_udml = float(getConfigValue("threshold-UDML"))
    threshold_dsl = float(getConfigValue("threshold-DSL"))
    threshold_smalltalk = float(getConfigValue("threshold-smallTalk"))
    score = 0.0
    intent = ""

    # Prediction Pipeline
    #=========================
    ### User Defined Intents classification
    #=========================#=========================
    intent_udml, score_udml = predict_user(pred_text, tokenizer_banking, model_banking, device)
    preds[intent_udml] = float(score_udml)

    ### Domain Specific prediction (Banking)
    #=========================#=========================
    intent_dsl, score_dsl = predict_dsl(pipe_xlmr_banking, pred_text)
    preds[intent_dsl] = float(score_dsl)

    ### Small and generic talk (XLMR Facebook multilingual model)
    #=========================#=========================
    intent_smalltalk, score_smalltalk = predict_smallTalk(pred_text, classifier_smallTalk)
    preds[intent_smalltalk] = float(score_smalltalk)

    intent = max(preds.items(), key=operator.itemgetter(1))[0]
    score = preds[intent]

    ### Score
    if score < threshold:
        intent = "fallback"
        score = 0.0
    elif score_smalltalk > threshold_smalltalk:
        intent = intent_smalltalk
        score = score_smalltalk

    return intent, score, preds
