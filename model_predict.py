from inspect import getclasstree
from xmlrpc.client import MAXINT
from transformers import pipeline
import torch
import pickle
import pandas as pd
import numpy as np
import operator
import mysqlconnector as mysqlconn
import globals_nlp


intents_df = None
conf_df = None


def load_intents():
    global intents_df
    try:
        query = "SELECT * FROM intents;"
        intents_df = mysqlconn.read_df_sqlalchemy(query)
        # print(intents_df.head())
        return intents_df
    except mysqlconn.Error as error:
        print("database error occured: {}".format(error))


def load_config():
    global conf_df
    try:
        query = "SELECT * FROM configs;"
        conf_df = mysqlconn.read_df_sqlalchemy(query)
        # print(conf_df.head())
        return conf_df
    except mysqlconn.Error as error:
        print("database error occured: {}".format(error))


def load_small_talk_intents():
    smalltalk_df = None
    try:
        query = "SELECT * FROM intents_small_talk;"
        smalltalk_df = mysqlconn.read_df_sqlalchemy(query)
        # print(smalltalk_df.head())
        return smalltalk_df
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
    model_file = "./Data/UDModel.sav"
    loaded_model = pickle.load(open(model_file, 'rb'))
    return loaded_model


def predict(pred_text, tokenizer, model, device):

    load_intents()
    load_config()
    
    preds = {}
    threshold = float(getConfigValue("threshold"))
    # threshold_udml = float(getConfigValue("threshold-UDML"))
    # threshold_dsl = float(getConfigValue("threshold-DSL"))
    # threshold_smalltalk = float(getConfigValue("threshold-smallTalk"))
    score = 0.0
    intent = ""

    # Prediction Pipeline
    #=========================
    # SVM classifier prediction 
    pred_inputs = tokenizer(pred_text, return_tensors="pt")
    pred_inputs = {k:v.to(device) for k,v in pred_inputs.items()}
    
    with torch.no_grad():
        pred_hidden_state = model(**pred_inputs).last_hidden_state[:,0].cpu().numpy()

    clf_svm = getClassifier()
    pred = clf_svm.predict(pred_hidden_state), clf_svm.predict_proba(pred_hidden_state)
    max_score = np.max(pred[1])
    max_intent = getIntent(pred[0][0])
    # ranked_intents = pred[1]

    if max_score > threshold:
        intent = max_intent
        score = max_score
    else:
        intent = "fallback"
        score = "0.0"
        
    return intent, score, pred, max_intent, max_score


def predictNER(predText):
    pip_ner = pipeline("ner", model=globals_nlp.clfr_NER, tokenizer=globals_nlp.tokenizer_NER)
    ner_results = pip_ner(predText, aggregation_strategy="simple")
    return str(ner_results)


globals_nlp.init()
text = "i want to add new beneficiary account"
intent, score, preds, max_intent, max_score = predict(text, globals_nlp.tokenizer_xlmr_base, 
        globals_nlp.model_xlmr_base, globals_nlp.device)
print(intent)
print(score)
print(max_intent)
print(max_score)
# print(ranked_intents)
