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
    # candidate_labels = ["greeting", "general inquiry"]
    smalltalk_df = load_small_talk_intents()
    candidate_labels = smalltalk_df['small_talk_intent'].tolist()
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
    # preds[intent_smalltalk] = float(score_smalltalk)

    max_intent = max(preds.items(), key=operator.itemgetter(1))[0]
    max_score = preds[max_intent]

    ### Score
    # if score_smalltalk > threshold_smalltalk:
    #     intent = intent_smalltalk
    #     score = score_smalltalk
    # elif score_dsl > threshold_dsl:
    #     intent = intent_dsl
    #     score = score_dsl
    # elif score_udml > threshold_udml:
    #     intent = intent_udml
    #     score = score_udml
    # elif score <= threshold:
    #     intent = "fallback"
    #     score = 0.0

    if max_score > threshold:
        intent = max_intent
        score = max_score
        
    return intent, score, preds


def predictNER(predText):
    pip_ner = pipeline("ner", model=globals_nlp.clfr_NER, tokenizer=globals_nlp.tokenizer_NER)
    ner_results = pip_ner(predText, aggregation_strategy="simple")
    return str(ner_results)
