from inspect import getclasstree, getmodule
from posixpath import basename
import requests
import json
from transformers import AutoModel, AutoTokenizer 
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
import torch
import pickle
import pandas as pd
import numpy as np
import operator




config_df = pd.read_csv("Data/config.csv")
    # # confValue = config_df["confKey"==confKey].item()
    # return config_df
labels_df = pd.read_csv("Data/intents.csv")


def getConfigValue(conf_df, confKey):
    confValue = conf_df.loc[conf_df['confKey'] == confKey, 'confValue'].values[0]
    return confValue


def getIntent(label):
    intent = labels_df[labels_df.label == label].intent.item()
    return intent


def getClassifier(lang="all"):
    # load the model from disk
    foldername = "./Data"
    filename = ""
    
    if lang=="ar":
        filename = foldername + '/intent-classification-ar.sav'
    elif lang=="en":
        filename = foldername + '/intent-classification-en.sav'
    elif lang=="all":
        filename = foldername + "/UDModel_banks.sav"

    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


# def getModel(device, model_ckpt):
#     model = AutoModel.from_pretrained(model_ckpt)
#     model = model.to(device)


def predict_user(pred_text, tokenizer_xlmr_banking, model_xlmr_banking, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenize the input
    pred_inputs = tokenizer_xlmr_banking(pred_text, return_tensors="pt")
    pred_inputs = {k:v.to(device) for k,v in pred_inputs.items()}
    
    # get the hidden state
    # model = getModel(device, model_ckpt)
    # model = AutoModel.from_pretrained(model_ckpt)
    # model = model.to(device)

    with torch.no_grad():
        pred_hidden_state = model_xlmr_banking(**pred_inputs).last_hidden_state[:,0].cpu().numpy()

    # get the classifier and predict
    clfr = getClassifier()
    pred = clfr.predict(pred_hidden_state), clfr.predict_proba(pred_hidden_state)
    intent = getIntent(pred[0][0])
    score = np.max(pred[1])
    return intent, score


def predict_dsl(pipeline, predtext):
    # model_ckpt = 'nickprock/xlm-roberta-base-banking77-classification'
    # model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
    # pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    pipepred = pipeline(predtext)
    intent = pipepred[0]['label']
    score = pipepred[0]['score']
    return intent, score


def predict_smallTalk(predtext, classifier):
    # classifier = pipeline("zero-shot-classification",
    #                 model="joeddav/xlm-roberta-large-xnli")
    candidate_labels = ["greeting", "general inquiry"]
    pred = classifier(predtext, candidate_labels)
    score = np.max(pred["scores"])
    idx = pred["scores"].index(score)
    intent = pred["labels"][idx]
    return intent, score


def predict(pred_text, lang, model_banking, classifier_smallTalk, 
            tokenizer_banking, device, pipe_xlmr_banking):

    # conf_df = getConfig()

    model_ckpt = ""
    if lang == "ar":
        #define the tokenizer
        model_ckpt = "aubmindlab/bert-base-arabertv2"
    elif lang=="en":
        model_ckpt = "distilbert-base-uncased"
    elif lang=="all":
        model_ckpt = 'nickprock/xlm-roberta-base-banking77-classification'

    # tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    preds = {}
    threshold = 0.0
    score = 0.0
    intent = ""

    # Prediction Pipeline
    #=========================
    ### User Defined Intents classification
    #=========================#=========================
    intent_udml, score_udml = predict_user(pred_text, tokenizer_banking, model_banking, device)
    preds[intent_udml] = float(score_udml)
    # threshold = int(getConfigValue(conf_df, "threshold-UDML"))
    # if score >= threshold:
    #     return intent, score

    ### Domain Specific prediction (Banking)
    #=========================#=========================
    intent_dsl, score_dsl = predict_dsl(pipe_xlmr_banking, pred_text)
    preds[intent_dsl] = float(score_dsl)
    # threshold = int(getConfigValue(conf_df, "threshold-DSL"))
    # if score >= threshold:
    #     return intent, score 

    ### Small and generic talk (XLMR Facebook multilingual model)
    #=========================#=========================
    intent_smalltalk, score_smalltalk = predict_smallTalk(pred_text, classifier_smallTalk)
    preds[intent_smalltalk] = float(score_smalltalk)
    # threshold = int(getConfigValue(conf_df, "threshold-smallTalk"))
    # if score >= threshold:
    #     return intent, score

    ### Score
    # intent = max(preds)
    if score < threshold:
        intent = "fallback"
        score = 0.0
    elif score_smalltalk > 0.8:
        intent = intent_smalltalk
        score = score_smalltalk
    else:    
        intent = max(preds.items(), key=operator.itemgetter(1))[0]
        score = preds[intent]
        threshold = getConfigValue(config_df, "threshold")

    return intent, score, preds
