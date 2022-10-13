from inspect import getclasstree
from datasets import Dataset, concatenate_datasets
from transformers import pipeline
import torch
import pandas as pd
import numpy as np
import mysqlconnector as mysqlconn
import globals_nlp


intents_df = None
conf_df = None


def load_intents():
    print("loading the intents ...")
    global intents_df
    try:
        query = "SELECT * FROM intents;"
        intents_df = mysqlconn.read_df_sqlalchemy(query)
        # print(intents_df.head())
        return intents_df
    except mysqlconn.Error as error:
        print("database error occured: {}".format(error))


def load_config():
    print("loading config...")
    global conf_df
    try:
        query = "SELECT * FROM configs;"
        conf_df = mysqlconn.read_df_sqlalchemy(query)
        # print(conf_df.head())
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
    print("loading the embeddings...")
    embedding_chitchat_banking = Dataset.load_from_disk("models/embedding_chitchat/")
    embedding__user = Dataset.load_from_disk("models/embedding_user")
    embs_all = concatenate_datasets([embedding__user, embedding_chitchat_banking])
    return embs_all


def cls_pooling(model_output):
    # Extract the token embeddings
    return model_output.last_hidden_state[:, 0]


def embed_text(text_list):
    print("embedding the text ...")
    encoded_input = globals_nlp.tokenizer_xlmr_banking(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )

    encoded_input = {k: v.to(globals_nlp.device) for k, v in encoded_input.items()}
    model_output = globals_nlp.model_xlmr_banking(**encoded_input)
    return cls_pooling(model_output)


def predict(pred_text):
    print("prediction started...")
    load_intents()
    load_config()
    
    threshold = float(getConfigValue("threshold"))
    score = 0.0
    intent = ""
    k_labels = 5

    # Prediction Pipeline
    #=========================
    embedding_all = getClassifier()
    embedding_all.add_faiss_index("embedding")
    
    embedding_text = embed_text([pred_text]).cpu().detach().numpy()
    scores, samples = embedding_all.get_nearest_examples(
                                        "embedding", embedding_text, k=k_labels)
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    samples_df['intent'] = samples_df['label'].apply(lambda x: getIntent(x))

    # Scoring
    #=========================
    max_score = samples_df.scores[k_labels-1]
    max_intent = samples_df.intent[k_labels-1]
    ranked_intents = samples_df[["intent", "scores"]]
    # ranked_intents['scores'] = ranked_intents['scores'].apply(lambda x: str(x))

    if max_score > threshold:
        intent = max_intent
        score = max_score
    else:
        intent = "fallback"
        score = "0.0"
        
    return intent, score, ranked_intents


def predictNER(predText):
    print("NER started...")
    pip_ner = pipeline("ner", model=globals_nlp.clfr_NER, tokenizer=globals_nlp.tokenizer_NER)
    ner_results = pip_ner(predText, aggregation_strategy="simple")
    return str(ner_results)


# globals_nlp.init()
# text = "can I add new beneficiary account?"
# intent, score, ranked_intents = predict(text)
# print(f"Intent: {intent}")
# print(f"Score: {score}")
# print(f"Ranked intents: {ranked_intents}")
