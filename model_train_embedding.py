from datasets import Dataset, DatasetDict
import mysqlconnector as mysqlconn
import torch
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import globals_nlp
import pickle
from datasets import concatenate_datasets


def load_utterances():
    print("loading utterances...")
    
    dataset_utter = None

    try:
        # load utterances from db
        #TODO: create one table for train/test
        dataset_utter_train = mysqlconn.read_df_sqlalchemy("SELECT * FROM user_utterances_train;")
        dataset_utter_test = mysqlconn.read_df_sqlalchemy("SELECT * FROM user_utterances_test;")

        # convert dataframes into dataset
        ds_train = Dataset.from_pandas(dataset_utter_train)
        ds_test = Dataset.from_pandas(dataset_utter_test)

        dataset_utter = DatasetDict()
        dataset_utter['train'] = ds_train
        dataset_utter['test'] = ds_test

    except mysqlconn.Error as error:
        print("database error occured: {}".format(error))
    
    return dataset_utter
# load_utterances()


# Embed user text
def cls_pooling(model_output):
    # Extract the token embeddings
    return model_output.last_hidden_state[:, 0]


def embed_text(text_list):
    encoded_input = globals_nlp.tokenizer_xlmr_banking(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )

    encoded_input = {k: v.to(globals_nlp.device) for k, v in encoded_input.items()}
    model_output = globals_nlp.model_xlmr_banking(**encoded_input)
    return cls_pooling(model_output)


def embed_all_examples(dataset):
    print("Embedding all examples ...")
    globals_nlp.tokenizer_xlmr_banking.pad_token = globals_nlp.tokenizer_xlmr_banking.eos_token

    embs_train_user = dataset["train"].map(
        lambda x: {"embedding": embed_text(x["text"]).detach().cpu().numpy()[0]}
    )
    embs_valid_user = dataset["test"].map(
        lambda x: {"embedding": embed_text(x["text"]).detach().cpu().numpy()[0]}
    )

    return embs_train_user


def save_model(embedding_user):
    print("Saving the embeddings")
    embedding_user.save_to_disk("models/embedding_user/")


# Training a simple classifier
def train_classifier():
    print("training started ...")
    dataset_utter = load_utterances()
    embs_train_user = embed_all_examples(dataset_utter)
    save_model(embs_train_user)


# globals_nlp.init()
# train_classifier()
