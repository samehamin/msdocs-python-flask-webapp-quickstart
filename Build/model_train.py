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


dataset_utter = None

def load_utterances():
    print("loading utterances...")
    global dataset_utter

    try:
        # load utterances from db
        #TODO: create one table for train/test
        dataset_utter_train = mysqlconn.read_df_sqlalchemy("SELECT * FROM user_utterances_train;")
        dataset_utter_test = mysqlconn.read_df_sqlalchemy("SELECT * FROM user_utterances_test;")
        dataset_chitchat_train = mysqlconn.read_df_sqlalchemy("SELECT * FROM chitchat_utterances_train;")
        dataset_chitchat_test = mysqlconn.read_df_sqlalchemy("SELECT * FROM chitchat_utterances_test;")
        dataset_banking_train = mysqlconn.read_df_sqlalchemy("SELECT * FROM banking_utterances_train;")
        dataset_banking_test = mysqlconn.read_df_sqlalchemy("SELECT * FROM banking_utterances_test;")

        # concat train and test frames
        train_frames = [dataset_utter_train, dataset_chitchat_train, dataset_banking_train]
        test_frames = [dataset_utter_test, dataset_chitchat_test, dataset_banking_test]
        dataset_utter_train = pd.concat(train_frames)
        dataset_utter_test = pd.concat(test_frames)

        # convert dataframes into dataset
        ds_train = Dataset.from_pandas(dataset_utter_train)
        ds_test = Dataset.from_pandas(dataset_utter_test)

        dataset_utter = DatasetDict()
        dataset_utter['train'] = ds_train
        dataset_utter['test'] = ds_test

    except mysqlconn.Error as error:
        print("database error occured: {}".format(error))
load_utterances()
print(dataset_utter["test"].shape)
print(dataset_utter)


# Tokenize the whole dataset
def tokenize(batch):
    return globals_nlp.tokenizer_xlmr_base(batch["text"], padding=True, truncation=True)


def tokenize_dataset():
    try:
        print("tokenizing the dataset ...")
        # applying the extract_hidden_states() function has added a new hidden_state
        dataset_encoded = dataset_utter.map(tokenize, batched=True)

        return dataset_encoded

    except Exception as e:
        print("Error: ", e)
# tokenize_dataset()


#  we’ll use the map() method of DatasetDict to extract all the hidden states in one go
def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(globals_nlp.device) for k,v in batch.items()
                if k in globals_nlp.tokenizer_xlmr_base.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = globals_nlp.model_xlmr_base(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


def extract_hidden_states_dataset():
    print("Extracting hidden states...")
    
    dataset_encoded = tokenize_dataset()

    #  convert the input_ids and attention_mask columns to the "torch" format
    dataset_encoded.set_format("torch", columns=["input_ids", 
                                              "attention_mask", "label"])
    dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True)

    return dataset_hidden
# extract_hidden_states_dataset()


def create_feature_matrix():
    print("Creating feature matrix...")
    
    dataset_hidden = extract_hidden_states_dataset()

    X_train = np.array(dataset_hidden["train"]["hidden_state"])
    y_train = np.array(dataset_hidden["train"]["label"])
    X_valid = np.array(dataset_hidden["test"]["hidden_state"])
    y_valid = np.array(dataset_hidden["test"]["label"])

    return X_train, y_train, X_valid, y_valid


def save_model(clf):
    print("Saving the model ...")
    
    import pickle
    filename = 'Data/UDModel.sav'
    pickle.dump(clf, open(filename, 'wb'))


# Training a simple classifier
def train_classifier():
    X_train, y_train, X_valid, y_valid = create_feature_matrix()

    print("Training the model...")
    #Linear Support Vector Machine
    svm = make_pipeline(StandardScaler(),
                     LinearSVC(random_state=0, tol=1e-5, max_iter=3000))

    clf_svm = CalibratedClassifierCV(svm)
    clf_svm.fit(X_train, y_train)
    
    score = clf_svm.score(X_valid, y_valid)
    print(score)

    save_model(clf_svm)


# globals_nlp.init()
# train_classifier()